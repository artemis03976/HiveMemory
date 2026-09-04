---
title: Perception Topic Buffer Boundary Refactor
status: active
owner: patchouli
target: post-p7-perception-boundary-cleanup
scope: short-term-storage-topic-lifecycle-working-set-and-perception-engine-cleanup
code_paths:
  - src/hivememory/patchouli/memory_library/stores.py
  - src/hivememory/patchouli/memory_library/ports.py
  - src/hivememory/patchouli/memory_library/adapters/short_term.py
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/engines/perception/
  - src/hivememory/core/models/topic.py
  - src/hivememory/system/config/patchouli.py
updates:
  - docs/patchouli/perception.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/README.md
  - docs/todo/short-term-memory-store-boundary-cleanup.md
related_docs:
  - docs/todo/short-term-memory-store-boundary-cleanup.md
  - docs/architecture/decisions/0001-data-model-mutability-and-boundary-projection.md
  - docs/architecture/decisions/0002-unique-identities-and-minimal-concurrency.md
last_reviewed: 2026-09-04
---

# Perception Topic Buffer 边界重组计划（第三版）

## 0. 执行摘要

### 当前状况

经过两轮边界清理后，系统陷入了结构性困境：

1. `TopicBufferService`（730 行）同时承担存储代理、工作集管理、状态机、触发矩阵和跨系统编排，成为职责不清的混合体；
2. `ShortTermMemoryStore` 被收缩为纯 CRUD 后，短期存储自身的一致性机制（Pool/Buffer 联动、全局 ID 唯一性、访问追踪）无处安放；
3. `engines/perception/` 仍持有 Topic 路由、retry、状态机逻辑，"无状态 Engine"名不副实；
4. LRU、idle 候选查询通过 O(n) 遍历 Store + 状态过滤实现，没有独立的工作集索引。

### 根本原因

三个范畴错误导致职责无法合理分配：

1. **锁被建模成了记录字段**（`BufferState.PROCESSING/SETTLING`）：
   - 跨 await 的占用权通过记录状态 + 后续校验实现；
   - Relay 在锁外执行，回来后需比对 `expected_state` 和 `fold_prefix` 确认没人改过；
   - 导致状态机逻辑散布在 Store、Service、Familiar 三处，没有清晰的所有者。

2. **工作集策略被建模成了存储**：
   - LRU、驻留容量、候选查询是**驻留工作集（Working Set）** 的职责，不是持久化的职责；
   - 但因为它们需要"遍历所有 Topic"，被错误地放进了 Store；
   - 类比：把 OS 的页表逻辑放进磁盘驱动，因为"页表需要知道哪些页在磁盘上"。

3. **有状态 runtime 混在无状态算法层**：
   - `engines/perception/` 持有 Store、Journal、Patchouli 错误；
   - 实现 Topic 路由、retry、状态预约——这些是 runtime 职责，不是算法；
   - 导致 Engine 无法独立测试，也无法被其他 runtime 复用。

### 解决方案

**不是重新分配职责，而是纠正三个范畴错误**：

1. **锁 → lease 表**：跨 await 占用权通过进程内 lease 表管理，不再作为记录字段；
2. **工作集 → TopicWorkingSet**：独立的驻留管理器，持有 `OrderedDict`（LRU）+ lease 表，零外部依赖；
3. **Perception → 拆成两层**：
   - `MemoryPerceptionEngine`（`engines/`）：纯算法（block 构造、token 估算、folding 判断）；
   - `PerceptionFamiliar`（`patchouli/services/`）：持有 WorkingSet、Store、Engine，负责编排、Journal、生命周期。

### 目标架构

```text
PerceptionFamiliar (350 行)
├─ TopicWorkingSet (120 行，零依赖)
│   ├─ 驻留追踪（OrderedDict，LRU）
│   ├─ Lease 表（跨 await 占用权）
│   └─ 候选查询（idle / LRU / shutdown）
│
├─ ShortTermMemoryStore (80 行)
│   ├─ Port 封装（get/put/delete/list）
│   └─ 全局 ID 唯一性检查
│
└─ MemoryPerceptionEngine (150 行)
    ├─ InteractionPayload → LogicalBlock
    ├─ TurnEvent → ActionReducer
    └─ 纯 folding 算法
```

- **删除** `TopicBufferService`、`TriggerManager`、`SemanticBuffer`、`BasePerceptionLayer`、`NullPerceptionLayer`；
- **删除** `TRIGGER_PLANS` 决策矩阵，换成 3 个具名用例（`settle_topic` / `compact_topic` / `evict_topic`）；
- **删除** `BufferState` 枚举，状态机通过 lease 表实现；
- **保留** `TriggerReason` 作为 provenance 标签（记录"为什么触发"），但不再驱动分支逻辑。

### 收益

| 指标 | 旧方案 | 新方案 | 改善 |
|:---|---:|---:|:---|
| **总行数** | 1180 行 | 700 行 | -40% |
| **LRU 查询** | O(n) 遍历 | O(1) 尾部 | 性能提升 |
| **并发保护** | 2 把锁 + 补偿校验 | 1 把锁 + lease 表 | 简化 |
| **测试隔离** | 必须 mock Store + Relay | WorkingSet 可纯单元测试 | 可测试性提升 |
| **循环依赖** | 存在（Engine → Patchouli） | 消除 | 架构清晰 |

---

## 1. 背景：为什么前两轮重构失败

### 1.1 最初的困境（f5d56b7 之前）

todo 的初衷是"让 `ShortTermMemoryStore` 回归纯 CRUD"，当时的分工：

```
ShortTermMemoryStore (685 行)
├─ Port 封装（get/put/delete/list）
├─ 状态机预约（reserve/release/commit/abort × 2 套）
├─ 命名写操作（add_block/clear_blocks/apply_compaction/apply_interaction）
├─ LRU 候选查询（get_lru_candidate_by_key）
└─ 原子操作（freeze_and_evict/freeze_for_manual_settle）

TriggerManager (453 行，在 engines/perception/)
├─ 统一调度（resolve_topic）
├─ 决策矩阵（DECISION_MATRIX: 7 reasons → 3 booleans）
├─ 三条执行路径（settle_and_evict / prepare+commit+abort_manual_settle / _compact_path）
└─ 调用 Relay 生成摘要
```

**核心问题**：TriggerManager 直接调用 Store 的"高级方法"（`freeze_and_evict` / `freeze_for_manual_settle`），这些方法不是 CRUD，而是**组合操作**（冻结 + 修改状态 + 可能驱逐），**只为 TriggerManager 服务，从不被其他人直接调用**。

这暴露了一个矛盾：
- TriggerManager 在 `engines/perception/`，不能 `import patchouli.errors.TopicBusyError` 或操作 `self._port`（循环依赖）；
- 状态机必须在锁内原子执行（freeze + evict 必须在同一临界区）；
- 妥协方案：TriggerManager 调用 Store 的"私有实现"，这些实现在 Store 里作为公开方法暴露。

**结果**：Store 的 API 里充斥着"只有一个调用者"的方法，职责边界模糊。

### 1.2 第一次清理（f5d56b7）

删掉 TriggerManager，创建 `TopicBufferService`（在 `patchouli/services/`，可以依赖 Store 和错误）：

```
TopicBufferService (730 行)
├─ 持有 Store（转发 get/put/delete/create）
├─ 持有 Relay（生成摘要）
├─ 决策矩阵（TRIGGER_PLANS）
├─ 状态机（SettlementReservation、各种 begin/complete/abort）
├─ Pool 查询（list_idle_candidates / select_lru_candidate）
├─ 统一调度（handle_trigger）
└─ 原子操作（apply_interaction）
```

**新问题更严重**：

1. **既是 Store 代理，又是 Store 调用者**：
   - Familiar 调 `service.apply_interaction(...)`；
   - Service 内部调 `self._store.put(snapshot)`；
   - 但 Service 同时还提供 `get/list/count`，只是转发给 Store。
   
   调用者不知道该直接调 Store 还是调 Service。

2. **状态机仍在记录里**（`BufferState.PROCESSING / SETTLING`）：
   - `begin_settlement` 返回 `SettlementReservation`，调用者必须在 `finally` 里调 `complete/abort`；
   - Relay 在锁外执行，回来后需比对 `expected_state` 和 `fold_prefix`；
   - `list_idle_candidates` 要过滤 `state is IDLE`。

3. **WorkingSet 逻辑仍通过遍历 Store 实现**：
   ```python
   topics = self._store.list_by_workspace(scope)
   return tuple(
       TopicCandidate(...) for topic in topics
       if topic.state is IDLE and topic.is_idle(timeout)
   )
   ```
   **每次查候选都要全量遍历 Store**——这不是工作集管理，这是用 O(n) 查询模拟索引。

### 1.3 第二次清理（14b4f53，当前状态）

把 `TriggerReason` / `TopicMaterializeTask` 从 Engine 模型移出，统一 settle 时序（admission-before-evict），补充测试。但**核心问题未解决**：

- `TopicBufferService` 仍是 730 行的混合体；
- 状态机仍是记录字段 + 补偿校验；
- WorkingSet 仍是 O(n) 遍历；
- Engine 仍持有 Store、Journal、Patchouli 错误。

无论怎么搬家，**总有一个组件被迫承担错误边界的逻辑**。

---

## 2. 根本原因：三个范畴错误

### 2.1 错误一：锁被建模成了记录字段

**表现**：`BufferState.PROCESSING / SETTLING` 作为 `TopicData.state` 字段，用来表示"这个 Topic 正在被谁操作"。

**问题**：

1. **跨 await 占用权通过记录状态实现**：
   ```python
   # Interaction 路径
   self._store.reserve_processing(scope, topic_id)  # IDLE -> PROCESSING
   try:
       self._store.apply_interaction(...)
       await self._maybe_fold_pages(...)  # 锁外 Relay
   finally:
       self._store.release_processing(scope, topic_id)  # PROCESSING -> IDLE
   ```
   如果 Relay 执行时有另一个线程尝试读取 Topic，它会看到 `state=PROCESSING`——这不是"Topic 的业务状态"，而是**临时的执行锁**。

2. **状态校验散布在多处**：
   - Store 的 `apply_interaction` 检查 `state is PROCESSING`；
   - Service 的 `begin_settlement` 检查 `state is IDLE`；
   - Service 的 `_execute_compact` 校验 `current.state is expected_state`；
   - Familiar 的 `_settle_candidate` 检查 `reservation is not None`。
   
   没有单一的"占用表"所有者，每个组件都自己检查状态。

3. **补偿逻辑复杂**：
   ```python
   # topic_buffer.py:646
   if (current is None or current.state is not expected_state
       or tuple(current.blocks[:fold_count]) != fold_prefix):
       # 预约失效，放弃写回
       return TriggerExecution(...)
   ```
   因为锁在记录里，锁外操作完成后必须重新检查"记录还是我当初锁的那个吗"。

**正确建模**：占用权是**进程内状态**，不是记录属性。应该用 lease 表：

```python
class TopicWorkingSet:
    def __init__(self):
        self._leases: Dict[tuple[IdentityScope, str], LeaseToken] = {}
    
    def acquire(self, scope: IdentityScope, topic_id: str) -> LeaseToken | None:
        key = (scope, topic_id)
        if key in self._leases:
            return None  # 已被占用
        lease = LeaseToken(scope, topic_id, acquired_at=time.time())
        self._leases[key] = lease
        return lease
    
    def release(self, lease: LeaseToken) -> None:
        self._leases.pop((lease.scope, lease.topic_id), None)
```

调用者持有 `LeaseToken`，在 `finally` 里 `release(lease)`。Store 返回的 `TopicData` 不再有 `state` 字段——读取者看到的是**业务快照**，不是执行锁。

### 2.2 错误二：工作集策略被建模成了存储

**表现**：LRU、驻留容量、候选查询的逻辑在 `ShortTermMemoryStore` 或 `TopicBufferService` 里，通过遍历所有 Topic 实现。

**问题**：

1. **LRU 不是存储能力**：
   - 中期存储（向量库）不需要 LRU——它的容量是磁盘，不是"活跃话题池"；
   - 长期存储（冷存储）也不需要 LRU；
   - LRU 是**驻留工作集（Working Set）** 的策略，不是持久化的职责。

2. **每次查询都是 O(n)**：
   ```python
   # 当前实现（topic_buffer.py:297）
   topics = [topic for topic in self._store.list_by_workspace(scope)
             if topic.state is IDLE]
   return min(topics, key=lambda t: t.last_accessed_at).topic_id
   ```
   查 LRU 候选需要遍历所有 Topic、过滤状态、找 min——这应该是 O(1) 的 `OrderedDict.popitem(last=False)`。

3. **访问追踪职责混乱**：
   - Store 的 `get(..., touch=True)` 更新 `last_accessed_at`；
   - Familiar 调 `touch_topic(...)` 又更新一次；
   - 但"谁在驻留集合里"没有独立索引，每次都要遍历 Store。

**正确建模**：驻留工作集是独立的管理器，持有 `OrderedDict` 和 lease 表：

```python
class TopicWorkingSet:
    def __init__(self, max_resident: int = 5):
        self._max_resident = max_resident
        self._resident: OrderedDict[tuple[IdentityScope, str], float] = OrderedDict()
        self._leases: Dict[tuple[IdentityScope, str], LeaseToken] = {}
    
    def touch(self, scope: IdentityScope, topic_id: str) -> None:
        key = (scope, topic_id)
        self._resident[key] = time.time()
        self._resident.move_to_end(key)  # LRU: 移到末尾
    
    def select_lru_candidate(self, scope: IdentityScope) -> str | None:
        for (candidate_scope, topic_id) in self._resident:
            if candidate_scope.workspace != scope.workspace:
                continue
            if (candidate_scope, topic_id) in self._leases:
                continue  # 跳过正在占用的
            return topic_id
        return None  # O(k)，k = 驻留数量（通常 ≤ 5）
```

Store 不再持有 `last_accessed_at` 和 `max_resident_topics`——那些是 WorkingSet 的内部状态。

### 2.3 错误三：有状态 runtime 混在无状态算法层

**表现**：`engines/perception/semantic_flow_perception_layer.py` 持有：
- `self._topic_buffer: TopicBufferService`（Store 状态）；
- `self._interaction_journal: InMemoryInteractionApplyJournal`（重试状态）；
- `self.config: SemanticFlowPerceptionConfig`（runtime 配置）；
- 实现 `route_and_ingest` / `prepare_topic`（Topic 用例）。

**问题**：

1. **Engine 不是"引擎"**：
   - 它持有外部状态（Store、Journal）；
   - 它实现业务流程（路由、retry、状态预约）；
   - 它不是可复用的算法组件，而是 Patchouli 专用的 Perception Service。

2. **循环依赖**：
   - Engine 在 `engines/` 里，却 `import hivememory.patchouli.services.topic_buffer`；
   - Engine 使用 `TopicBusyError`，这是 `patchouli.errors` 的领域错误；
   - 这让 `engines/` 不再是"可被其他 runtime 复用的算法层"。

3. **测试困难**：
   - 测试 "block 构造逻辑" 需要 mock Store、Journal、TopicBufferService；
   - 无法独立测试"token 估算"、"折叠判断"——它们和 Topic 用例绑死了。

**正确建模**：拆成两层：

```python
# engines/perception/memory_perception_engine.py（无状态）
class MemoryPerceptionEngine:
    def build_block(self, payload: InteractionPayload, scope: IdentityScope) -> LogicalBlock:
        """纯函数：payload → block"""
        actions = ActionReducer.reduce(payload.turn_events)
        total_tokens = estimate_tokens(...)
        return LogicalBlock(turn=..., total_tokens=total_tokens, ...)
    
    def should_compact(self, total_tokens: int, threshold: int) -> bool:
        """纯函数：判断是否需要 compact"""
        return total_tokens > threshold

# patchouli/services/perception.py（有状态 runtime）
class PerceptionFamiliar:
    def __init__(self, *, engine, store, working_set, journal, ...):
        self._engine = engine  # 纯算法
        self._store = store  # 状态
        self._working_set = working_set  # 工作集
        self._journal = journal  # 重试
    
    async def apply_interaction(self, payload, ...) -> str:
        """编排 engine + store + working_set + journal"""
        block = self._engine.build_block(payload, scope)
        lease = self._working_set.acquire(scope, topic_id)
        try:
            self._store.put(topic.model_copy(update={"blocks": (*topic.blocks, block)}))
            if self._engine.should_compact(topic.total_tokens, threshold):
                await self._compact_topic(scope, topic_id, lease)
        finally:
            self._working_set.release(lease)
```

Engine 回归纯算法，Familiar 持有状态并编排。

---

## 3. 目标架构

### 3.1 四组件分层

```text
PerceptionFamiliar (350 行，patchouli/services/perception.py)
├─ 路由与当前 Topic 选择
├─ InteractionApplyJournal / retry
├─ Relay 与 Generation admission
├─ 用户结果、maintenance、shutdown 报告
└─ 编排下面三个组件

TopicWorkingSet (120 行，新增，零依赖)
├─ 驻留追踪（OrderedDict，LRU）
├─ Lease 表（跨 await 占用权）
├─ 候选查询（idle / LRU / shutdown）
└─ 容量判断（needs_eviction）

ShortTermMemoryStore (80 行，patchouli/memory_library/stores.py)
├─ Port 封装（get/put/delete/list）
└─ 全局 ID 唯一性检查

MemoryPerceptionEngine (150 行，engines/perception/memory_perception_engine.py)
├─ InteractionPayload → LogicalBlock
├─ TurnEvent → ActionReducer
├─ token estimate
└─ 纯 folding / 摘要算法能力
```

**删除的组件**：
- `TopicBufferService`（730 行）；
- `SemanticBuffer`（90 行）；
- `BasePerceptionLayer` / `NullPerceptionLayer`（85 行）；
- `TriggerManager`（已在 14b4f53 删除，但逻辑仍在 TopicBufferService 里）。

**保留但改变职责**：
- `TriggerReason`：从"驱动分支逻辑的枚举"变成"provenance 标签"（记录"为什么触发"，但不影响执行路径）。

### 3.2 TopicWorkingSet：独立的驻留管理器

```python
@dataclass(frozen=True)
class LeaseToken:
    """Topic 占用权凭证，调用者在 finally 里 release。"""
    scope: IdentityScope
    topic_id: str
    acquired_at: float

class TopicWorkingSet:
    """短期话题的驻留工作集与占用表。
    
    职责：
    - 维护有限容量的驻留集合（max_resident_topics）
    - 追踪访问顺序（LRU）
    - 提供候选查询（idle / LRU / shutdown）
    - 管理 Topic 占用状态（lease 机制）
    
    不职责：
    - 不持有 TopicData 内容（内容在 Store 里）
    - 不调用 Store/Relay/Queue（纯内存逻辑）
    - 不解释触发原因（调用者决定）
    """
    
    def __init__(self, max_resident: int = 5):
        self._max_resident = max_resident
        # (scope, topic_id) → last_access_time
        self._resident: OrderedDict[tuple[IdentityScope, str], float] = OrderedDict()
        # (scope, topic_id) → LeaseToken
        self._leases: Dict[tuple[IdentityScope, str], LeaseToken] = {}
    
    def touch(self, scope: IdentityScope, topic_id: str) -> None:
        """标记访问，更新 LRU 顺序"""
        key = (scope, topic_id)
        self._resident[key] = time.time()
        self._resident.move_to_end(key)  # LRU: 移到末尾
    
    def needs_eviction(self, scope: IdentityScope) -> bool:
        """判断是否需要为该 Workspace 驱逐 Topic"""
        workspace_topics = [k for k in self._resident 
                           if k[0].workspace_identity == scope.workspace_identity]
        return len(workspace_topics) >= self._max_resident
    
    def select_lru_candidate(
        self, 
        scope: IdentityScope, 
        *, 
        exclude: set[str] = frozenset()
    ) -> str | None:
        """选择同 Workspace 内最久未访问的、未被占用的 Topic"""
        for (candidate_scope, topic_id) in self._resident:
            if candidate_scope.workspace_identity != scope.workspace_identity:
                continue
            if topic_id in exclude:
                continue
            if (candidate_scope, topic_id) in self._leases:
                continue  # 跳过正在被占用的
            return topic_id
        return None
    
    def list_idle_candidates(
        self, 
        timeout_seconds: int, 
        *, 
        now: float = None
    ) -> list[tuple[IdentityScope, str]]:
        """返回所有超时且未被占用的 Topic"""
        now = now or time.time()
        return [
            (scope, topic_id)
            for (scope, topic_id), last_access in self._resident.items()
            if (now - last_access) > timeout_seconds
            and (scope, topic_id) not in self._leases
        ]
    
    def list_shutdown_candidates(self) -> list[tuple[IdentityScope, str]]:
        """返回所有驻留 Topic（shutdown 时全部清理）"""
        return list(self._resident.keys())
    
    def acquire(self, scope: IdentityScope, topic_id: str) -> LeaseToken | None:
        """非阻塞获取占用权；已被占用时返回 None"""
        key = (scope, topic_id)
        if key in self._leases:
            return None
        lease = LeaseToken(scope, topic_id, acquired_at=time.time())
        self._leases[key] = lease
        return lease
    
    def release(self, lease: LeaseToken) -> None:
        """释放占用权"""
        key = (lease.scope, lease.topic_id)
        self._leases.pop(key, None)
    
    def remove(self, scope: IdentityScope, topic_id: str) -> None:
        """从驻留集合移除（evict 后调用）"""
        self._resident.pop((scope, topic_id), None)
        # lease 不自动清理——如果持有者还在操作，让它完成后 release
```

**关键特性**：

1. **零外部依赖**：可以纯单元测试，不需要 mock Store 或 Relay；
2. **O(1) LRU 查询**：`move_to_end` + `popitem(last=False)`；
3. **lease 明确所有权**：调用者持有 `LeaseToken`，在 `finally` 里 `release`，不再通过记录状态表达占用；
4. **与 Store 解耦**：Store 不再知道"什么是 LRU"，WorkingSet 不知道"Store 怎么索引"。

### 3.3 ShortTermMemoryStore：回归纯 CRUD

```python
class ShortTermMemoryStore:
    """短期 Topic 持久化 facade。
    
    职责：
    - 封装 Port（get/put/delete/list）
    - 全局 ID 唯一性检查
    - 返回不可变 TopicData 快照
    
    不职责：
    - 不持有 max_resident_topics（那是 WorkingSet 的配置）
    - 不持有 last_accessed_at（那是 WorkingSet 的索引）
    - 不持有 BufferState（lease 在 WorkingSet 里）
    - 不调用 Relay/Queue（那是 Familiar 的编排）
    """
    
    def __init__(self, port: ShortTermStoragePort | None = None):
        self._port = port or InMemoryShortTermStorage()
        self._lock = threading.RLock()  # 保护 Port 操作和索引一致性
    
    def get(
        self, 
        scope: IdentityScope, 
        topic_id: str
    ) -> TopicData | None:
        """读取不可变快照；不再有 touch 参数（访问追踪在 WorkingSet 里）"""
        scope = require_identity_scope(scope)
        with self._lock:
            return self._port.get(scope.workspace_identity, topic_id)
    
    def put(self, topic: TopicData) -> None:
        """写入或替换 Topic；全局 ID 唯一性检查在 Port 内部"""
        if not isinstance(topic, TopicData):
            raise TypeError("short-term store accepts TopicData snapshots")
        with self._lock:
            self._port.put(topic)
    
    def create(
        self, 
        scope: IdentityScope, 
        *, 
        topic_title: str = "新建话题", 
        topic_summary: str = ""
    ) -> TopicData:
        """创建新 Topic 并返回快照"""
        scope = require_identity_scope(scope)
        now = datetime.now().timestamp()
        topic = TopicData(
            topic_id=str(uuid4()),
            workspace_identity=scope.workspace_identity,
            topic_title=topic_title,
            topic_summary=topic_summary,
            last_update=now,
        )
        self.put(topic)
        return topic
    
    def delete(self, scope: IdentityScope, topic_id: str) -> bool:
        """删除 Topic；不再检查 BufferState（lease 在 WorkingSet 里检查）"""
        scope = require_identity_scope(scope)
        with self._lock:
            return self._port.delete(scope.workspace_identity, topic_id)
    
    def list_by_workspace(
        self, 
        scope: IdentityScope, 
        *, 
        include_empty: bool = True
    ) -> list[TopicData]:
        """列出 Workspace 内所有 Topic"""
        scope = require_identity_scope(scope)
        with self._lock:
            topics = self._port.list_by_workspace(scope.workspace_identity)
            if not include_empty:
                topics = [t for t in topics if t.has_content]
            return topics
    
    def count(self, scope: IdentityScope) -> int:
        """统计 Workspace 内 Topic 数量"""
        scope = require_identity_scope(scope)
        with self._lock:
            return self._port.count(scope.workspace_identity)
```

**关键变化**：

1. **删除 `touch` 参数**：访问追踪在 `WorkingSet.touch()` 里；
2. **删除 `max_resident_topics`**：容量限制在 `WorkingSet.needs_eviction()` 里；
3. **删除 `last_accessed_at` 字段**：`TopicData` 不再有这个字段；
4. **删除所有状态机方法**：`reserve/release/commit/abort` 全删，换成 WorkingSet 的 lease；
5. **删除 `apply_interaction` 等组合方法**：Store 只提供 `put`，组合逻辑在 Familiar 里。

### 3.4 MemoryPerceptionEngine：纯算法

```python
class MemoryPerceptionEngine:
    """无状态的短期记忆摄入算法。
    
    职责：
    - InteractionPayload → LogicalBlock
    - TurnEvent → ActionReducer / TraceReducer
    - token 估算
    - 纯 folding / 摘要算法能力
    
    不职责：
    - 不持有 Store / Journal / Queue
    - 不实现 route_and_ingest / prepare_topic（那是 Familiar 的用例）
    - 不调用 Patchouli errors（TopicBusyError 等）
    """
    
    def __init__(self, config: SemanticFlowPerceptionConfig):
        self.config = config
    
    def build_block(
        self, 
        payload: InteractionPayload, 
        scope: IdentityScope
    ) -> LogicalBlock:
        """纯函数：InteractionPayload → LogicalBlock"""
        actions = ActionReducer.reduce(payload.turn_events)
        traces = payload.mtp_traces
        turn = TurnRecord(
            identity=scope.actor_identity,
            user_query=payload.user_message,
            rewritten_query=payload.rewritten_query,
            assistant_final_text=payload.assistant_final_text or "",
            turn_events=payload.turn_events,
            actions=actions,
            semantic_traces=traces,
        )
        total_tokens = (
            estimate_tokens(turn.user_query)
            + estimate_tokens(turn.assistant_final_text)
            + sum(estimate_tokens(trace.query or "") + estimate_tokens(trace.target or "")
                  for trace in turn.semantic_traces)
        )
        return LogicalBlock(
            turn=turn,
            total_tokens=total_tokens,
            worth_saving=payload.worth_saving,
            gateway_intent=payload.gateway_intent,
        )
    
    def should_compact(self, total_tokens: int) -> bool:
        """纯函数：判断是否需要 compact"""
        return total_tokens > self.config.fold_token_threshold
    
    def select_blocks_to_fold(
        self, 
        blocks: list[LogicalBlock], 
        retain_recent: int
    ) -> list[LogicalBlock]:
        """纯函数：选择需要折叠的 blocks"""
        if retain_recent < 1:
            raise ValueError("retain_recent must be >= 1")
        if len(blocks) <= retain_recent:
            return []
        return blocks[:-retain_recent]
```

**关键特性**：

1. **所有方法都是纯函数或无副作用方法**；
2. **不依赖 Patchouli**：可以被其他 runtime 复用；
3. **可独立测试**：不需要 mock Store、Journal、Queue。

### 3.5 PerceptionFamiliar：编排器

```python
class PerceptionFamiliar:
    """感知业务门面，负责摄入与短期话题管理。
    
    职责：
    - 路由与当前 Topic 选择
    - InteractionApplyJournal / retry
    - 编排 Engine + Store + WorkingSet
    - Relay 与 Generation admission
    - 用户结果、maintenance、shutdown 报告
    """
    
    def __init__(
        self, 
        *, 
        engine: MemoryPerceptionEngine,
        store: ShortTermMemoryStore,
        working_set: TopicWorkingSet,
        relay_controller: BaseRelayController,
        bus,
        config: MemoryPerceptionConfig,
        interaction_journal: InMemoryInteractionApplyJournal,
    ):
        self._engine = engine
        self._store = store
        self._working_set = working_set
        self._relay = relay_controller
        self._bus = bus
        self._config = config
        self._journal = interaction_journal
        self._last_active_topic_ids: dict[tuple[str, str], str] = {}
    
    async def apply_interaction(
        self, 
        payload: InteractionPayload, 
        *, 
        identity_scope: IdentityScope,
        target_topic_id: str = "NEW_TOPIC",
        interaction_id: str | None = None,
        asset_id_and_refs: tuple = (),
    ) -> str:
        """应用交互载荷并完成话题路由（核心用例）"""
        # 1. Journal 查重
        if interaction_id:
            apply_record = self._journal.get(interaction_id)
            if apply_record is not None:
                # retry 路径...
                pass
        
        # 2. LRU 驱逐（需要时）
        await self._maybe_evict_lru(identity_scope, target_topic_id)
        
        # 3. 路由到目标 Topic
        topic_id = await self._route_to_topic(
            target_topic_id, identity_scope, payload
        )
        
        # 4. 获取 lease
        lease = self._working_set.acquire(identity_scope, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，可稍后重试")
        
        try:
            # 5. 构造 block（纯算法）
            block = self._engine.build_block(payload, identity_scope)
            
            # 6. 写入 Store（原子操作）
            topic = self._store.get(identity_scope, topic_id)
            if topic is None:
                raise KeyError(f"topic '{topic_id}' not found")
            
            # 原子写入：blocks + bindings + metadata
            updated = topic.model_copy(update={
                "blocks": (*topic.blocks, block),
                "bindings": self._merge_bindings(
                    topic.bindings, asset_id_and_refs, interaction_id
                ),
                "total_tokens": topic.total_tokens + block.total_tokens,
                "last_update": datetime.now().timestamp(),
                "model_used": payload.model_used or topic.model_used,
            })
            self._store.put(updated)
            
            # 7. Journal 记录
            if interaction_id:
                self._journal.record_interaction_applied(
                    interaction_id, topic_id, digest
                )
            
            # 8. 检查是否需要 compact
            if self._engine.should_compact(updated.total_tokens):
                await self._compact_topic(identity_scope, topic_id, lease)
        
        finally:
            # 9. 释放 lease
            self._working_set.release(lease)
        
        # 10. 更新 WorkingSet 和 last_active
        self._working_set.touch(identity_scope, topic_id)
        workspace = identity_scope.workspace_identity
        self._last_active_topic_ids[
            (workspace.owner_user_id, workspace.workspace_id)
        ] = topic_id
        
        return topic_id
    
    async def _maybe_evict_lru(
        self, 
        scope: IdentityScope, 
        target_topic_id: str
    ) -> None:
        """需要创建新话题且池满时，驱逐 LRU 话题"""
        if target_topic_id != "NEW_TOPIC":
            # 已有话题无需驱逐
            if self._store.get(scope, target_topic_id) is None:
                raise KeyError(f"topic '{target_topic_id}' does not exist")
            return
        
        if not self._working_set.needs_eviction(scope):
            return
        
        attempted = set()
        while self._working_set.needs_eviction(scope):
            lru_id = self._working_set.select_lru_candidate(scope, exclude=attempted)
            if lru_id is None:
                raise TopicBusyError("LRU 驱逐无可用候选，稍后重试")
            
            try:
                await self._settle_topic(scope, lru_id, reason=TriggerReason.LRU_EVICTION)
                return  # 成功驱逐
            except TopicBusyError:
                attempted.add(lru_id)
                continue
    
    async def _settle_topic(
        self, 
        scope: IdentityScope, 
        topic_id: str, 
        *, 
        reason: TriggerReason
    ) -> tuple[bool, MemoryGenerationTask | None]:
        """统一 settle 时序：获取 lease → 冻结材料 → admission → 删除"""
        # 1. 获取 lease
        lease = self._working_set.acquire(scope, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙")
        
        try:
            # 2. 冻结材料
            topic = self._store.get(scope, topic_id)
            if topic is None or topic.is_empty:
                return False, None
            
            task = TopicMaterializeTask.from_topic_data(
                topic, identity_scope=scope, reason=reason
            )
            if task is None:
                # 无可保存 block，正常 skip
                self._store.delete(scope, topic_id)
                self._working_set.remove(scope, topic_id)
                return True, None
            
            # 3. Admission（锁外）
            generation_task = await self._bus.request(
                PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, task
            )
            
            # 4. 删除 Topic
            self._store.delete(scope, topic_id)
            self._working_set.remove(scope, topic_id)
            return True, generation_task
        
        except Exception:
            # admission 失败，lease 在 finally 里释放，Topic 保留
            raise
        
        finally:
            self._working_set.release(lease)
    
    async def _compact_topic(
        self, 
        scope: IdentityScope, 
        topic_id: str, 
        lease: LeaseToken
    ) -> None:
        """执行 compact：持有 lease 前提下，锁外生成摘要，写回"""
        # 调用者已持有 lease，不需要重新 acquire
        topic = self._store.get(scope, topic_id)
        if topic is None or topic.is_empty:
            return
        
        blocks_to_fold = self._engine.select_blocks_to_fold(
            list(topic.blocks), 
            retain_recent=self._config.fold_retain_recent_blocks
        )
        if not blocks_to_fold:
            return
        
        # 锁外生成摘要
        summary = self._relay.generate_summary(
            blocks_to_fold=blocks_to_fold,
            previous_summary=topic.state_summary,
        )
        
        # 写回
        retained = topic.blocks[len(blocks_to_fold):]
        updated = topic.model_copy(update={
            "state_summary": summary,
            "blocks": retained,
            "total_tokens": sum(b.total_tokens for b in retained),
            "last_update": datetime.now().timestamp(),
        })
        self._store.put(updated)
    
    async def manual_settle_topic(
        self, 
        scope: IdentityScope, 
        topic_id: str | None = None
    ) -> TopicSettleResult:
        """手动结算指定话题"""
        target_id = topic_id or self._last_active_topic_ids.get(...)
        if not target_id:
            raise ValueError("未指定 topic_id 且无活跃话题")
        
        try:
            completed, task = await self._settle_topic(
                scope, target_id, reason=TriggerReason.MANUAL_SETTLE
            )
        except TopicBusyError:
            raise
        except Exception as exc:
            raise TopicSettleAdmissionError(
                f"结算材料接纳失败，话题内容已保留，可重试: {target_id}"
            ) from exc
        
        if not completed:
            raise KeyError(f"话题 {target_id} 不存在")
        
        return TopicSettleResult(
            topic_id=target_id,
            generation_task_id=task.task_id if task else None,
        )
    
    async def evict_topic(
        self, 
        scope: IdentityScope, 
        topic_id: str
    ) -> TopicEvictionResult:
        """从活跃话题池中驱逐话题，不触发结算"""
        lease = self._working_set.acquire(scope, topic_id)
        if lease is None:
            return TopicEvictionResult(topic_id=topic_id, removed=False)
        
        try:
            removed = self._store.delete(scope, topic_id)
            if removed:
                self._working_set.remove(scope, topic_id)
            return TopicEvictionResult(topic_id=topic_id, removed=removed)
        finally:
            self._working_set.release(lease)
    
    async def scan_idle_buffers_once(self) -> list[str]:
        """扫描并 settle 空闲超时话题"""
        flushed = []
        candidates = self._working_set.list_idle_candidates(
            self._config.idle_timeout_seconds
        )
        for (scope, topic_id) in candidates:
            try:
                completed, _ = await self._settle_topic(
                    scope, topic_id, reason=TriggerReason.IDLE_TIMEOUT
                )
                if completed:
                    flushed.append(topic_id)
            except TopicBusyError:
                continue  # snapshot 后进入 busy，留给后续维护
        return flushed
    
    async def flush_all_for_shutdown(self) -> TopicShutdownFlushReport:
        """服务关闭前逐 Topic 执行统一 settle 协议"""
        candidates = self._working_set.list_shutdown_candidates()
        settled = []
        skipped = []
        failed = []
        
        for (scope, topic_id) in candidates:
            try:
                lease = self._working_set.acquire(scope, topic_id)
                if lease is None:
                    failed.append(topic_id)
                    continue
                
                try:
                    topic = self._store.get(scope, topic_id)
                    if topic is None:
                        continue
                    
                    task = TopicMaterializeTask.from_topic_data(
                        topic, identity_scope=scope, reason=TriggerReason.SHUTDOWN
                    )
                    if task is None:
                        skipped.append(topic_id)
                        self._store.delete(scope, topic_id)
                        self._working_set.remove(scope, topic_id)
                        continue
                    
                    generation_task = await self._bus.request(
                        PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, task
                    )
                    
                    self._store.delete(scope, topic_id)
                    self._working_set.remove(scope, topic_id)
                    settled.append(topic_id)
                    if generation_task is None:
                        skipped.append(topic_id)
                
                finally:
                    self._working_set.release(lease)
            
            except Exception:
                logger.exception(f"shutdown settle 失败: topic_id={topic_id}")
                failed.append(topic_id)
        
        return TopicShutdownFlushReport(
            settled_topic_ids=tuple(settled),
            generation_skipped_topic_ids=tuple(skipped),
            failed_topic_ids=tuple(failed),
            resident_block_count=...,
        )
```

**关键特性**：

1. **明确的 lease 生命周期**：`acquire → try ... finally release`，不再通过记录状态表达占用；
2. **没有决策矩阵**：`manual_settle_topic` / `evict_topic` / `scan_idle_buffers_once` 是三个具名用例，不再通过 `TriggerReason` 驱动分支；
3. **统一 settle 时序**：`_settle_topic` 是唯一实现，所有来源（manual / idle / LRU / shutdown）共享；
4. **编排而非实现**：Familiar 调用 Engine、Store、WorkingSet，自己不做算法或存储。

---

## 4. 关键设计决策

### 4.1 为什么删除 SemanticBuffer

**当前实现**（[short_term.py:48-69](src/hivememory/patchouli/memory_library/adapters/short_term.py#L48-L69)）：

```python
def put(self, topic: TopicData) -> None:
    self._buffers[key] = self._buffer_from_topic(topic)  # 整条替换

@staticmethod
def _buffer_from_topic(topic: TopicData) -> SemanticBuffer:
    return SemanticBuffer.model_validate(topic.model_dump())  # 深拷贝

@staticmethod
def _topic_from_buffer(buffer: SemanticBuffer) -> TopicData:
    return TopicData(
        blocks=tuple(block.model_copy(deep=True) for block in buffer.blocks),
        bindings=tuple(binding.model_copy(deep=True) for binding in buffer.bindings),
        ...
    )  # 又一次深拷贝
```

**问题**：

1. **SemanticBuffer 从不被原地修改**：每次写入都是 `_buffer_from_topic` → 整条替换，不是 `buffer.blocks.append(...)`；
2. **每次读写都付出两趟深拷贝成本**：`model_dump()` → `model_validate()` → `model_copy(deep=True)`；
3. **TopicData 已经是 `frozen=True`**：调用方拿到引用也无法修改。

**结论**：SemanticBuffer 的"可变性"从未被使用，反而每次读写都在做无用的深拷贝。adapter 内部直接用 `TopicData`：

```python
class InMemoryShortTermStorage:
    def __init__(self):
        self._topics: Dict[WorkspaceTopicKey, TopicData] = {}
        # ...
    
    def get(self, workspace: WorkspaceIdentity, topic_id: str) -> TopicData | None:
        key = self._key(workspace, topic_id)
        with self._lock:
            return self._topics.get(key)  # 直接返回 frozen 对象
    
    def put(self, topic: TopicData) -> None:
        key = self._key(topic.workspace_identity, topic.topic_id)
        with self._lock:
            # 全局 ID 唯一性检查
            previous_scope = self._topic_scopes.get(topic.topic_id)
            if previous_scope is not None and previous_scope != scope:
                raise ValueError(
                    f"topic '{topic.topic_id}' already belongs to another Workspace"
                )
            self._topics[key] = topic  # 直接存储 frozen 对象
            self._workspace_index.setdefault(scope, set()).add(key)
            self._topic_scopes[topic.topic_id] = scope
```

### 4.2 为什么删除 TRIGGER_PLANS 决策矩阵

**当前实现**（[topic_buffer.py:67-75](src/hivememory/patchouli/services/topic_buffer.py#L67-L75)）：

```python
TRIGGER_PLANS: dict[TriggerReason, TriggerPlan] = {
    TriggerReason.TOKEN_OVERFLOW: TriggerPlan(compact=True),
    TriggerReason.IDLE_TIMEOUT: TriggerPlan(settle=True, evict=True),
    TriggerReason.LRU_EVICTION: TriggerPlan(settle=True, evict=True),
    TriggerReason.SHUTDOWN: TriggerPlan(settle=True, evict=True),
    TriggerReason.MANUAL_SETTLE: TriggerPlan(settle=True, evict=True),
    TriggerReason.MANUAL_COMPACT: TriggerPlan(compact=True),
    TriggerReason.MANUAL_DELETE: TriggerPlan(evict=True),
}
```

**问题**：

1. **消费者仍按 reason 分支**：
   ```python
   # perception.py:254
   if trigger.reason is TriggerReason.LRU_EVICTION:
       completed, _ = await self._settle_candidate(..., TriggerReason.LRU_EVICTION)
   ```
   矩阵把 7 个 reason 映射成 3 个 boolean，但调用方仍然检查 `trigger.reason`——矩阵没有消除任何复杂度。

2. **settle / compact / evict 是**概念**，不是**操作组合****：
   - `settle` = 构造 task + admission + 删除 Topic；
   - `compact` = 生成摘要 + 保留 blocks；
   - `evict` = 从 WorkingSet 移除。
   
   它们不是"可以组合的原子操作"（像 Unix 的 `O_CREAT | O_EXCL`），而是**三个独立的用例**，每个都有完整的错误处理和业务语义。

3. **`TriggerReason` 的真正作用是 provenance**：
   - `TopicMaterializeTask` 携带 `reason`，告诉 Generation "这个 task 是怎么来的"（manual？idle？LRU？）；
   - 这是**记录来源**，不是**驱动分支**——就像 HTTP header 里的 `X-Request-ID`，不影响处理逻辑。

**结论**：删除 `TRIGGER_PLANS`，换成 3 个具名方法：

```python
class PerceptionFamiliar:
    async def _settle_topic(self, scope, topic_id, *, reason: TriggerReason):
        """统一 settle 时序（所有 settle 来源共享）"""
        lease = self._working_set.acquire(scope, topic_id)
        try:
            # ... 冻结材料 → admission → 删除
        finally:
            self._working_set.release(lease)
    
    async def _compact_topic(self, scope, topic_id, lease: LeaseToken):
        """执行 compact（调用者已持有 lease）"""
        # ... 生成摘要 → 写回
    
    async def _evict_topic(self, scope, topic_id):
        """从 WorkingSet 移除（不触发结算）"""
        lease = self._working_set.acquire(scope, topic_id)
        try:
            self._store.delete(scope, topic_id)
            self._working_set.remove(scope, topic_id)
        finally:
            self._working_set.release(lease)
```

调用者直接调 `_settle_topic` / `_compact_topic` / `_evict_topic`，不需要查矩阵。`TriggerReason` 只作为参数传给 `_settle_topic`，用于构造 `TopicMaterializeTask`（provenance）。

### 4.3 为什么 TopicWorkingSet 是零依赖的

**对比**：

| 如果 WorkingSet 依赖 Store | 如果 WorkingSet 零依赖 |
|:---|:---|
| `working_set.touch(scope, topic_id)` → 内部调 `store.get(..., touch=True)` | `working_set.touch(scope, topic_id)` → 只更新 `OrderedDict` |
| `working_set.select_lru()` → 遍历 `store.list_all()` + 过滤状态 | `working_set.select_lru()` → `OrderedDict.popitem(last=False)` |
| WorkingSet 必须知道 Store 的接口 | WorkingSet 是纯数据结构 |
| 测试需要 mock Store | 可以纯单元测试 |

**为什么选择零依赖**：

1. **WorkingSet 的职责是"追踪驻留集合"**，不是"读写 Topic 内容"——它只需要知道 `(scope, topic_id)` 在不在集合里、最后访问时间是多少，不需要知道 Topic 的 blocks、summary、bindings；

2. **Store 和 WorkingSet 是平级协作关系**：
   - Store 持有内容（`TopicData`）；
   - WorkingSet 持有索引（`OrderedDict` + lease 表）；
   - Familiar 编排它们。
   
   如果 WorkingSet 依赖 Store，就变成了"WorkingSet 是 Store 的上层"——这和"WorkingSet 是工作集策略、Store 是持久化"的分层矛盾。

3. **未来扩展性**：
   - 如果 Store 换成 Redis，WorkingSet 一行不改（因为它不知道 Store 是什么）；
   - 如果要支持"多个 Workspace 共享一个工作集"（未来可能），只需要调整 `TopicWorkingSet._max_resident` 的语义，不需要改 Store。

### 4.4 为什么 TopicData 不再有 last_accessed_at

**当前实现**：`TopicData.last_accessed_at` 作为记录字段，每次 `store.get(..., touch=True)` 时更新。

**问题**：

1. **访问追踪不是记录属性**：
   - 如果 Store 是 Redis，每次 `touch` 都要写 Redis（性能损耗）；
   - 如果有两个进程，它们的 `last_accessed_at` 会互相覆盖（并发问题）；
   - 类比：文件的 `atime`（access time）是文件系统元数据，不是文件内容——你不会在文件开头写一行"last accessed at 2026-09-04 12:34:56"。

2. **WorkingSet 已经持有访问时间**：
   ```python
   self._resident: OrderedDict[tuple[IdentityScope, str], float] = OrderedDict()
   #                                                        ^^^^^ 访问时间
   ```
   在 `TopicData` 里再存一遍是冗余的。

**结论**：`TopicData` 删除 `last_accessed_at` 字段，访问追踪由 `WorkingSet.touch()` 管理。

### 4.5 为什么 BufferState 枚举要删除

**当前实现**：`BufferState.IDLE / PROCESSING / SETTLING` 作为 `TopicData.state` 字段。

**问题**（已在 §2.1 详细分析）：

1. **`PROCESSING` / `SETTLING` 是执行锁，不是业务状态**：读取者看到 `state=PROCESSING` 没有业务意义——它只表示"有人正在操作这个 Topic"；
2. **跨 await 占用权通过记录状态实现**：锁外操作完成后必须校验"记录还是我当初锁的那个吗"；
3. **状态检查散布在多处**：没有单一的"占用表"所有者。

**结论**：删除 `BufferState` 枚举，用 `TopicWorkingSet._leases: Dict[..., LeaseToken]` 表达占用权。调用者在 `finally` 里 `release(lease)`，不再修改记录字段。

---

## 5. 实施阶段

### 阶段 1：创建 TopicWorkingSet（独立分支）

**目标**：实现独立的、零依赖的工作集管理器，通过测试验证其正确性。

**任务**：

1. 创建 `src/hivememory/patchouli/services/topic_working_set.py`；
2. 实现 `TopicWorkingSet` 类（`OrderedDict` + lease 表）；
3. 实现 `LeaseToken` dataclass；
4. 编写单元测试（`tests/unit/patchouli/services/test_topic_working_set.py`）：
   - LRU 排序（`touch` → `move_to_end`）；
   - 候选查询（idle / LRU / shutdown）；
   - Lease 获取/释放（acquire / release）；
   - 容量判断（`needs_eviction`）；
   - 边界条件（空集合、全部 busy、跨 Workspace）。

**验收**：
- TopicWorkingSet 测试全部通过；
- 测试不依赖 Store、Relay、Queue（纯单元测试）；
- 代码行数 ≤ 150 行（含注释）。

### 阶段 2：简化 ShortTermMemoryStore（同一分支）

**目标**：删除 Store 的状态机方法、组合操作、WorkingSet 逻辑，回归纯 CRUD。

**任务**：

1. 删除 `SemanticBuffer` 类（`buffer.py`）；
2. Adapter 内部直接用 `TopicData`（删除 `_buffer_from_topic` / `_topic_from_buffer`）；
3. 删除 `TopicData.last_accessed_at` 字段；
4. 删除 `TopicData.state` 字段（`BufferState` 枚举暂时保留，作为兼容字段）；
5. 删除 Store 的所有状态机方法：
   - `reserve_processing` / `release_processing`；
   - `freeze_for_manual_settle` / `commit_flushing` / `abort_flushing`；
   - `freeze_and_evict`；
   - `apply_compaction` / `clear_blocks`。
6. 简化 Store 接口：
   - `get(scope, topic_id) -> TopicData | None`（删除 `touch` 参数）；
   - `put(topic: TopicData) -> None`；
   - `create(scope, *, topic_title, topic_summary) -> TopicData`；
   - `delete(scope, topic_id) -> bool`；
   - `list_by_workspace(scope, *, include_empty) -> list[TopicData]`；
   - `count(scope) -> int`。
7. 更新 adapter 测试（`test_in_memory_short_term_storage.py`）；
8. 更新 Store 测试（`test_short_term_store_crud_boundary.py`）。

**验收**：
- `ShortTermMemoryStore` ≤ 100 行（含注释）；
- 所有 Store 测试通过（CRUD + 全局 ID 唯一性）；
- Adapter 测试通过（不再有 `SemanticBuffer` 相关测试）。

### 阶段 3：重构 PerceptionFamiliar（同一分支）

**目标**：让 Familiar 持有 WorkingSet、Store、Engine，实现 lease 机制，删除对 TopicBufferService 的依赖。

**任务**：

1. Familiar `__init__` 增加 `working_set: TopicWorkingSet` 参数；
2. 重写 `apply_interaction`：
   ```python
   lease = self._working_set.acquire(scope, topic_id)
   if lease is None:
       raise TopicBusyError(...)
   try:
       block = self._engine.build_block(payload, scope)
       topic = self._store.get(scope, topic_id)
       updated = topic.model_copy(update={"blocks": (...)})
       self._store.put(updated)
       if self._engine.should_compact(updated.total_tokens):
           await self._compact_topic(scope, topic_id, lease)
   finally:
       self._working_set.release(lease)
   ```
3. 重写 `_maybe_evict_lru`：
   ```python
   if not self._working_set.needs_eviction(scope):
       return
   lru_id = self._working_set.select_lru_candidate(scope, exclude=attempted)
   await self._settle_topic(scope, lru_id, reason=TriggerReason.LRU_EVICTION)
   ```
4. 实现 `_settle_topic` / `_compact_topic` / `_evict_topic`（删除 `handle_trigger` 和矩阵）；
5. 更新 `scan_idle_buffers_once` / `flush_all_for_shutdown`；
6. 删除对 `TopicBufferService` 的依赖（改为直接持有 Store + Relay）。

**验收**：
- `PerceptionFamiliar` ≤ 400 行；
- `test_perception.py` 测试通过（LRU / idle / shutdown）；
- `test_interaction_submission.py` 测试通过（retry / compact）。

### 阶段 4：拆分 MemoryPerceptionEngine（新分支）

**目标**：从 `semantic_flow_perception_layer.py` 中提取纯算法，删除对 Patchouli 的依赖。

**任务**：

1. 创建 `src/hivememory/engines/perception/memory_perception_engine.py`；
2. 实现 `MemoryPerceptionEngine` 类（只包含纯函数）：
   - `build_block(payload, scope) -> LogicalBlock`；
   - `should_compact(total_tokens) -> bool`；
   - `select_blocks_to_fold(blocks, retain_recent) -> list[LogicalBlock]`。
3. 删除 `semantic_flow_perception_layer.py`（所有用例已迁移到 Familiar）；
4. 删除 `BasePerceptionLayer` / `NullPerceptionLayer`（`interfaces.py`）；
5. 更新 `engines/perception/__init__.py`（只导出 `MemoryPerceptionEngine` + `BaseRelayController`）；
6. 删除 Engine 对 Patchouli 的所有导入：
   - `from hivememory.patchouli.services.topic_buffer import ...` → 删除；
   - `from hivememory.patchouli.errors import ...` → 删除；
   - `from hivememory.patchouli.control.interaction_apply_journal import ...` → 删除。
7. 更新 Engine 测试（`test_layers.py` / `test_page_folding.py`）：
   - 只测试纯函数（`build_block` / `should_compact`）；
   - 删除对 Store / Journal 的 mock。

**验收**：
- `MemoryPerceptionEngine` ≤ 200 行；
- 静态检查确认 `engines/perception/` 不导入 `hivememory.patchouli.*`；
- Engine 测试通过（纯算法测试，不依赖外部状态）。

### 阶段 5：清理与文档（同一分支）

**目标**：删除旧组件、更新导入、收口文档。

**任务**：

1. 删除 `TopicBufferService`（`src/hivememory/patchouli/services/topic_buffer.py`）；
2. 删除 `TRIGGER_PLANS` / `TriggerPlan`（已被 3 个具名用例替代）；
3. 删除 `BufferState` 枚举（如果没有其他引用）；
4. 更新 Runtime 组装（`runtime/core.py`）：
   ```python
   working_set = TopicWorkingSet(max_resident=config.max_resident_topics)
   store = ShortTermMemoryStore(port=...)
   engine = MemoryPerceptionEngine(config=...)
   familiar = PerceptionFamiliar(
       engine=engine,
       store=store,
       working_set=working_set,
       relay_controller=relay,
       ...
   )
   ```
5. 更新顶层导入（`src/hivememory/__init__.py`）；
6. 更新文档：
   - `docs/patchouli/perception.md`（新架构图）；
   - `docs/patchouli/memory-library.md`（Store 回归 CRUD）；
   - `docs/patchouli/README.md`（更新组件列表）。
7. 归档或修订已被替代的 todo：
   - `docs/todo/short-term-memory-store-boundary-cleanup.md`（修订结论：Store 不是纯 CRUD，而是聚合存储 → 现已是纯 CRUD）。

**验收**：
- 所有生产代码不再导入 `TopicBufferService` / `SemanticBuffer` / `BasePerceptionLayer`；
- 所有测试通过（`pytest tests/unit/patchouli/ tests/unit/engines/perception/ -n auto`）；
- 文档更新完成。

---

## 6. 测试与验收

### 6.1 TopicWorkingSet 单元测试

```python
def test_touch_updates_lru_order():
    ws = TopicWorkingSet(max_resident=3)
    ws.touch(scope_a, "topic-1")
    ws.touch(scope_a, "topic-2")
    ws.touch(scope_a, "topic-1")  # 移到末尾
    
    lru = ws.select_lru_candidate(scope_a)
    assert lru == "topic-2"  # topic-1 刚被 touch，topic-2 最旧

def test_lease_prevents_lru_selection():
    ws = TopicWorkingSet(max_resident=3)
    ws.touch(scope_a, "topic-1")
    ws.touch(scope_a, "topic-2")
    
    lease = ws.acquire(scope_a, "topic-1")
    lru = ws.select_lru_candidate(scope_a)
    assert lru == "topic-2"  # topic-1 被占用，跳过
    
    ws.release(lease)
    lru = ws.select_lru_candidate(scope_a)
    assert lru == "topic-1"  # 释放后可以被选中

def test_idle_candidates_filters_leased():
    ws = TopicWorkingSet()
    ws.touch(scope_a, "topic-1")
    time.sleep(0.1)
    ws.touch(scope_a, "topic-2")
    
    lease = ws.acquire(scope_a, "topic-1")
    candidates = ws.list_idle_candidates(timeout_seconds=0.05)
    
    assert len(candidates) == 0  # topic-1 超时但被占用，topic-2 未超时
```

### 6.2 ShortTermMemoryStore 测试

```python
def test_put_rejects_duplicate_topic_id_across_workspaces():
    store = ShortTermMemoryStore()
    topic_a = store.create(scope_a, topic_title="Topic A")
    
    topic_b = TopicData(
        topic_id=topic_a.topic_id,  # 同一 ID
        workspace_identity=scope_b.workspace_identity,  # 不同 Workspace
        topic_title="Topic B",
        last_update=time.time(),
    )
    
    with pytest.raises(ValueError, match="already belongs to another Workspace"):
        store.put(topic_b)

def test_get_returns_frozen_snapshot():
    store = ShortTermMemoryStore()
    topic = store.create(scope_a, topic_title="Test")
    
    snapshot = store.get(scope_a, topic.topic_id)
    with pytest.raises(FrozenInstanceError):
        snapshot.topic_title = "Modified"  # frozen=True
```

### 6.3 PerceptionFamiliar 集成测试

```python
async def test_apply_interaction_with_lease():
    working_set = TopicWorkingSet()
    store = ShortTermMemoryStore()
    engine = MemoryPerceptionEngine(config=...)
    familiar = PerceptionFamiliar(
        engine=engine, store=store, working_set=working_set, ...
    )
    
    topic = store.create(scope_a, topic_title="Test")
    
    # 第一次摄入：正常获取 lease
    topic_id = await familiar.apply_interaction(
        payload=InteractionPayload(...),
        identity_scope=scope_a,
        target_topic_id=topic.topic_id,
    )
    
    # 验证 WorkingSet 更新
    assert working_set._resident.get((scope_a, topic_id)) is not None

async def test_lru_eviction_when_pool_full():
    working_set = TopicWorkingSet(max_resident=2)
    store = ShortTermMemoryStore()
    familiar = PerceptionFamiliar(working_set=working_set, store=store, ...)
    
    # 创建 2 个 Topic（满容量）
    topic_1 = await familiar.prepare_topic("NEW_TOPIC", ...)
    topic_2 = await familiar.prepare_topic("NEW_TOPIC", ...)
    
    # 创建第 3 个 Topic → 触发 LRU 驱逐
    with patch.object(familiar._bus, 'request') as mock_admission:
        mock_admission.return_value = MemoryGenerationTask(...)
        topic_3 = await familiar.prepare_topic("NEW_TOPIC", ...)
    
    # 验证 topic_1 被驱逐
    assert store.get(scope_a, topic_1) is None
    assert store.get(scope_a, topic_2) is not None
    assert store.get(scope_a, topic_3) is not None
```

### 6.4 结构契约测试

```python
def test_engines_does_not_import_patchouli():
    """静态检查：engines/perception/ 不导入 patchouli"""
    import ast
    perception_dir = Path("src/hivememory/engines/perception")
    for py_file in perception_dir.glob("*.py"):
        if py_file.name.startswith("test_"):
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not node.module.startswith("hivememory.patchouli"), \
                    f"{py_file.name} imports {node.module}"

def test_no_semantic_buffer_imports():
    """确认没有 SemanticBuffer 的生产导入"""
    result = subprocess.run(
        ["grep", "-r", "from.*buffer import SemanticBuffer", "src/"],
        capture_output=True, text=True
    )
    assert result.returncode != 0, "Found SemanticBuffer imports in production code"

def test_no_topic_buffer_service_imports():
    """确认没有 TopicBufferService 的生产导入"""
    result = subprocess.run(
        ["grep", "-r", "from.*topic_buffer import TopicBufferService", "src/"],
        capture_output=True, text=True
    )
    assert result.returncode != 0, "Found TopicBufferService imports"
```

---

## 7. 风险与缓解

### 7.1 风险

1. **WorkingSet 的 lease 机制可能遗漏 release**：
   - 如果调用者在 `acquire` 后忘记 `finally: release(lease)`，Topic 会永久 busy；
   - 缓解：每个 lease 持有 `acquired_at` 时间戳，WorkingSet 可以提供"清理超时 lease"的维护接口（未来扩展）。

2. **删除 `last_accessed_at` 后，Store 和 WorkingSet 的同步**：
   - Store 不再知道"这个 Topic 最后被谁访问"；
   - 缓解：WorkingSet 是唯一的访问追踪所有者，调用者统一通过 `working_set.touch()` 更新；Familiar 在每次 Store 读写后调用 `touch`。

3. **Engine 重命名影响大量旧测试**：
   - `semantic_flow_perception_layer` → `memory_perception_engine` 影响所有导入；
   - 缓解：集中迁移，阶段 4 完成后运行全量测试。

4. **Relay 在锁外执行，但不再有 `expected_state` 校验**：
   - 旧实现通过记录状态 + 校验保护；新实现通过 lease 保护；
   - 缓解：lease 在整个 Relay 执行期间持有，其他线程无法获取同一 Topic 的 lease，因此不需要事后校验。

### 7.2 回滚计划

如果阶段 3 或 4 出现不可预见的问题，可以回滚到阶段 2：

- WorkingSet 和简化 Store 已经完成且测试通过；
- 保留 TopicBufferService 作为兼容层，暂时不删除；
- 让 TopicBufferService 内部使用 WorkingSet + 简化 Store，但保持原有接口。

这样可以先收获"WorkingSet 独立测试"和"Store 简化"的好处，推迟"删除 TopicBufferService"的风险。

---

## 8. 完成条件

- [ ] `TopicWorkingSet` 实现完成，单元测试通过（≤ 150 行）；
- [ ] `ShortTermMemoryStore` 简化为纯 CRUD（≤ 100 行）；
- [ ] `SemanticBuffer` / `BufferState` 删除；
- [ ] `PerceptionFamiliar` 重写，持有 WorkingSet + Store + Engine（≤ 400 行）；
- [ ] `MemoryPerceptionEngine` 提取为纯算法（≤ 200 行）；
- [ ] `TopicBufferService` / `BasePerceptionLayer` / `NullPerceptionLayer` 删除；
- [ ] `TRIGGER_PLANS` 决策矩阵删除，换成 3 个具名用例；
- [ ] 静态检查确认 `engines/perception/` 不导入 `hivememory.patchouli.*`；
- [ ] 所有单元测试 + 集成测试通过；
- [ ] 文档更新（`perception.md` / `memory-library.md` / `README.md`）；
- [ ] 归档或修订已被替代的 todo。

---

## 9. 附录：历史教训

### 9.1 为什么 TriggerManager 失败

TriggerManager 试图同时做两件事：
1. **管理工作集状态**（LRU、占用）；
2. **解释触发原因**（决策矩阵、组合操作）。

但它们的依赖方向相反：
- 管理工作集 → 需要访问 Store 和状态机（向下依赖）；
- 解释触发原因 → 是跨系统编排（向上依赖）。

把它们绑在一起后，无论放哪都会制造循环依赖或反向依赖：
- 放 `engines/` → 不能依赖 patchouli 错误和 Port；
- 放 `services/` → 但它又不该暴露给其他服务。

**正确的拆分**：
- 工作集状态 → `TopicWorkingSet`（零依赖的数据结构）；
- 触发原因 → 删除矩阵，换成 3 个具名用例；
- 跨系统编排 → `PerceptionFamiliar`（持有 WorkingSet + Store + Engine）。

### 9.2 为什么 TopicBufferService 也失败

TopicBufferService 把所有逻辑都塞进一个类，试图"成为短期存储的唯一入口"。但这制造了新的混乱：

1. **既是 Store 代理，又是 Store 调用者**：Familiar 不知道该调 Store 还是调 Service；
2. **状态机仍在记录里**：没有解决 BufferState 的根本问题；
3. **WorkingSet 仍通过遍历实现**：没有独立的索引。

**教训**：**大一统 Service 不是解决职责混乱的办法**——正确的办法是**纠正范畴错误，让每个概念用正确的形式表达**。

### 9.3 为什么新方案能成功

新方案不是"重新分配职责"，而是**纠正三个范畴错误**：

1. **锁 → lease 表**（不再是记录字段）；
2. **工作集 → TopicWorkingSet**（独立对象，不是 Store 的一部分）；
3. **Perception → 拆成算法和 runtime**（Engine 和 Familiar 各司其职）。

纠正后，**职责分配变得自然**——不再需要"把 X 的能力塞给 Y，因为 Z 不能依赖 X"这种妥协。每个组件做它该做的事，依赖关系单向且清晰。

---

## 10. 修订记录

### 10.1 2026-09-04：第三版计划（推翻前两轮）

本次修订废止此前以下目标：

- `TopicBufferService` 作为短期存储的唯一入口（阶段 2）；
- `ShortTermMemoryStore` 收缩为纯 CRUD，禁止承接状态机制（阶段 1）；
- 保留 `BufferState` 作为记录字段；
- 保留 `TRIGGER_PLANS` 决策矩阵。

新的冻结判断为：

1. **三个范畴错误**（锁、工作集、Perception）是职责混乱的根本原因，必须纠正；
2. **TopicWorkingSet** 是独立的、零依赖的驻留管理器（OrderedDict + lease 表）；
3. **ShortTermMemoryStore** 回归纯 CRUD（≤ 100 行）；
4. **MemoryPerceptionEngine** 是纯算法（不依赖 Patchouli）；
5. **PerceptionFamiliar** 持有 WorkingSet + Store + Engine，负责编排；
6. **删除** TopicBufferService、SemanticBuffer、BufferState、TRIGGER_PLANS、TriggerManager、BasePerceptionLayer、NullPerceptionLayer。

此前阶段 1-2 的调用点盘点和失败测试清单仍可作为迁移证据，但其中关于"TopicBufferService 作为唯一入口"和"Store 纯 CRUD"的结论均由本修订替代。后续实施与验收以本版本的目标架构、四组件分层和完成条件为准。