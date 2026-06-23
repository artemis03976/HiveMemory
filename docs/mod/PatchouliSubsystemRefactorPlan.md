# Patchouli 子系统重构规划

**文档状态**: 规划中  
**关联版本**: v0.5.3（预计）  
**前置条件**: v0.5.2 全部 Phase 完成（AsyncQdrantClient + generation/retrieval async-native）

---

## 1. 背景：LibrarianCore 是 v4 架构演进的残留

### 1.1 历史演变

旧系统（v3）的结构非常直接：`PatchouliSystem` 对外承接 API，`PatchouliKernel` 承载运行时，`Koakuma` 负责 Agent 工具调用，`RetrievalFamiliar` 负责检索，**剩余所有记忆编排工作被封装进 LibrarianCore**。

v4 架构演进时，Agent 部分（Alice）被完全分离出去，但 Patchouli 的记忆管理部分基本未动——LibrarianCore 成为了"Agent 分离后剩余的一切"的容器。

`MemoryGenerationTaskController` 的出现（v0.5.0）是第一次有意识地从 LibrarianCore 中解构职责，但当时没有意识到这是解构 LibrarianCore 本身的起点。**LibrarianCore 才是需要被解构的那一个。**

### 1.2 现有结构的问题

LibrarianCore 同时做了两类性质不同的事：

| 职责 | 性质 | 问题 |
|---|---|---|
| 感知回调路由 | 编排逻辑 | 应属于流水线，不应是类方法 |
| Topic context 拼装 | 领域逻辑碎片 | 从引擎泄漏进来的逻辑 |
| 记忆生成任务管理 | 运行时编排 | 已拆出为 TaskController，但仍作为内部组件 |
| 对外暴露任务 API | 代理层 | 只是透传 TaskController 接口 |

`LibrarianCore` 这个名字隐含了"图书管理员管理与图书库的一切交互"的隐喻，理论上三级记忆系统的管理都应由它完成。而如今，**整个 Patchouli 子系统已经是这个隐喻的具象化**——它不应该是一个类，而是整个子系统的设计思路。

### 1.3 三级记忆体系的正确定位

三级记忆体系（短期 buffer → 中期向量库 → 长期冷存储）是**面向存储的概念框架**，而不是服务边界的划分依据。

记忆操作本身具有天然的跨层性：一次完整的用户交互会同时经过感知（短期）→ 查重（中期）→ 生成（写中期）→ 生命力更新（中期 lifecycle）→ 归档（长期）。强行按存储层划分三个平行服务，会倒逼重建一个等价于 LibrarianCore 的协调者。

正确用法：三级模型指导"数据该存哪里"与"状态如何转移"，而不是"代码怎么拆"。

---

## 2. 目标架构

### 2.1 总体结构

```
PatchouliRuntime（装配根）
├── MemoryLibrary            ← 三级存储 + 状态转移
├── PerceptionFamiliar       ← 感知业务门面
├── RetrievalFamiliar        ← 全层检索服务（扩展）
├── MemoryGenerationFamiliar ← 生成写入业务门面
├── LifecycleFamiliar        ← 生命周期维护业务门面
├── MemoryGenerationCoordinator ← 生成入口协调层（高于业务门面）
└── MemoryGenerationTaskController ← 生成任务控制面（高于业务门面）
```

**图书馆隐喻的对应**：
- `MemoryLibrary`：书库，管书的物理存放和流转
- `PerceptionFamiliar`：使魔，代理人接收新记录并整理短期话题
- `RetrievalFamiliar`：使魔，代理人去库里取书
- `MemoryGenerationFamiliar`：使魔，代理人把材料写成书并入库
- `LifecycleFamiliar`：使魔，代理人维护书的活性、归档与复活
- `MemoryGenerationCoordinator`：入口协调者，把原始请求整理为任务需求
- `MemoryGenerationTaskController`：任务控制面，管理任务创建、运行状态与事件

### 2.2 MemoryLibrary

统一的三级存储层，负责记忆的物理存放和各层之间的状态转移。

```
MemoryLibrary
├── ShortTermMemoryStore    ← SemanticBufferManager（当前在 PerceptionLayer 内）
│     - 内存态 buffer
│     - topic 创建/查询/驱逐
│     - 为 MemoryGenerationFamiliar / RetrievalFamiliar 提供 topic 读取能力（不编排生成流程）
│
├── MidTermMemoryStore      ← QdrantMemoryStore（当前直接暴露）
│     - 向量数据库 CRUD
│     - 精确/模糊检索（由 RetrievalFamiliar 调用）
│
└── LongTermMemoryStore     ← ArchiveStore（待实现）
      - 冷存储（文件系统 / SQL）
      - 归档记录管理
      - 复活键（revival_keys）管理
```

**状态转移协议**：

`SemanticBuffer → MemoryAtom` 涉及 LLM 提取、查重、向量编码等生成过程，**不是存储层的状态迁移**。原始触发先由 `PerceptionFamiliar` 从感知层 callback 接住，再通过 local bus 送入 `MemoryGenerationCoordinator` 整理为统一任务需求，随后由 `MemoryGenerationTaskController` 创建任务；Controller 作为控制面通过 local bus 请求 `MemoryGenerationFamiliar` 执行生成写入。MemoryLibrary 只提供两端的读写能力。

```
# 短期 → 中期：Generation 驱动，不是 Library 方法
TriggerManager 触发 → PerceptionFamiliar callback
    → local bus: generation.submit_archive
    → MemoryGenerationCoordinator.submit_archive(ArchivePayload)
    → local bus: memory_task.submit_generation
    → MemoryGenerationTaskController.submit_generation(spec)
    → local bus: generation.execute_spec
    → MemoryGenerationFamiliar.execute(spec)
    → LLM 提取 / 查重 / 向量编码
    → MidTermMemoryStore.upsert()

# 中期 → 长期：MemoryLibrary 直接管理（纯数据搬运）
MemoryLibrary.archive(memory_id)
    → MidTermMemoryStore.get() → LongTermMemoryStore.persist() → MidTermMemoryStore.delete()

# 长期 → 中期：MemoryLibrary 直接管理（纯数据搬运）
MemoryLibrary.revive(memory_id)
    → LongTermMemoryStore.load() → MidTermMemoryStore.upsert() → LongTermMemoryStore.remove()
```

### 2.3 StoragePort 抽象层

三层存储的结构形态一致：**XxxMemoryStore 持有 XxxStoragePort，Port 多态实现**。各层 Port 契约根据数据语义分别定义。

#### ShortTermStoragePort

Buffer 语义是 `topic_id → SemanticBuffer` 的键值映射：

```python
class ShortTermStoragePort(ABC):
    async def get(self, topic_id: str) -> Optional[SemanticBuffer]: ...
    async def put(self, topic_id: str, buffer: SemanticBuffer) -> None: ...
    async def pop(self, topic_id: str) -> Optional[SemanticBuffer]: ...
    async def list_by_user(self, user_id: str) -> List[SemanticBuffer]: ...
    async def list_all(self) -> List[SemanticBuffer]: ...
```

实现：`InMemoryShortTermStorage`（当前 `SemanticBufferManager._buffers` 迁入）、`RedisShortTermStorage`（future）。

`ShortTermMemoryStore` 持有 Port，LRU 驱逐、`fold_blocks`、`last_active_topic` 等上层调度逻辑**不下沉到 Port**，保留在 Store 层。

#### MidTermStoragePort

Port 以 `MemoryAtom` 为边界，向量编码等实现细节封装在适配器内：

```python
class MidTermStoragePort(ABC):
    async def upsert(self, memory: MemoryAtom) -> None: ...
    async def get(self, memory_id: UUID) -> Optional[MemoryAtom]: ...
    async def get_by_alias(self, alias: str, user_id: Optional[str] = None) -> Optional[MemoryAtom]: ...
    async def delete(self, memory_id: UUID) -> bool: ...
    async def batch_delete(self, ids: List[UUID]) -> int: ...
    async def search(self, query: str, top_k: int, filters: ..., mode: str) -> List[SearchResult]: ...
    async def scroll(self, filters: ..., limit: int) -> List[MemoryAtom]: ...
    async def count(self, filters: ...) -> int: ...
```

实现：`QdrantStorageAdapter`（包装现有 `QdrantMemoryStore`，含 embedding 逻辑）、`GraphStorageAdapter`（future）。

`MidTermMemoryStore` 可持有多个 Port 实例，写入时同步到所有后端，查询按类型分发：

```python
class MidTermMemoryStore:
    _primary: MidTermStoragePort          # 主存储（向量库），承接大多数操作
    _secondary: list[MidTermStoragePort]  # 扩展存储（图库等），写入时同步
```

#### LongTermStoragePort

现有 `BaseMemoryArchiver` 仅有 `archive`/`resurrect`，缺少查询能力，需补全：

```python
class LongTermStoragePort(ABC):
    async def persist(self, memory: MemoryAtom) -> None: ...
    async def load(self, memory_id: UUID) -> MemoryAtom: ...
    async def remove(self, memory_id: UUID) -> None: ...
    async def is_archived(self, memory_id: UUID) -> bool: ...
    async def query(self, filters: ArchiveQuery) -> List[ArchiveRecord]: ...
```

实现：`FileBasedStorageAdapter`（`FileBasedArchiver` 纯文件读写部分）、`DBBasedStorageAdapter`（future）。

#### 关键耦合修正

`FileBasedArchiver` 目前直接持有 `QdrantMemoryStore`，在归档/复活时调用 `storage.get_memory()` / `storage.upsert_memory()`——**长期存储耦合了中期存储的具体实现**。

修正：`archive`/`resurrect` 的跨层操作上移至 `MemoryLibrary` 的状态转移方法。`LongTermStoragePort` 的实现只负责冷存储的读写，不感知中期存储。

```
# 状态转移由 MemoryLibrary 编排，不再由 archiver 内部跨层调用：
ShortTermMemoryStore.flush(topic_id) → MidTermMemoryStore.upsert()
MidTermMemoryStore.archive(id)       → LongTermMemoryStore.persist() + MidTermMemoryStore.delete()
LongTermMemoryStore.revive(id)       → MidTermMemoryStore.upsert()   + LongTermMemoryStore.remove()
```

### 2.4 感知层解耦

感知层（PerceptionLayer + TriggerManager）目前与存储实现深度纠缠，是短期记忆解耦的主要工作面。

#### TriggerManager 的存储依赖

`TriggerManager` 目前直接持有 `SemanticBufferManager`，对 `SemanticBuffer` 字段做直接写入（`buffer.blocks.clear()`、`buffer.state_summary = ...` 等）。

重构后持有 **`ShortTermMemoryStore`**，不走总线，不持有整个 `MemoryLibrary`。

理由：TriggerManager 的 compact/evict 操作是同步的内存操作，总线徒增异步边界，也会让 shutdown flush 的等待语义变复杂；它只需要短期存储能力，无需感知中期/长期。

#### buffer 状态修改的收束

散落的直接字段修改全部收束为 `ShortTermMemoryStore` 的命名方法：

| 现有的直接修改 | 收束为 Store 方法 |
|---|---|
| `buffer.blocks.clear()` + `total_tokens = 0` | `store.clear_blocks(topic_id)` |
| `buffer.state_summary = s` + `last_update` | `store.update_summary(topic_id, s)` |
| `buffer.topic_title = title` | `store.update_title(topic_id, title)` |

`fold_blocks` / `update_metadata` 已有，保持不变。

#### Compact 的计算与写入分离

`_compact_topic` 目前混合了两件性质不同的事：

```python
# 重构前：计算 + 写入耦合
new_summary = self._relay_controller.generate_summary(...)  # 纯计算
buffer.state_summary = new_summary                          # 直接字段写入（越权）

# 重构后：各司其职
new_summary = self._relay_controller.generate_summary(blocks_to_fold, previous_summary)
self._short_term_store.update_summary(topic_id, new_summary)
```

`RelayController.generate_summary` 保持纯计算，不变。

#### PerceptionLayer 的依赖注入

`SemanticFlowPerceptionLayer` 目前自己创建 `SemanticBufferManager`，TriggerManager 持有同一实例。重构后改为注入：

```
MemoryLibrary 初始化时创建 ShortTermMemoryStore
    ↓ 注入
SemanticFlowPerceptionLayer(short_term_store=...)
    ↓ 注入（同一实例）
TriggerManager(short_term_store=...)
```

`SemanticBuffer` 继续作为只读 DTO 传出（`get_buffer()` / `get_active_topics_snapshots()` 的返回值供调用方读取字段），但所有写操作必须通过 Store 接口。

#### `_archive_topic` 不受影响

`_archive_topic` 操作的是调用方在 `resolve_topic` 中提前提取的 `blocks_snapshot`，自身不访问任何 buffer 字段。感知层解耦对该方法无影响。它通过 `_on_generate_memory` 回调触发生成链路；Phase 4 中该回调由 `PerceptionFamiliar` 接住，再通过 local bus 进入 `MemoryGenerationCoordinator`，与感知层存储解耦是独立的重构步骤，互不阻塞。

### 2.5 中期记忆系统重构要点

#### Deduplicator 搜索优化

`MemoryDeduplicator` 目前自持 `QdrantMemoryStore` 仅为执行一次 `search_memories`，导致对存储层的不必要直接依赖。

**方案：将 dedup 改为纯决策引擎，搜索职责交还给 MemoryGenerationFamiliar。**

```python
# 重构前：dedup 内部发起搜索
async def check_duplicate(self, draft) -> Tuple[Decision, Optional[MemoryAtom]]:
    results = await self.storage.search_memories(...)  # 自持存储，自己搜

# 重构后：dedup 只做决策，候选由 MemoryGenerationFamiliar 传入
def check_duplicate(
    self,
    draft: ExtractedMemoryDraft,
    candidates: list[SearchResult],  # MemoryGenerationFamiliar 检索后传入
) -> Tuple[DuplicateDecision, Optional[MemoryAtom]]:
    ...  # 纯逻辑，无 I/O
```

`MemoryDeduplicator` 不再持有任何存储引用，`merge_memory` / `_make_decision` 等逻辑不变。`MemoryGenerationFamiliar` 在生成写入步骤中执行搜索（`MidTermMemoryStore.search()`），将结果传入 dedup——搜索本就在这一步发生，不增加额外网络往返。

#### 两条主要处理链路

**记忆摄入链路**（dedup → create/update → upsert）已在 TaskController 中基本成型。核心工作：dedup 改为纯决策后，链路内所有存储操作切换到 `MidTermStoragePort`，TaskController 不再直接持有 `QdrantMemoryStore`。

**生命力维护链路**（vitality 衰减 / 强化 / 垃圾回收）：LifecycleEngine 目前直接持有 `QdrantMemoryStore` 做 vitality 更新和批量删除，同样需要切换到 `MidTermStoragePort`。两条链路可独立推进。

### 2.6 长期记忆系统重构要点

`FileBasedArchiver` 承担的职责（归档、索引维护、resurrect）就是 `LongTermStoragePort` 的 `FileBasedStorageAdapter` 实现的完整映射。

迁移路径：
1. `LongTermMemoryStore` 建立，`FileBasedStorageAdapter` 包装现有 `FileBasedArchiver` 逻辑
2. 状态转移（archive / revive）上移至 `MemoryLibrary`（参见 §2.5 关键耦合修正）
3. `archiver.py` 作为独立文件退场，其逻辑以适配器形式存活于 `LongTermStoragePort` 实现中

长期存储是三层中改动最干净的部分，无深度耦合，可最后推进。

### 2.7 ArtifactStore（附属资产仓库）

`ArtifactStore` 与 `LongTermMemoryStore` 一样依赖外部持久化介质（文件系统 / SQL / 对象存储等），但它不属于三级记忆体系本身。Artifact 不是第四级记忆，而是记忆原子的附属资产与溯源产物，通常与 `MemoryAtom` 强关联，并以 append-only 形式保存 interaction snapshot、memory creation、memory version 等不可变记录。

当前 `ArtifactStore` 由 `ArtifactEngine` 私有创建并独占管理，这会导致其他链路无法将 artifact 仓库作为稳定的系统级存储能力使用。重构后，`ArtifactStore` 应上移至 `MemoryLibrary` 的装配范围，作为三级记忆之外的附属存储仓库：

```
MemoryLibrary
├── ShortTermStore
├── MidTermStore
├── LongTermStore
└── ArtifactRepository / ArtifactStore
```

但需要注意：`MemoryLibrary` 只负责 artifact 仓库的持有、注入和基础读写能力暴露，不负责决定何时构建 creation/version/interaction artifact。Artifact 的构建逻辑仍然属于 `ArtifactEngine` / builder，生成写入链路由 `MemoryGenerationFamiliar` 编排。

目标结构保持与其他存储系统一致：

```python
class ArtifactStoragePort(ABC):
    async def put(self, artifact: BaseArtifact) -> ArtifactRef: ...
    async def get(self, ref_or_id: ArtifactRef | str) -> dict: ...
    async def exists(self, artifact_id: str) -> bool: ...
    async def list_by_memory(self, memory_id: str, artifact_type: ArtifactType | None = None) -> list[ArtifactRef]: ...
    async def verify(self, ref: ArtifactRef) -> ArtifactIntegrityResult: ...
```

实现：`FilesystemArtifactStorageAdapter`（当前 `FilesystemArtifactStore` 迁移而来）、`SQLArtifactStorageAdapter`（future）、`ObjectStorageArtifactStorageAdapter`（future）。

职责边界：

```
ArtifactRepository / Store
    → 只管 artifact 的 put/get/exists/query/verify

ArtifactEngine / Builders
    → 只管如何从 interaction、memory create、memory update 构造 artifact

MemoryGenerationFamiliar
    → 编排 compute → artifact build/store → attach refs → memory persist
```

因此，ArtifactStore 应从 ArtifactEngine 私有资源上升为 `MemoryLibrary` 管理的附属持久化仓库；但它不参与短期 / 中期 / 长期记忆状态转移，也不污染三级记忆模型。

---

## 3. RetrievalFamiliar 能力增强

### 3.1 目标

将 `RetrievalFamiliar` 从"仅限中期向量检索"扩展为**三层统一检索入口**，同时承接从 `LibrarianCore` 迁出的 topic context 查询职责。

**依赖变更**：Phase 3 以 `memory_library: MemoryLibrary` **替换** `storage: QdrantMemoryStore`。所有层的查询均通过 `self._memory_library` 访问对应 Store，不直接持有任何 Port 或存储实现。

```python
def __init__(
    self,
    engine: RetrievalEngine,
    memory_library: MemoryLibrary,
    passive_renderer: Optional[BaseContextRenderer] = None,
    local_bus: Optional[Any] = None,
):
    ...
    self._memory_library = memory_library
```

`storage: QdrantMemoryStore` 为 Phase 1-2 遗留参数，随本次重构移除。现有两处 `self.storage` 用途迁移至 `mid_term`：

| 原调用 | 迁移后 |
|---|---|
| `storage.get_memory_by_alias(alias, user_id)` | `memory_library.mid_term.get_by_alias(alias, user_id)` |
| `storage.update_access_info(memory.id)` | `memory_library.mid_term` 需补充 `update_access_info(memory_id)` ，适配器包装现有实现 |

### 3.2 迁移项

| 来源 | 迁移内容 | 目标层 | 状态 |
|---|---|---|---|
| `LibrarianCore.get_active_topics_snapshots` | `list_active_topics` | 短期检索 | Phase 3 |
| `LibrarianCore` / 感知层 `prepare_topic` | `get_topic` | 短期检索 | Phase 3 |
| 现有 retrieval engine | 精确取回（alias）、语义检索 | 中期检索 | 已实现 |
| `self.storage`（Phase 1-2 遗留） | `get_by_alias` / `update_access_info` 切换至 `mid_term` | 中期检索 | Phase 3 清理 |
| 待实现 | `query_archive`、`is_archived` | 长期检索 | Phase 3 |

### 3.3 `list_active_topics`

#### 现状与问题

`LibrarianCore.get_active_topics_snapshots()`（librarian.py:262）持有感知层引用，通过 `getattr(self.perception_layer, "short_term_store", None)` 越权绕过架构边界直接访问 `ShortTermMemoryStore`：

```python
store = getattr(self.perception_layer, "short_term_store", None)
if store is None:
    return []
topic_data = store.list_topic_data(user_id=identity.user_id, include_empty=False)
return [t.to_topic_snapshot() for t in topic_data]
```

#### 新方法设计

```python
# retrieval.py — RetrievalFamiliar
def list_active_topics(
    self,
    identity: Identity,
    *,
    include_empty: bool = False,
    sort_by_access: bool = True,
) -> List[TopicSnapshot]:
    """
    列出指定用户的话题快照（短期检索入口）。

    兼顾两个迁移来源：
    - get_active_topics_snapshots: include_empty=False（默认）
    - get_topic_pool_snapshot: include_empty=True, sort_by_access=True

    通过 MemoryLibrary.short_term 访问，不穿透到 PerceptionLayer。
    返回的 TopicSnapshot 已包含 block_count / last_accessed_at（见 §3.6）。
    """
    topics = self._memory_library.short_term.list_topic_data(
        user_id=identity.user_id,
        include_empty=include_empty,
        deep_copy=False,
    )
    if sort_by_access:
        topics = sorted(topics, key=lambda t: t.last_accessed_at, reverse=True)
    return [t.to_topic_snapshot() for t in topics]
```

方法为同步调用，返回 `List[TopicSnapshot]`，与迁移前的 `get_active_topics_snapshots` 签名保持兼容。`get_topic_pool_snapshot` 的调用方改用 `include_empty=True` 变体（详见 §3.6）。

#### 迁移步骤

1. `RetrievalFamiliar.__init__` 注入 `memory_library`（见 §3.1）
2. 添加 `list_active_topics(identity)` 方法
3. `LibrarianCore.get_active_topics_snapshots` 删除，路由方法应同步改为使用 `retrieval_familiar.list_active_topics(identity)`
4. `LibrarianCore` 不再需要持有 `perception_layer` 仅为此方法；`getattr` 模式随迁移删除
5. `SemanticFlowPerceptionLayer.get_topic_pool_snapshot` 删除，`StreamPrelude` 构建处改为调用 `list_active_topics(identity, include_empty=True)`（见 §3.6）

#### 调用方影响

过渡期内 `PatchouliLocalRoutes.GET_ACTIVE_TOPICS_SNAPSHOTS` 可继续保持兼容签名，但 handler 应直接切换至 `RetrievalFamiliar.list_active_topics(identity)`。第 5 章 public API 与总线契约迁移完成后，`patchouli/application/topic_management_service.py` 不再直接持有 `RetrievalFamiliar`，而是通过 `topic.list_active` local route 请求该能力。

### 3.4 topic_context compat dict 迁移

两处 TODO（semantic_flow_perception_layer.py:389、librarian.py:164）均将 `TopicData` 降格为 `Dict[str, Any]` 对外传递。两条调用链路独立，可分别推进。

#### Path 1：prepare_topic → 前台 Agent 历史上下文

**现状调用链**

```
SemanticFlowPerceptionLayer.prepare_topic()    # [TODO:389] 构建 compat dict
  → return (topic_id, pool_snapshot, topic_context: Dict)
LibrarianCore.prepare_topic()                  # 薄委托
  → PatchouliLocalRoutes.PREPARE_TOPIC（总线）
PatchouliService.prepare_agent_run()
  → AgentRunContext(topic_context=topic_context)   # core/protocol/models.py:156
prompts/assembler.py
  → topic_context.get("state_summary") → SystemPromptBuilder.with_topic_state()
  → topic_context.get("blocks")        → PerceptionContextConverter.blocks_to_messages()
```

**迁移步骤**

1. `SemanticFlowPerceptionLayer.prepare_topic()` 直接返回 `TopicData`（或 `None`），删除 compat dict 构建逻辑：
   ```python
   topic_data = self._short_term_store.get_topic_data(topic_id)
   return topic_id, pool_snapshot, topic_data   # TopicData | None，不再包装为 dict
   ```
2. `AgentRunContext.topic_context` 字段类型由 `Dict[str, Any]` 改为 `Optional[TopicData]`
3. `prompts/assembler.py` 更新两处访问：
   ```python
   # 旧
   topic_context.get("state_summary", "")
   topic_context.get("blocks", [])
   # 新
   context.topic_context.state_summary if context.topic_context else ""
   context.topic_context.recent_blocks(5) if context.topic_context else []
   ```

**涉及文件**：`semantic_flow_perception_layer.py`、`core/protocol/models.py`、`prompts/assembler.py`

#### Path 2：finalize_agent_run → 记忆生成背景信息

**现状调用链**

```
PatchouliService.finalize_agent_run()
  → bus.request(generation.submit_active, tasks, topic_id)
      → MemoryGenerationCoordinator 构建 WRITE / UPDATE specs
      → bus.request(memory_task.submit_generation_many, specs)
      → MemoryGenerationTaskController 创建任务
      → bus.request(generation.execute_spec, spec)
      → MemoryGenerationFamiliar.execute(spec)
```

**迁移步骤**

1. `LibrarianCore.run_active_generation()` 不再作为主动生成入口；入口迁移为 `generation.submit_active` local route。
2. 若构建 active generation spec 需要短期 topic 上下文，优先由调用方传入 `TopicData` 或必要快照；必须在生成链路内部补齐时，通过 local bus 请求 `topic.get`，不得直接持有 `RetrievalFamiliar` 或 perception store：
   ```python
   topic_data = await bus.request("topic.get", topic_id)
   ```
3. `ArtifactEngine.interaction.build_and_store()` 直接取字段（`topic_data.topic_title` 等），或重载接受 `Optional[TopicData]`。
4. `GenerationContext` 构建迁入 `MemoryGenerationFamiliar` 或 spec builder，使用 `topic_data.recent_blocks(5)` 和 `topic_data.state_summary`。

**涉及文件**：`librarian.py`（删除入口）、`memory_generation_coordinator.py`（新增）、`memory_generation.py`（新增 Familiar）

两条路径均无需改动 `ShortTermMemoryStore` 或 `TriggerManager`，感知层解耦（§2.4）对这两处无阻塞关系。

### 3.5 ShortTermMemoryStore 读方法 deep_copy 开关

#### 现状

`ShortTermMemoryStore._to_topic_data()` 目前对所有 block 无条件执行深拷贝：

```python
blocks=tuple(block.model_copy(deep=True) for block in buf.blocks)
```

这对需要操作 block 数据的写路径是必要的，但对纯读路径（如 `list_active_topics` 只消费 `TopicSnapshot`，根本不暴露 block 内容）是无效开销。

#### 方案

`get_topic_data` / `list_topic_data` 增加 `deep_copy: bool = True` 参数，下穿至 `_to_topic_data`：

```python
def get_topic_data(self, topic_id, *, touch=True, deep_copy=True) -> Optional[TopicData]: ...
def list_topic_data(self, user_id=None, *, include_empty=True, deep_copy=True) -> List[TopicData]: ...

def _to_topic_data(self, buf, *, deep_copy=True) -> TopicData:
    return TopicData(
        ...
        blocks=tuple(block.model_copy(deep=True) for block in buf.blocks)
               if deep_copy else tuple(buf.blocks),
        ...
    )
```

默认值保持 `True`，对现有调用方无感知变更。

#### 调用规则

| 调用场景 | `deep_copy` | 理由 |
|---|---|---|
| `list_active_topics` → `list_topic_data` | `False` | 只消费 `to_topic_snapshot()`，block 不对外暴露 |
| `get_topic` → `get_topic_data` | `False` | 只读取 summary/title/token count |
| `prepare_topic` / `run_active_generation` 拿 blocks 传下游 | `True`（默认） | blocks 会进入 generation context，需隔离 |
| `TriggerManager` 内部 compact/fold | `True`（默认） | 需要修改 block 列表 |

### 3.6 TopicSnapshot 扩展与 get_topic_pool_snapshot 迁移

#### 字段扩展

`TopicSnapshot` 新增两个标量字段，使其能够承接 pool snapshot 的前端展示需求：

```python
class TopicSnapshot(BaseModel):
    topic_id: str
    topic_title: str
    topic_summary: str = ""
    state_summary: str = ""
    last_turn: Optional[Dict[str, str]] = None
    total_tokens: int = 0
    block_count: int = 0           # 新增
    last_accessed_at: float = 0.0  # 新增
```

`TopicData.to_topic_snapshot()` 填充这两个字段：

```python
def to_topic_snapshot(self) -> TopicSnapshot:
    ...
    return TopicSnapshot(
        ...,
        block_count=self.block_count,
        last_accessed_at=self.last_accessed_at,
    )
```

#### 归属迁移

`TopicSnapshot` 从 `engines/perception/models.py`（感知层内部）迁移至 `core/models.py`（`Identity`、`TurnRecord` 的所在位置）。`engines/perception/models.py` 保留 re-export 向后兼容。

迁移后的消费方分布：

| 消费方 | 主要用到的字段 |
|---|---|
| TheEye 话题路由 | `topic_id`, `topic_title`, `state_summary`, `last_turn`, `total_tokens` |
| `list_active_topics`（默认） | `topic_id`, `topic_title`, `topic_summary`, `state_summary` |
| StreamPrelude / pool snapshot | 全部字段，`last_accessed_at` 用于排序，`block_count` 用于展示 |

#### get_topic_pool_snapshot 迁移

`SemanticFlowPerceptionLayer.get_topic_pool_snapshot()` 废弃。调用方改为：

```python
retrieval_familiar.list_active_topics(identity, include_empty=True)
# sort_by_access=True 默认已启用，last_accessed_at 降序
```

pool wrapper 中的容器元数据（`max_resident_topics`、`current_count`）不属于 per-topic 信息，迁入 `StreamPrelude`：

```python
# 当前：pool_snapshot: Dict[str, Any]
# 迁移后：
pool_topics: List[TopicSnapshot] = Field(default_factory=list)
max_resident_topics: int = 0
# current_count 由 len(pool_topics) 推导，不单独存储
```

**涉及文件**：`core/models.py`（新增 TopicSnapshot）、`engines/perception/models.py`（保留 re-export）、`patchouli/memory_library/models.py`（`to_topic_snapshot` 填充新字段）、`semantic_flow_perception_layer.py`（删除 `get_topic_pool_snapshot`）、`patchouli/models.py`（StreamPrelude 字段替换）

---

## 4. LibrarianCore 编排职责解构

### 4.1 目标

本章的实施重心不是给旧 `LibrarianCore` 的每个方法找一个新函数，而是补齐它过去隐式承担的分层：

1. 将 `LibrarianCore` 代行的感知代理职责拆为 `PerceptionFamiliar`。
2. 将读、写、生命周期、感知四类业务能力收束为同级 Familiar。
3. 将生成入口归一化与任务生命周期管理明确为控制面组件，而不是业务 Familiar。
4. 所有跨业务能力调用统一通过 `PatchouliLocalBus`，避免 service 之间互相依赖注入。
5. 保持现有 public API 形态基本稳定，application service 的 bus-only 改造、Bridge 收窄、LocalRoutes 完整补齐后置到第 5 章。

### 4.2 分层模型

`LibrarianCore` 解构后的 Patchouli 内部组件分为四层：

```text
PatchouliRuntime（装配根）
    → 构建 engine / store / familiar / control plane
    → 注册 local routes
    → 管理 warmup / health / shutdown drain

控制面 / 应用编排层
    → MemoryGenerationCoordinator
    → MemoryGenerationTaskController
    → 只持有 local bus、task registry、event sink 等编排依赖

Familiar 业务能力层
    → PerceptionFamiliar
    → RetrievalFamiliar
    → MemoryGenerationFamiliar
    → LifecycleFamiliar
    → 同级业务门面，互不注入，跨能力通信走 local bus

Engine / Store 层
    → PerceptionLayer / TriggerManager
    → RetrievalEngine
    → MemoryGenerationEngine
    → MemoryLifecycleEngine
    → MemoryLibrary / Store / Port
    → 不持有 bus，不知道上层业务组件
```

依赖规则：

- Runtime 是唯一装配根，可以知道所有组件并负责 route 挂载。
- Familiar 是业务能力实现，不得在构造函数或字段中直接持有另一个 Familiar。
- `MemoryGenerationCoordinator`、`MemoryGenerationTaskController` 属于控制面，不是 Familiar；它们也不得直接持有任何 Familiar。
- Engine / Store 不持有 bus，不调用 route，不反向依赖 Familiar 或控制面。
- 跨 Familiar、控制面到 Familiar、Familiar 到控制面的调用都通过 local bus 完成。
- 允许多个 Familiar 共同依赖底层稳定能力，例如 `MemoryLibrary`、各自 engine、store port、artifact engine；这些底层依赖不得反向持有 Familiar。

### 4.3 PerceptionFamiliar

新增 `PerceptionFamiliar`，承接旧 `LibrarianCore` 对感知层的代理职责。原因是感知层本身是 engine 实现，不应持有总线，也不应知道 generation / task / public API。旧系统中这个边界由 `LibrarianCore` 代行；拆掉 `LibrarianCore` 后，必须显式补上这一层。

职责：

```text
PerceptionFamiliar
    → 持有 PerceptionLayer
    → 对 local bus 暴露 ingestion / topic 能力
    → 向 PerceptionLayer 注入 archive callback
    → callback 内部通过 local bus 请求 generation.submit_archive
```

承接的能力：

| 方法 | 说明 |
|---|---|
| `submit_interaction(payload, target_topic_id)` | 调用 `PerceptionLayer.route_and_ingest` |
| `prepare_topic(...)` | 调用 `PerceptionLayer.prepare_topic` |
| `manual_archive_topic(...)` | 调用 `PerceptionLayer.manual_trigger` |
| `evict_topic(...)` / `discard_if_empty(...)` | 调用感知层 topic 管理能力 |
| `_on_archive_payload(payload)` | 感知层 callback，内部 `bus.request(generation.submit_archive, payload)` |

`PerceptionLayer` / `TriggerManager` 仍然只接收 callback，不持有 bus：

```text
PerceptionLayer.route_and_ingest()
    → TriggerManager archive_topic
    → callback(ArchivePayload)
    → PerceptionFamiliar._on_archive_payload()
    → local bus: generation.submit_archive
```

这样感知引擎保持底层纯净，跨业务通信发生在 Familiar 层。

### 4.4 MemoryGeneration 控制面与业务面

生成链路拆成三个角色：

```text
MemoryGenerationCoordinator   # 控制面：入口归一化
MemoryGenerationTaskController # 控制面：任务生命周期
MemoryGenerationFamiliar      # Familiar：生成写入业务能力
```

三者不直接互相注入。调用链通过 local bus 串联：

```text
generation.submit_archive / submit_active / submit_evolution
    → MemoryGenerationCoordinator
        → local bus: memory_task.submit_generation / submit_generation_many
            → MemoryGenerationTaskController
                → local bus: generation.execute_spec
                    → MemoryGenerationFamiliar.execute(spec)
```

#### MemoryGenerationCoordinator

`MemoryGenerationCoordinator` 只负责 raw input → `MemoryGenerationTaskSpec`，不执行生成、不创建 task、不持久化、不发布事件、不直接调用 Familiar。

| Builder | 输入 | 输出 |
|---|---|---|
| `_build_archive_spec()` | `ArchivePayload` | `MemoryGenerationTaskSpec(source=ARCHIVE, ...)` |
| `_build_active_specs()` | `PendingAtomMaterializeTask[] + topic_id` | `MemoryGenerationTaskSpec(source=WRITE/UPDATE, ...)[]` |
| `_build_evolution_specs()` future | split / merge plan | `MemoryGenerationTaskSpec(source=SPLIT/MERGE, ...)[]` |

产出 spec 后只通过 bus 进入 Controller：

```python
await self._bus.request("memory_task.submit_generation", spec)
await self._bus.request("memory_task.submit_generation_many", specs)
```

如果构建 spec 需要补齐短期 topic 或已有 memory，优先由调用方传入必要上下文；必须跨域读取时，Coordinator 也只能通过 local bus 请求 `topic.get` / `memory.get`，不得直接持有 `RetrievalFamiliar` 或 store。

#### MemoryGenerationTaskController

`MemoryGenerationTaskController` 是任务控制面，不是业务类。它只接收规范化后的 `MemoryGenerationTaskSpec`：

```text
MemoryGenerationTaskController
    → submit_generation(spec)
    → submit_generation_many(specs)
    → _create_and_run_task(spec)
    → _run_task(memory_task, spec)
    → _start_task() / _finish_task()
```

`_run_task()` 内部通过 bus 请求生成执行：

```python
results = await self._bus.request("generation.execute_spec", spec)
```

Controller 可以处理 task registry、取消、失败、settlement、runtime event、task snapshot，但不应知道 WRITE 如何组装 `GenerationRequest`、UPDATE 如何读取目标记忆、ARCHIVE 如何构建上下文。这些属于 `MemoryGenerationFamiliar.execute(spec)`。

#### MemoryGenerationFamiliar

`MemoryGenerationFamiliar` 是生成写入业务门面，负责执行统一 spec：

```text
MemoryGenerationFamiliar
    → MemoryGenerationEngine：生成、更新、查重决策等算法计算
    → ArtifactEngine：interaction / creation / version artifact 构建
    → MidTermMemoryStore / MemoryLibrary.mid_term：提交 CREATE / UPDATE 结果
```

承接当前 `MemoryGenerationTaskController` 中的数据面方法：

| 当前方法 | 迁移后归属 | 说明 |
|---|---|---|
| `_run_mode_b()` | `MemoryGenerationFamiliar` | MTP WRITE 请求组装与执行 |
| `_run_mode_c()` | `MemoryGenerationFamiliar` | MTP UPDATE 目标读取、请求组装与执行 |
| `_run_generation()` | `MemoryGenerationFamiliar` | compute → artifact → persist 主流程 |
| `_build_memory_artifacts()` | `MemoryGenerationFamiliar` | creation / version artifact 构建与 refs 挂载 |

主入口：

```python
async def execute(self, spec: MemoryGenerationTaskSpec) -> list[MemoryGenerationResult]:
    ...
```

输出契约必须保持纯结果返回：**所有方法只返回 `MemoryGenerationResult` / result list，不直接发布事件**。当前 `_run_generation()` 在 persist 后直接 publish `PENDING_ATOM_SETTLED` 的职责应迁移到 Controller 或更外层任务编排层。

短期到中期的写入不命名为 archive。`SemanticBuffer / TopicData → MemoryAtom` 是 materialize / consolidate 过程，属于 `MemoryGenerationFamiliar`；`archive` 保留给中期到长期的生命周期归档。

### 4.5 生成链路时序

被动话题归档：

```text
PerceptionFamiliar.submit_interaction()
    → PerceptionLayer.route_and_ingest()
        → TriggerManager archive_topic
        → callback ArchivePayload
    → PerceptionFamiliar._on_archive_payload()
    → bus.request(generation.submit_archive, payload)
    → MemoryGenerationCoordinator builds ARCHIVE spec
    → bus.request(memory_task.submit_generation, spec)
    → MemoryGenerationTaskController creates task
    → bus.request(generation.execute_spec, spec)
    → MemoryGenerationFamiliar.execute(spec)
    → Controller handles task result / event
```

主动 MTP 生成：

```text
PatchouliService / future public application service
    → bus.request(generation.submit_active, materialize_tasks, topic_id)
    → MemoryGenerationCoordinator builds WRITE / UPDATE specs
    → bus.request(memory_task.submit_generation_many, specs)
    → MemoryGenerationTaskController creates tasks
    → bus.request(generation.execute_spec, spec)
    → MemoryGenerationFamiliar.execute(spec)
    → Controller handles task result / settlement event
```

维护期 split / merge（future）：

```text
LifecycleFamiliar.run_gardening_once()
    → MemoryLifecycleEngine refresh / reinforce / GC
    → MemoryEvolutionEngine plans split / merge
    → bus.request(generation.submit_evolution, plan)
    → MemoryGenerationCoordinator builds SPLIT / MERGE specs
    → bus.request(memory_task.submit_generation_many, specs)
    → MemoryGenerationTaskController creates tasks
    → bus.request(generation.execute_spec, spec)
    → MemoryGenerationFamiliar.execute(spec)
```

### 4.6 LifecycleFamiliar

生命周期相关职责先从 `LibrarianCore` 中拆出为独立业务类，命名为 `LifecycleFamiliar`。

命名理由：

- 与 `PerceptionFamiliar` / `RetrievalFamiliar` / `MemoryGenerationFamiliar` 对齐：四者都是围绕单个能力域的 Patchouli 内部业务门面，向 local bus 暴露可组合能力。
- 比 `MemoryGardener` 更宽：gardening 只覆盖定时维护 / GC，无法表达 hit、citation、feedback、revive 等事件响应。
- 避免与 `MemoryLifecycleEngine` 混淆：`Engine` 负责算法和策略，`Familiar` 负责业务入口、事件响应与总线挂载。

职责边界：

```
LifecycleFamiliar
    → MemoryLifecycleEngine：生命力计算、事件强化、GC 策略
    → MemoryLibrary：需要跨层状态转移时调用 archive / revive
```

`LifecycleFamiliar` 承接以下业务入口：

| 方法 | 说明 |
|---|---|
| `run_gardening_once()` | 从 `LibrarianCore` 迁入，作为全局维护调度器的单次 lifecycle 维护入口 |
| `refresh_memory_vitality(memories, persist=False)` | 当前 runtime local handler 下沉，用于检索结果展示前刷新生命力 |
| `record_hit(memory_id, source)` | 响应检索命中事件，例如 active finalize 后记录 retrieval hit |
| `record_citation(memory_id, source)` | 响应 MTP READ / RUN 等主动引用事件 |
| `record_feedback(memory_id, positive, source)` | 响应 UI 或用户反馈事件 |
| `revive_memory(memory_id)` | 长期记忆复活入口，底层通过 `MemoryLibrary.revive()` 执行跨层状态转移 |

`archive_memory(memory_id)` 暂不固定为 `LifecycleFamiliar` 的公开方法。它涉及“谁负责触发中期到长期的归档决策”的边界，需要与后续 `MemoryGenerationFamiliar` 和 `MemoryLibrary` 状态转移链路一起确定。现阶段只确认：短期到中期是生成写入，不命名为 archive；如果 lifecycle 维护链路需要将中期记忆移入长期层，应通过 `MemoryLibrary.archive()` 执行跨层搬运，而不是让 `MemoryLifecycleEngine` 或 LongTermStore 自行穿透中期存储。

归档查询不放入 `LifecycleFamiliar`。`list_archived()` / `is_archived()` 这类读取能力归 `RetrievalFamiliar`，因为 RetrievalFamiliar 在本次重构中的定位是三级记忆的统一读操作者。

#### 生命力维护链路

```
定时触发 / 事件触发
    → LifecycleFamiliar
        → MemoryLifecycleEngine（vitality 衰减 / 强化 / GC）
        → MemoryLibrary（必要时执行 archive / revive 状态转移）
```

`MemoryLifecycleEngine` 不拆分，整体作为生命周期算法/策略执行单元；但对外 local route、事件响应和调度器入口均收束到 `LifecycleFamiliar`，不再经由 `LibrarianCore` 中转。

调度器回调同步从：

```python
runtime.librarian_core.run_gardening_once
```

迁移为：

```python
runtime.lifecycle_familiar.run_gardening_once
```

#### 与 v0.9.0 记忆 split / merge 的关系

需要区分两类 merge：

1. 写入期 merge：当前 generation / dedup 中的 merge 属于生成写入链路，回答“新写入内容与已有记忆是否重复、是否应合并”。它仍归 `MemoryGenerationFamiliar` / Generation 体系，不归 `LifecycleFamiliar`。
2. 维护期 split / merge：v0.9.0 计划中的记忆分裂与合并如果用于整理已有记忆库结构，则属于生命周期维护的子域，但算法复杂度足够高，不应塞进 `MemoryLifecycleEngine`。

未来引入维护期 split / merge 时，新增独立引擎，例如：

```python
MemoryEvolutionEngine
```

其接入方式：

```
LifecycleFamiliar.run_gardening_once()
    → MemoryLifecycleEngine：刷新 vitality / 强化 / GC
    → MemoryEvolutionEngine（future）：规划并执行 split / merge
    → local bus: generation.submit_evolution
    → 生成控制面创建 SPLIT / MERGE task
    → MemoryGenerationFamiliar 执行必要的生成写入
```

即：split / merge 属于 lifecycle 维护域，但以独立引擎接入；`LifecycleFamiliar` 负责调度和业务边界，不承载算法实现。

### 4.7 Local Routes 切换

| Local route | 迁移前 | 迁移后 |
|---|---|---|
| `ingestion.submit_interaction` | `LibrarianCore.submit_interaction` | `PerceptionFamiliar.submit_interaction` |
| `topic.prepare` | `PerceptionLayer.prepare_topic` | `PerceptionFamiliar.prepare_topic` |
| `topic.manual_archive` | `PerceptionLayer.manual_trigger` | `PerceptionFamiliar.manual_archive_topic` |
| `generation.submit_archive` | `LibrarianCore._on_generate_memory` | `MemoryGenerationCoordinator.submit_archive` |
| `generation.submit_active` | `LibrarianCore.run_active_generation` | `MemoryGenerationCoordinator.submit_active` |
| `generation.execute_spec` | 无 | `MemoryGenerationFamiliar.execute` |
| `memory_task.submit_generation` | `MemoryGenerationTaskController.run_archive_generation` / `run_active_generation` | `MemoryGenerationTaskController.submit_generation` |
| `memory_task.submit_generation_many` | `MemoryGenerationTaskController.run_active_generation` | `MemoryGenerationTaskController.submit_generation_many` |
| `lifecycle.run_gardening_once` | `LibrarianCore.run_gardening_once` | `LifecycleFamiliar.run_gardening_once` |

### 4.8 迁移顺序

1. 保持现有 public API 与 application service 构造方式基本不动，避免先铺设缺少自然 handler 的完整 local routes。
2. 引入 `PerceptionFamiliar`，承接 `submit_interaction`、topic 管理、感知层 callback 注入；callback 内部通过 bus 请求 `generation.submit_archive`。
3. 引入 `LifecycleFamiliar`，承接 `run_gardening_once`、vitality refresh、hit / citation / feedback 响应与 revive 入口；归档写入入口 `archive_memory(memory_id)` 仍保持待定。
4. 引入 `MemoryGenerationFamiliar`，承接 `_run_mode_b()`、`_run_mode_c()`、`_run_generation()`、`_build_memory_artifacts()` 等生成执行数据面；所有方法只返回 result，不直接发布事件。
5. 引入 `MemoryGenerationTaskSpec`，将 `MemoryGenerationTaskController` 收缩为统一 `submit_generation(spec)` / `submit_generation_many(specs)` / `_run_task(memory_task, spec)`；`_run_task` 通过 bus 请求 `generation.execute_spec`。
6. 引入 `MemoryGenerationCoordinator`，负责把 `ArchivePayload`、`PendingAtomMaterializeTask` 与 future split / merge plan 转为 `MemoryGenerationTaskSpec`，再通过 bus 请求 `memory_task.submit_generation*`。
7. 将 scheduler / runtime 中指向 `LibrarianCore.run_gardening_once` 的引用切换到 `LifecycleFamiliar.run_gardening_once`。
8. 删除 `LibrarianCore`，从 runtime 服务图中移除。

---

## 5. Public API 与总线契约迁移

第 5 章在 `LibrarianCore` 解构完成后执行。此时 lifecycle、generation、task、retrieval、perception 等能力都有稳定业务落点，public API 可以逐个迁移到 bus-only application service，而不需要把临时业务逻辑塞进 `PatchouliRuntime`。

本章的核心判断是：**`patchouli.application.*` 与 `PatchouliService` 中对外公开的方法，均应视为 public API handler。Patchouli 内部业务不应依赖这些 public API 方法；内部协作只通过 local route 请求领域能力原语。**

这不意味着 application service 被放到 Patchouli 外部。它仍然是 Patchouli 子系统内部对象，只是由 `PatchouliBridge` 把其 public 方法挂载到 `GlobalSystemBus`，供外层调用方访问。完整调用链应为：

```
system / server / Alice
  → GlobalSystemBus / PatchouliRoutes
      → PatchouliBridge（挂载 public route）
          → Patchouli public application service method
              → PatchouliLocalBus / local primitive routes
                  → Familiar / controller / runtime primitive / domain handler
```

### 5.1 迁移原则

1. 先确定 API 的自然归属，再补 local route。不得为了满足 bus-only 形式而在 runtime 中实现业务逻辑。
2. `PatchouliRuntime` 只做装配、health、warmup、shutdown drain、route handler 注册，不承载 public use-case。
3. `PatchouliBridge` 只做公开路由名称翻译和转发，不直接绑定 engine / store / controller。
4. `PatchouliLocalRoutes` 只表达 Patchouli 内部可组合能力，不镜像完整 public workflow。
5. `patchouli.application.*` 作为 public use-case 层，只持有 `PatchouliBus`。
6. public API 方法负责组合 local primitives；不得让 public API 方法请求一个同名 local handler 来绕开自身复杂度。
7. 如果某个能力会被 Patchouli 内部复用，应抽成 local primitive；如果只面向 system / server / Alice 等外部调用方，则保留为 public route。

换言之，正确方向是：

```
GlobalRoute
  → public application service method
      → local primitive routes
          → Familiar / controller / domain handler
```

错误方向是：

```
GlobalRoute
  → public application service method
      → local route with same public workflow name
          → runtime 临时 handler / public service 自身
```

后者会让 runtime 变成业务 handler 容器，也会让 local routes 退化为 public routes 的镜像。

### 5.2 当前边界问题

#### Bridge 越权

当前 `PatchouliBridge` 直接绑定多类具体对象的方法：

```
GlobalRoutes
  → PatchouliBridge
      → PatchouliService / PatchouliRuntime / LibrarianCore
      → MemoryManagementService / TopicManagementService / ModelReadinessService
      → storage / task_controller / perception_layer / retrieval_familiar
```

这使 bridge 同时知道公开 API、运行时对象图和具体 handler 归属。正确职责应收束为：

```
GlobalRoutes
  → PatchouliBridge
      → Patchouli public application service（只持有 local bus）
          → PatchouliLocalRoutes
              → runtime/domain handlers
```

Bridge 可以持有一个 route map 或一组 public application service，但这些 service 必须是 bus-only；Bridge 不应再直接引用 `runtime.librarian_core`、`runtime.retrieval_familiar`、`runtime._task_controller`、`runtime._engines[...]` 或 `storage`。

#### LocalRoutes 不完整且混入 public workflow

当前 local routes 主要覆盖 `PatchouliService.prepare_agent_run()` 内部需要的少数能力，例如 `GET_AGENT_PROFILE`、`PREPARE_TOPIC`、`MEMORY_RETRIEVE`。但 memory/task/topic/profile/model 等 public application service 仍直接持有具体组件，说明 local bus 没有覆盖 application service 所需的完整能力原语。

同时，`PREPARE_AGENT_RUN`、`FINALIZE_AGENT_RUN`、`CLEANUP_PREPARED_AGENT_RUN` 是面向 system 层 `ChatApplicationService` 的完整工作流入口，Patchouli 内部目前不会通过 local bus 请求这些 route。它们应保留为 public routes，不应作为 local routes 常量存在。

`ANALYZE_AND_RETRIEVE` 也需要按同一规则重新评估：如果它只作为外部 passive ingress / prepare 前置能力使用，则应保留在 public API 层，由 application service 内部组合 `gateway.gaze` 与 `memory.retrieve`；只有当 Patchouli 内部业务确实需要复用“分析并检索”这个组合能力时，才保留对应 local route。

#### PatchouliService / Application Services 持有 runtime

`PatchouliService` 现在既持有 `PatchouliRuntime` 又持有 `TheEye`，并在方法中直接访问 `runtime.retrieval_familiar`、`runtime.memory_library`、`runtime.librarian_core`、`runtime.check_storage_health()`。但它还混有 `get_memory_task` 等本应归属 controller / application service 的 API，因此应在本章先剥离这些代理 API，再处理剩余主动交互 workflow。

`patchouli.application.*` 也直接持有 `storage`、`lifecycle_engine`、`task_controller`、`perception_layer`、`retrieval_familiar`、`runtime` 等实现层对象。这会导致 public use-case 层与 runtime 装配细节耦合，继续扩大 API 迁移面。

### 5.3 分层契约

#### GlobalSystemBus：跨子系统公开契约

`GlobalSystemBus` 只服务跨子系统调用与 HTTP/API 入口。典型调用方包括：

- system application service（如 `ChatApplicationService`、`MemoryApplicationService`）
- Alice 子系统（MTP / profile resolver / alias resolver）
- server router 间接调用的 system application service

这些调用方只能看到 `PatchouliRoutes` / `GlobalRoutes.PATCHOULI_*`，不能知道 Patchouli 内部由哪个 engine、service 或 store 执行。

#### PatchouliBridge：公开路由转发层

Bridge 是 Patchouli public API 的挂载适配器。它负责把 Patchouli 内部的 bus-only public application service 方法注册到 `GlobalSystemBus`，使外层调用方只能通过 `PatchouliRoutes` 访问 Patchouli。

Bridge 的职责：

- 注册 / 卸载 `PatchouliRoutes`
- 将 `PatchouliRoutes.X` 转发到 bus-only public application service 的方法
- 将 Patchouli local events 选择性转发为 global events

Bridge 不负责：

- 选择 runtime 内部具体组件
- 拼装业务流程
- 调用 storage / engine / controller
- 暴露 local route 给外部调用方

因此，Bridge 可以知道“某个 public route 对应哪个 public application service 方法”，但不能知道“这个方法内部组合了哪些 local routes”。

#### PatchouliLocalBus：子系统内部能力总线

Local bus 仅供 Patchouli 子系统内部对象使用。这里的“内部对象”包括 Familiar、controller、coordinator、runtime primitive，以及 public application service。外部调用方不能直接看到或请求 `PatchouliLocalRoutes`。

Local bus 是 Patchouli 子系统内业务代码、控制面组件、public application service 的唯一跨组件通信入口。它需要覆盖三类能力：

1. Familiar 之间的业务通信，例如 RetrievalFamiliar 请求 lifecycle 刷新生命力。
2. 控制面到业务能力的调用，例如 TaskController 请求 `generation.execute_spec`。
3. public application service 编排所需的内部能力，例如 memory CRUD、topic prepare、model readiness、agent profile 查询。

Local bus 上的 route 应是可组合的领域能力原语，而不是完整外部 workflow。

public application service 可以调用 local bus，但这不意味着 public API 本身要出现在 local bus 上。它只是 local primitives 的调用方之一，职责是对外 API 的 use-case 编排。换言之，`prepare_agent_run()` 可以组合 `topic.prepare`、`memory.retrieve`、`runtime.storage_health`，但不应存在 `service.prepare_agent_run` 作为同名 local workflow。

#### Runtime / Domain：实现层

`PatchouliRuntime` 仍是装配根，负责：

- 构建 MemoryLibrary、engine、PerceptionFamiliar、RetrievalFamiliar、MemoryGenerationCoordinator、MemoryGenerationTaskController、MemoryGenerationFamiliar、LifecycleFamiliar
- 注册 local route handler
- 管理模型预热、存储健康、shutdown drain、维护任务注册等 runtime 行为

Domain service / engine 可以直接持有其必要依赖，但这些依赖不向 application/bridge 层泄漏。

### 5.4 LocalRoutes 重整

#### 删除 public workflow 镜像

以下 local routes 从 `PatchouliLocalRoutes` 移除：

| Local route | 原因 |
|---|---|
| `PREPARE_AGENT_RUN` | 完整 public workflow，仅由 system `ChatApplicationService` 通过 global route 调用 |
| `FINALIZE_AGENT_RUN` | 同上 |
| `CLEANUP_PREPARED_AGENT_RUN` | 同上 |
| `ANALYZE_AND_RETRIEVE` | 若仅面向 passive ingress / public API，则不作为 local primitive；改由 public service 组合 `gateway.gaze + memory.retrieve` |

这些 route 继续作为 `PatchouliRoutes` / `GlobalRoutes.PATCHOULI_*` 存在，由 public application service 使用 local primitives 编排实现。

不得为了删除直接 runtime 依赖而新增 `service.prepare_agent_run`、`service.finalize_agent_run` 这类同名 local handler。public workflow 的复杂度属于 public application service；local bus 只承载 workflow 内部可复用的能力节点。

#### 补齐内部能力原语

新增或规范化以下 local routes。命名建议按领域分组，避免 `service.*` 这类含糊前缀。

| 分组 | Local route | Handler 归属 |
|---|---|---|
| gateway | `gateway.gaze` | `TheEye.gaze` |
| ingestion | `ingestion.submit_interaction` | `PerceptionFamiliar.submit_interaction` |
| generation | `generation.submit_archive` | `MemoryGenerationCoordinator.submit_archive` |
| generation | `generation.submit_active` | `MemoryGenerationCoordinator.submit_active` |
| generation | `generation.submit_evolution` | `MemoryGenerationCoordinator.submit_evolution` future |
| generation | `generation.execute_spec` | `MemoryGenerationFamiliar.execute` |
| memory | `memory.create` / `memory.update` / `memory.delete` | `MemoryLibrary.mid_term` |
| memory | `memory.list` / `memory.get` / `memory.get_agent_profile` | `RetrievalFamiliar` |
| memory | `memory.retrieve` / `memory.retrieve_by_aliases` | `RetrievalFamiliar` |
| memory | `memory.record_feedback` / `memory.record_citation` / `memory.record_hit` | `LifecycleFamiliar` |
| memory_task | `memory_task.submit_generation` / `memory_task.submit_generation_many` | `MemoryGenerationTaskController` |
| memory_task | `memory_task.list` / `memory_task.get` / `memory_task.cancel` | `MemoryGenerationTaskController` |
| topic | `topic.prepare` | `PerceptionFamiliar.prepare_topic` |
| topic | `topic.list_active` | `RetrievalFamiliar.list_active_topics` |
| topic | `topic.get` | `RetrievalFamiliar.get_topic` |
| topic | `topic.manual_archive` / `topic.evict` / `topic.discard_if_empty` | `PerceptionFamiliar` |
| lifecycle | `lifecycle.refresh_memory_vitality` | `LifecycleFamiliar.refresh_memory_vitality` |
| lifecycle | `lifecycle.run_gardening_once` | `LifecycleFamiliar.run_gardening_once` |
| runtime | `runtime.storage_health` | `PatchouliRuntime.check_storage_health` |
| runtime | `runtime.models.warmup` / `runtime.models.ready` | `PatchouliRuntime.warmup_models` / `is_models_ready` |

兼容期可以保留旧常量别名，但新代码只使用分组后的 route 名称。

### 5.5 Application Service 逐个迁移

Application service 是 public API 编排层。其 public 方法均默认面向外部调用方，包括 system application service、server router、Alice 子系统或其他通过 `GlobalSystemBus` 访问 Patchouli 的调用方。Patchouli 内部业务不得调用这些 public 方法；如存在内部复用需求，应把可复用能力下沉为 local primitive。

应用服务迁移按“落点明确、风险从低到高”推进。优先处理薄代理和已有自然 handler 的 API，最后处理 `PatchouliService.prepare_agent_run()` / `finalize_agent_run()` 这类组合 workflow。

| 顺序 | 服务 | 迁移重点 | 目标落点 |
|---|---|---|---|
| 1 | `MemoryTaskManagementService` | task list/get/cancel | `MemoryGenerationTaskController` 暴露的 `memory_task.*` routes |
| 2 | `ModelReadinessService` | warmup / ready | `PatchouliRuntime` 的 runtime primitives |
| 3 | `TopicManagementService` | active topic 查询、手动归档、驱逐、空 topic 清理 | `RetrievalFamiliar` + `PerceptionFamiliar` 的 `topic.*` routes |
| 4 | `MemoryManagementService` | memory CRUD、feedback、citation、hit | `MemoryLibrary.mid_term` / `RetrievalFamiliar` / `LifecycleFamiliar` |
| 5 | `AgentProfileManagementService` | profile 创建 / 查询 | 复用 `memory.create` / `memory.list`；profile get-by-alias 走 `memory.get_agent_profile` |
| 6 | `PatchouliService` | prepare/finalize/cleanup 组合 workflow | local primitives 编排 |

每迁移一个 service，同步完成：

1. 增加或规范化对应 `PatchouliLocalRoutes` 常量。
2. 在 runtime 装配阶段注册 handler。
3. 将 application service 构造函数改为只接收 `PatchouliBus`。
4. 移除该 service 对 runtime、store、engine、controller、familiar 的直接引用。
5. 保持 public method 签名和返回模型稳定，避免同时重构调用方。

测试迁移与重构在第 5 章代码调整完成后集中处理；本阶段只保留必要的 smoke check / 编译检查，避免旧 `LibrarianCore` 测试与新路由边界交叉拖慢迁移。

迁移后每个 public 方法应保留 use-case 语义。例如 `prepare_agent_run()` 仍负责准备一次 Agent run 所需的完整上下文；它不会退化为：

```python
return await self._bus.request(PatchouliLocalRoutes.PREPARE_AGENT_RUN, ...)
```

而是显式组合内部能力：

```python
agent_profile = await self._bus.request(PatchouliLocalRoutes.GET_AGENT_PROFILE, ...)
topics = await self._bus.request(PatchouliLocalRoutes.TOPIC_LIST_ACTIVE, ...)
gaze = await self._bus.request(PatchouliLocalRoutes.GATEWAY_GAZE, ...)
...
```

构造函数统一形态：

```python
class MemoryManagementService:
    def __init__(self, bus: PatchouliBus) -> None:
        self._bus = bus
```

示例映射：

| Public service 方法 | 迁移后请求的 local route |
|---|---|
| `MemoryManagementService.create_memory` | `memory.create` |
| `MemoryManagementService.list_memories` | `memory.list` + 可选 `lifecycle.refresh_memory_vitality` |
| `MemoryManagementService.record_feedback` | `memory.record_feedback` |
| `MemoryTaskManagementService.list_memory_tasks` | `memory_task.list` |
| `AgentProfileManagementService.create_agent_profile` | `memory.create`（强制 `MemoryType.AGENT_PROFILE`） |
| `AgentProfileManagementService.list_agent_profiles` | `memory.list` + `index.memory_type = AGENT_PROFILE` filter |
| `TopicManagementService.list_active_topics` | `topic.list_active` |
| `TopicManagementService.archive_topic` | `topic.manual_archive` |
| `TopicManagementService.evict_topic` | `topic.evict` |
| `ModelReadinessService.warmup_models` | `runtime.models.warmup` |
| `ModelReadinessService.is_models_ready` | `runtime.models.ready` |

### 5.6 PatchouliService 瘦身

`PatchouliService` 的代理 API 在第 5 章移出，而不是在第 4 章提前处理。

| 当前 API 类型 | 迁移方向 |
|---|---|
| task 代理 API，例如 `get_memory_task` | `MemoryTaskManagementService` / `memory_task.*` |
| topic 代理 API | `TopicManagementService` / `topic.*` |
| memory 代理 API | `MemoryManagementService` / `memory.*` |
| profile / model 代理 API | 对应 application service；profile 复用 `memory.*`，model 使用 `runtime.models.*` |
| `prepare_agent_run()` / `finalize_agent_run()` | 暂保留为主动交互组合入口，最后改为 bus-only 编排 |

最终 `prepare_agent_run()` 通过 local routes 编排：

```
agent_profile.get
topic.list_active
gateway.gaze
topic.prepare
topic.list_active(include_empty=True)
topic.get
memory.retrieve
runtime.storage_health
```

`finalize_agent_run()` 通过 local routes 编排：

```
ingestion.submit_interaction
generation.submit_active
memory.record_hit / memory.record_citation
```

`cleanup_prepared_agent_run()` 如果仍需保留，则通过 `topic.discard_if_empty` 实现；若只作为 prepare/finalize 的临时配套 API，可随 public workflow 整理一并评估是否保留。

### 5.7 Bridge 收窄

Bridge 的构造参数从“完整对象图”收束为：

```python
class PatchouliBridge:
    def __init__(
        self,
        *,
        global_bus: GlobalSystemBus | None = None,
        public_api: PatchouliPublicApi,
    ) -> None:
        ...
```

其中 `PatchouliPublicApi` 可以是一个聚合对象，也可以是一组 bus-only application services。关键约束是：这些对象不能持有 runtime / engine / storage / controller；它们内部自行持有 `PatchouliBus` 并完成 public use-case 编排。

Bridge 可以为了 local event → global event 转发持有 `PatchouliBus` 或事件订阅端口，但不能用它把 public route 直接 forward 到 local route。

Bridge 注册 public routes 时只绑定 public API 方法：

```python
GlobalRoutes.PATCHOULI_MEMORY_LIST        -> public_api.memory.list_memories
GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN -> public_api.chat.prepare_agent_run
GlobalRoutes.PATCHOULI_MODELS_READY      -> public_api.readiness.is_models_ready
```

即使某个 public route 当前只是 local route 的薄代理，也优先由对应 public application service 方法完成转发。这样 public API 的参数校验、错误映射、返回模型和未来兼容策略都有稳定落点。Bridge 不保留通用 local forwarder，避免外部路由绕过 public API 层直接触达 local primitives。

完整 workflow（如 `prepare_agent_run`）不应由 Bridge 拼装，也不应在 runtime 中新增同名 handler，仍由 public application service 编排。

### 5.8 实施顺序

第 5 章按以下批次推进。核心原则是先建立自然 handler，再切 public API；不在 runtime 中补临时业务 handler，也不让 public API 请求同名 local workflow。

#### Step 1：冻结边界与命名

1. 盘点 `PatchouliRoutes` 与 `PatchouliLocalRoutes`，标记 public-only workflow：
   `PREPARE_AGENT_RUN`、`FINALIZE_AGENT_RUN`、`CLEANUP_PREPARED_AGENT_RUN`，以及待评估的 `ANALYZE_AND_RETRIEVE`。
2. 新代码禁止继续新增 `service.*` / public workflow mirror 形式的 local route。
3. 规范 local primitive 命名，优先补齐 `memory_task.*`、`topic.*`、`memory.*`、`agent_profile.*`、`runtime.*`。

#### Step 2：补齐低风险 local primitives

先注册已经有明确落点的 primitives：

| 能力组 | 优先 route | Handler |
|---|---|---|
| memory_task | `memory_task.list/get/cancel` | `MemoryGenerationTaskController` |
| runtime | `runtime.models.warmup/ready`、`runtime.storage_health` | `PatchouliRuntime` |
| topic | `topic.list_active/get/prepare/manual_archive/evict/discard_if_empty` | `RetrievalFamiliar` / `PerceptionFamiliar` |
| lifecycle | `memory.record_hit/citation/feedback`、`lifecycle.refresh_memory_vitality` | `LifecycleFamiliar` |

这一批只新增或规范化 local primitives，不迁移复杂 public workflow。

#### Step 3：迁移薄 public application services

按以下顺序把 application service 改为只持有 `PatchouliBus`：

1. `MemoryTaskManagementService`
2. `ModelReadinessService`
3. `TopicManagementService`

这些服务大多是薄代理，迁移后 Bridge 仍可以继续绑定其 public 方法，调用方无感。

#### Step 4：迁移 memory / profile public API

补齐 memory CRUD 与 profile 相关 primitives 后，再迁移：

1. `MemoryManagementService`
2. `AgentProfileManagementService`

如果某个 API 缺少自然 handler，不在 runtime 中写临时 handler；应先补 domain handler 或明确落到 `MemoryLibrary.mid_term` / profile domain service。

#### Step 5：瘦身 PatchouliService

1. 将 task/topic/memory/profile/model 代理 API 移出 `PatchouliService`，归还给对应 application service。
2. `PatchouliService` 剩余职责收束为主动交互组合 workflow：`prepare_agent_run()`、`finalize_agent_run()`、`cleanup_prepared_agent_run()`、必要的 `record_memory_citation()`。
3. 将这些方法改为只持有 `PatchouliBus` 与必要 public workflow 依赖，通过 local primitives 显式编排：
   `gateway.gaze`、`topic.*`、`memory.retrieve`、`ingestion.submit_interaction`、`generation.submit_active`、`memory.record_hit`、`runtime.storage_health`。

#### Step 6：收窄 Bridge

当所有 public API 对象均已 bus-only 后，再调整 `PatchouliBridge`：

1. 构造参数收窄为 `global_bus`、`local_bus`、bus-only public API 聚合。
2. public routes 只绑定 public application service 方法。
3. Bridge 不直接引用 runtime、Familiar、controller、engine、store，也不保留通用 local forwarder。

#### Step 7：删除 public workflow local 镜像

最后清理 local contract：

1. 从 `PatchouliLocalRoutes` 移除 `PREPARE_AGENT_RUN` / `FINALIZE_AGENT_RUN` / `CLEANUP_PREPARED_AGENT_RUN`。
2. 若 `ANALYZE_AND_RETRIEVE` 确认为 public-only，也从 local routes 移除；否则保留为明确的内部组合 primitive。
3. 保留必要旧 route 常量别名作为短期兼容，但新代码只使用分组命名，例如 `memory_task.get`、`topic.list_active`。

#### Step 8：测试迁移与回归

第 5 章代码边界稳定后，再集中迁移测试：

1. 删除或重写旧 `LibrarianCore` / direct runtime 依赖测试。
2. 为 public route → application service → local primitive 的映射补契约测试。
3. 为缺失 route、错误映射、public-only route 不出现在 local routes 中补回归测试。
4. 最后再做端到端链路测试，验证 prepare/finalize、passive ingest、memory task、topic 管理等 public API 仍保持外部行为稳定。

---

## 6. 事件流接入

> 本章待详细设计。以下为占位概要。

### 6.1 MemoryLibrary 状态事件

`MemoryLibrary` 的状态转移（flush / archive / revive）目前无可观测性。引入事件流后，每次状态转移发出对应领域事件，供下游订阅，例如 Lifecycle 维护、前端 topic 状态更新、审计日志。

待设计：事件类型定义、local event 与 global event 的边界、订阅关系、失败重试策略。

### 6.2 Generation / Task 事件边界

`MemoryGenerationFamiliar` 不直接发布事件，只返回 `MemoryGenerationResult`。`MemoryGenerationTaskController` 通过 `generation.execute_spec` route 获取 result 后，根据 result 与 task 终态发布任务状态、settlement、失败、取消、pending atom settled 等事件。

待设计：`PENDING_ATOM_SETTLED` 的新发布位置、active MTP 应答桥事件、task snapshot 事件字段。

### 6.3 Public API 错误与事件契约

第 5 章完成 bus-only public API 后，第 6 章补齐公开 API 的错误结构、超时策略与事件可观测性。

待设计：public route 的异常映射、`BusRouteUnavailableError` 处理、runtime event 字段、前端可消费的错误码。

---

## 7. 重构阶段计划

以下阶段按依赖顺序排列，每个阶段可独立完成和测试。

| 阶段 | 主要工作 | 前置依赖 |
|---|---|---|
| **Phase 1** | MemoryLibrary 骨架建立：三层 Store + StoragePort 接口定义，InMemory / Qdrant / File 适配器，感知层注入改造 | v0.5.2 完成 |
| **Phase 2** | Deduplicator 改为纯决策引擎；TaskController / LifecycleEngine 存储依赖切换到 MidTermStoragePort；ArtifactStore 上移至 MemoryLibrary | Phase 1 |
| **Phase 3** | RetrievalFamiliar 扩展：topic context 方法迁入，三层检索入口统一 | Phase 1 |
| **Phase 4** | LibrarianCore 编排职责解构：PerceptionFamiliar / LifecycleFamiliar / MemoryGenerationFamiliar 独立化，MemoryGenerationCoordinator 与 TaskController 作为控制面通过 local bus 串联，LibrarianCore 退场 | Phase 2 + 3 |
| **Phase 5** | Public API 与总线契约迁移：逐个将 `patchouli.application.*` 改为 bus-only，补齐 local primitives，移除 public workflow local 镜像，PatchouliService 瘦身并改为 bus-only，Bridge 收窄 | Phase 4 |
| **Phase 6** | 事件流接入：MemoryLibrary 状态转移事件、generation/task settlement 事件、public API 错误与事件契约 | Phase 5 |
| **Phase 7** | LongTermStore 完整实现（DBBasedStorageAdapter）与长期记忆链路补强 | Phase 6 |

**不在本次规划范围内**：记忆 split/merge 机制、MTP READ 历史编译、图数据库适配器。

**注意事项**：LifecycleEngine 同时维护中期记忆（vitality）和触发长期归档，不应强行分裂，整体作为独立链路保持不变。
