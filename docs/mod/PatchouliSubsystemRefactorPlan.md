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
PatchouliRuntime（极薄的对外协调层）
├── MemoryLibrary            ← 三级存储 + 状态转移
├── RetrievalFamiliar        ← 全层检索服务（扩展）
└── MemoryIngestionPipeline  ← 生成写入管道
      └── MemoryGenerationTaskController（已存在）
```

**图书馆隐喻的对应**：
- `MemoryLibrary`：书库，管书的物理存放和流转
- `RetrievalFamiliar`：使魔，代理人去库里取书
- `MemoryIngestionPipeline`：写书流程，新书入库的完整流水线

### 2.2 MemoryLibrary

统一的三级存储层，负责记忆的物理存放和各层之间的状态转移。

```
MemoryLibrary
├── ShortTermStore    ← SemanticBufferManager（当前在 PerceptionLayer 内）
│     - 内存态 buffer
│     - topic 创建/归档/查询
│     - flush 触发（短期 → 中期 转移入口）
│
├── MidTermStore      ← QdrantMemoryStore（当前直接暴露）
│     - 向量数据库 CRUD
│     - 精确/模糊检索（由 RetrievalFamiliar 调用）
│
└── LongTermStore     ← ArchiveStore（待实现）
      - 冷存储（文件系统 / SQL）
      - 归档记录管理
      - 复活键（revival_keys）管理
```

**状态转移协议**（Library 内部管理，外部只感知事件）：

```
ShortTermStore.flush() → MidTermStore.upsert()   # 短期 → 中期
MidTermStore.archive() → LongTermStore.persist() # 中期 → 长期
LongTermStore.revive() → MidTermStore.upsert()   # 长期 → 中期（复活）
```

### 2.3 RetrievalFamiliar（扩展）

从"仅限中期向量检索"扩展为"三层统一检索入口"。

| 检索类型 | 对应层 | 当前状态 |
|---|---|---|
| Topic buffer 查询、话题快照 | 短期 | 分散在 LibrarianCore 方法中 |
| 精确取回（alias）、模糊语义检索 | 中期 | 已实现 |
| 冷存储复活检索（revival_keys） | 长期 | 待实现 |

将 LibrarianCore 中现有的 topic context 相关方法（`get_topic_context`、`get_active_topics_snapshots`）迁移至 RetrievalFamiliar，对外提供统一的"取书"接口。

### 2.4 MemoryIngestionPipeline

已基本成型（v0.5.0）。主要工作是将其从 LibrarianCore 内部组件升格为独立的流水线服务。

```
InteractionArtifactBuilder
    ↓ (topic raw interaction snapshot)
MemoryGenerationTaskController
    ↓ (compute → artifacts → persist 三步)
MemoryLibrary.MidTermStore
```

LibrarianCore 目前承担的"感知回调路由"职责（`_on_generate_memory`、`run_active_generation`）迁移至 Pipeline 的入口。

### 2.5 StoragePort 抽象层

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

### 2.6 感知层解耦

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

`_archive_topic` 操作的是调用方在 `resolve_topic` 中提前提取的 `blocks_snapshot`，自身不访问任何 buffer 字段。感知层解耦对该方法无影响。它通过 `_on_generate_memory` 回调触发 Pipeline，属于 Step 4（MemoryIngestionPipeline 独立化）的范畴，与感知层解耦是独立的重构步骤，互不阻塞。

### 2.7 中期记忆系统重构要点

#### Deduplicator 搜索优化

`MemoryDeduplicator` 目前自持 `QdrantMemoryStore` 仅为执行一次 `search_memories`，导致对存储层的不必要直接依赖。

**方案：将 dedup 改为纯决策引擎，搜索职责交还给 MemoryIngestionPipeline。**

```python
# 重构前：dedup 内部发起搜索
async def check_duplicate(self, draft) -> Tuple[Decision, Optional[MemoryAtom]]:
    results = await self.storage.search_memories(...)  # 自持存储，自己搜

# 重构后：dedup 只做决策，候选由 Pipeline 传入
def check_duplicate(
    self,
    draft: ExtractedMemoryDraft,
    candidates: list[SearchResult],  # Pipeline 检索后传入
) -> Tuple[DuplicateDecision, Optional[MemoryAtom]]:
    ...  # 纯逻辑，无 I/O
```

`MemoryDeduplicator` 不再持有任何存储引用，`merge_memory` / `_make_decision` 等逻辑不变。Pipeline 在摄入步骤中执行搜索（`MidTermMemoryStore.search()`），将结果传入 dedup——搜索本就在这一步发生，不增加额外网络往返。

#### 两条主要处理链路

**记忆摄入链路**（dedup → create/update → upsert）已在 TaskController 中基本成型。核心工作：dedup 改为纯决策后，链路内所有存储操作切换到 `MidTermStoragePort`，TaskController 不再直接持有 `QdrantMemoryStore`。

**生命力维护链路**（vitality 衰减 / 强化 / 垃圾回收）：LifecycleEngine 目前直接持有 `QdrantMemoryStore` 做 vitality 更新和批量删除，同样需要切换到 `MidTermStoragePort`。两条链路可独立推进。

### 2.8 长期记忆系统重构

`FileBasedArchiver` 承担的职责（归档、索引维护、resurrect）就是 `LongTermStoragePort` 的 `FileBasedStorageAdapter` 实现的完整映射。

迁移路径：
1. `LongTermMemoryStore` 建立，`FileBasedStorageAdapter` 包装现有 `FileBasedArchiver` 逻辑
2. 状态转移（archive / revive）上移至 `MemoryLibrary`（参见 §2.5 关键耦合修正）
3. `archiver.py` 作为独立文件退场，其逻辑以适配器形式存活于 `LongTermStoragePort` 实现中

长期存储是三层中改动最干净的部分，无深度耦合，可最后推进。

### 2.9 ArtifactStore（附属资产仓库）

`ArtifactStore` 与 `LongTermMemoryStore` 一样依赖外部持久化介质（文件系统 / SQL / 对象存储等），但它不属于三级记忆体系本身。Artifact 不是第四级记忆，而是记忆原子的附属资产与溯源产物，通常与 `MemoryAtom` 强关联，并以 append-only 形式保存 interaction snapshot、memory creation、memory version 等不可变记录。

当前 `ArtifactStore` 由 `ArtifactEngine` 私有创建并独占管理，这会导致其他链路无法将 artifact 仓库作为稳定的系统级存储能力使用。重构后，`ArtifactStore` 应上移至 `MemoryLibrary` 的装配范围，作为三级记忆之外的附属存储仓库：

```
MemoryLibrary
├── ShortTermStore
├── MidTermStore
├── LongTermStore
└── ArtifactRepository / ArtifactStore
```

但需要注意：`MemoryLibrary` 只负责 artifact 仓库的持有、注入和基础读写能力暴露，不负责决定何时构建 creation/version/interaction artifact。Artifact 的构建逻辑仍然属于 `ArtifactEngine` / builder，写入链路由 `MemoryIngestionPipeline` 编排。

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

MemoryIngestionPipeline
    → 编排 compute → artifact build/store → attach refs → memory persist
```

因此，ArtifactStore 应从 ArtifactEngine 私有资源上升为 `MemoryLibrary` 管理的附属持久化仓库；但它不参与短期 / 中期 / 长期记忆状态转移，也不污染三级记忆模型。

---

## 3. RetrievalFamiliar 能力增强

> 本章待详细设计。以下为占位概要。

### 3.1 目标

将 `RetrievalFamiliar` 从"仅限中期向量检索"扩展为**三层统一检索入口**，同时承接从 `LibrarianCore` 迁出的 topic context 查询职责。

### 3.2 迁移项

| 来源 | 迁移内容 | 目标层 |
|---|---|---|
| `LibrarianCore` | `get_topic_context`、`get_active_topics_snapshots` | 短期检索 |
| 现有 retrieval engine | 精确取回（alias）、语义检索 | 中期检索（已实现，保持） |
| 待实现 | revival_keys 冷存储检索 | 长期检索 |

### 3.3 注意事项

- `RetrievalFamiliar` 持有 `MemoryLibrary` 引用，通过各层 Store 接口检索，不绕过 Library 直接访问 Port
- 短期 topic 查询目前分散在 `SemanticFlowPerceptionLayer` 中，迁移时需与感知层解耦（§2.6）协调时序

---

## 4. LibrarianCore 编排职责解构

> 本章待详细设计。以下为占位概要。

### 4.1 目标

将 `LibrarianCore` 中残留的编排逻辑拆解为若干条各自独立的处理链路，每条链路对应一个具体的记忆流程。

### 4.2 主要处理链路

**记忆摄入链路**（MemoryIngestionPipeline）

```
感知层 archive 回调
    → InteractionArtifactBuilder
    → MemoryGenerationTaskController（dedup → create/update → upsert）
    → MidTermMemoryStore
```

当前 `LibrarianCore._on_generate_memory` / `run_active_generation` 是该链路的入口碎片，迁移后由 Pipeline 统一承接。

**生命力维护链路**（LifecycleMaintenancePipeline）

```
定时触发 / 事件触发
    → LifecycleEngine（vitality 衰减 / 强化 / GC）
    → MidTermMemoryStore / LongTermMemoryStore
```

LifecycleEngine 不拆分，整体作为独立链路的执行单元，由事件或调度器驱动，不再经由 LibrarianCore 中转。

### 4.3 LibrarianCore 退场策略

上述链路独立后，LibrarianCore 剩余职责应已清空。退场方式：
- 保留为薄代理层（方法委托到各链路入口），供向后兼容过渡期使用
- 过渡期结束后直接删除，由 `PatchouliRuntime` 承担剩余的薄路由

---

## 5. 新体系下的额外调整

> 本章待详细设计。以下为占位概要。

### 5.1 事件流接入 MemoryLibrary

`MemoryLibrary` 的状态转移（flush / archive / revive）目前无可观测性。引入事件流后，每次状态转移发出对应领域事件，供下游订阅（如 LifecycleEngine 响应 flush 事件更新 vitality、前端订阅 topic 状态变化等）。

待设计：事件类型定义、SystemBus 接入点、订阅关系。

### 5.2 patchouli/application 对外 API 重构

`patchouli/application/` 层目前的内部实现直接依赖 `LibrarianCore` 方法。LibrarianCore 退场后，各 Application Service 需要改为直接调用对应的 `MemoryLibrary`、`RetrievalFamiliar`、Pipeline 入口。

待设计：各 Service 的依赖重映射、接口兼容性保证。

---

## 6. 重构阶段计划

以下阶段按依赖顺序排列，每个阶段可独立完成和测试。

| 阶段 | 主要工作 | 前置依赖 |
|---|---|---|---|
| **Phase 1** | MemoryLibrary 骨架建立：三层 Store + StoragePort 接口定义，InMemory / Qdrant / File 适配器，感知层注入改造 | v0.5.2 完成 |
| **Phase 2** | Deduplicator 改为纯决策引擎；TaskController / LifecycleEngine 存储依赖切换到 MidTermStoragePort；ArtifactStore 上移至 MemoryLibrary | Phase 1 |
| **Phase 3** | RetrievalFamiliar 扩展：topic context 方法迁入，三层检索入口统一 | Phase 1 |
| **Phase 4** | LibrarianCore 编排职责解构：MemoryIngestionPipeline 独立化，LifecycleMaintenancePipeline 独立化 | Phase 2 + 3 | 
| **Phase 5** | LibrarianCore 退场（保留代理层或直接删除）；patchouli/application 依赖重映射 | Phase 4 |
| **Phase 6** | 事件流接入 MemoryLibrary 状态转移；LongTermStore 完整实现（DBBasedStorageAdapter） | Phase 4 |

**不在本次规划范围内**：记忆 split/merge 机制、MTP READ 历史编译、图数据库适配器。

**注意事项**：LifecycleEngine 同时维护中期记忆（vitality）和触发长期归档，不应强行分裂，整体作为独立链路保持不变。
