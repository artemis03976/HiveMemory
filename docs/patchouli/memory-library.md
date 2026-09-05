---
title: Patchouli MemoryLibrary
status: current
owner: patchouli
scope: storage-ownership-and-tier-transitions
code_paths:
  - src/hivememory/patchouli/memory_library/
  - src/hivememory/infrastructure/storage/
related_contracts:
  - docs/architecture/boundaries.md
  - docs/contracts/subsystem-contracts.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-09-05
---

# MemoryLibrary 与存储层

MemoryLibrary 是 Patchouli 对记忆存储所有权的工程表达。它不是一个对所有后端都透明的万能 Repository，也不负责决定何时生成记忆；它把短期话题、中期可检索记忆、长期冷存储和 artifact 旁路放在同一个组合边界内，并确保跨层状态转移只有一个编排者。

早期实现让感知层直接创建短期 store，让 archiver 同时知道向量库与文件系统，让 generation 又直接决定落库。这样的组件各自都能完成局部任务，却没有任何一个位置能够回答“这条记忆此刻究竟在哪一层”。MemoryLibrary 的首要价值不是减少代码，而是建立存储事实的单一所有者。

## 1. 四种存储角色

```text
MemoryLibrary
  ├─ ShortTermMemoryStore -> ShortTermStoragePort
  │    └─ InMemoryShortTermStorage
  ├─ MidTermMemoryStore -> MidTermStoragePort
  │    └─ QdrantStorageAdapter -> QdrantMemoryStore
  ├─ LongTermMemoryStore -> LongTermStoragePort
  │    └─ FileBasedStorageAdapter
  └─ ArtifactStore -> ArtifactStoragePort
       └─ FilesystemArtifactStorageAdapter (optional)
```

### 1.1 短期：话题工作台

短期存储保存 Topic 记录，是纯 CRUD 的存储事实封装：adapter 直接存储 frozen `TopicData`，Store/Port 对外交接不可变快照；记录包含话题标题、展示摘要、折叠后的 `state_summary`、结构化 `LogicalBlock[]`、token 估算与最近模型名。记录不再携带执行状态或访问时间——占用权归 `TopicWorkingSet` 的 lease，访问顺序归 WorkingSet 的 LRU 索引（见[感知与短期话题](./perception.md)）。它服务于当前进程中的对话连续性，不是崩溃后可恢复的 durable session store。

`topic_id` 是领域上的全局唯一身份；调用方通过 `IdentityScope + topic_id` 访问，`WorkspaceTopicKey` 只由短期 adapter 在内部构造，用于 Workspace 归属校验和物理索引，不是允许不同 Workspace 复用同一 Topic ID 的局部命名空间。写入同一 `topic_id` 到另一 Workspace 会被 adapter 以全局唯一性检查拒绝。Topic 的编排规则由 PerceptionFamiliar 负责，MemoryLibrary 只维护存储事实与归属检查。

感知热路径使用同步接口，因此 `ShortTermStoragePort` 也是同步契约。未来若替换为远程后端，adapter 必须把 I/O 边界封装在 port 后方，不能让一组随机 `await` 穿透 Perception 的状态修改顺序。

`ShortTermMemoryStore` 的长期公共 API 只有 `get`、`put`、`create`、`delete`、`list_by_workspace`、`list_all`、`count` 和 `check_health`。Perception 在自己的领域边界内从快照形成新记录，再通过一次 `put` 写回；Gateway 和前端只能读取不可变 `TopicData` / `TopicSnapshot`。读取原样返回存储的 frozen 实例（零拷贝），frozen 模型保证调用方无法通过引用改写 Store 内事实。

活跃话题容量与 LRU 驱逐不属于 Store：`perception.engine.max_resident_topics`（默认 5）配置的是 `TopicWorkingSet` 的驻留上限。创建新话题且池已满时，PerceptionFamiliar 从 WorkingSet 选择最近最少访问的话题，先按 LRU 结算，再从 Store 与驻留集合移除；命中已有话题不会触发驱逐。

### 1.2 中期：当前可检索书库

中期存储以 `MemoryAtom` 为边界，当前主后端是 Qdrant。它提供 upsert、UUID/alias 读取、删除、scroll、count、访问统计更新以及 dense/sparse search，是 Retrieval、Generation 和 Lifecycle 共同依赖的当前记忆事实库。

`MidTermMemoryStore` 可以持有一个 primary 和可选 secondary port。写入会依次同步到各后端，读取只走 primary。当前 Runtime 只装配 Qdrant primary；secondary 仍是扩展点，不代表已经拥有多后端一致性协议。

### 1.3 长期：冷藏库

长期存储保存已经退出中期检索热集合的完整 `MemoryAtom`，当前使用文件系统 adapter，并可启用压缩。它支持 persist、load、remove、is_archived 和 archive record query，不参与普通向量召回。

长期层的意义不是“另一份备份”，而是让低活跃记忆可以退出高成本索引，同时保留被重新唤醒的可能。Lifecycle 决定何时归档，MemoryLibrary 执行数据搬运，LongTermStoragePort 只处理自己的介质；三者不能互相越权。

### 1.4 Artifact：不可变证据旁路

ArtifactStore 不属于三段冷热迁移链。它保存原始交互、外源文档、记忆创建记录和版本快照，以 `ArtifactRef` 挂到 `MemoryAtom` 上。Artifact 默认启用但在健康报告中是非必需组件；它失败时当前主记忆链仍可继续。

详细模型与一致性边界见[Artifacts 与来源追踪](./artifacts.md)。

## 2. 状态转移

### 2.1 短期到中期

短期 blocks 不由 MemoryLibrary 直接“升级”。Perception 形成 `TopicMaterializeTask`，Generation 从中提取、去重或更新记忆，MemoryGenerationFamiliar 挂载 artifacts 后再写入 MidTermMemoryStore。

```text
TopicData 快照
  -> TopicMaterializeTask
  -> GenerationRequest
  -> GenerationOutcome
  -> artifact side effects, including promoted external sources when present
  -> MidTermMemoryStore.upsert(MemoryAtom)
```

这种设计保留了一个重要区分：话题结算只是提供候选材料，不等于每组 blocks 必然产生正式记忆。

### 2.2 中期到长期：archive

```text
mid_term.get(memory_id)
  -> append ARCHIVED event
  -> long_term.persist(memory)
  -> mid_term.delete(memory_id)
```

Lifecycle 的 garbage collector 只筛选候选并调用 `MemoryLibrary.archive()`，不会直接操作两个 store。若中期不存在目标，archive 抛出 `ValueError`。

### 2.3 长期到中期：revive

```text
long_term.load(memory_id)
  -> append REVIVED event
  -> mid_term.upsert(memory)
  -> long_term.remove(memory_id)
```

Revive 当前由 Patchouli local route 暴露给内部用例。它是显式状态转移，不是普通 Retrieval 的自动 cache miss 行为；搜索不到某条记忆时，系统不会自动扫描冷藏库并复活。

## 3. 健康语义

`MemoryLibrary.check_storage_health()` 聚合四类组件：

- short term、mid term、long term 是 required；
- artifact store 为 optional，关闭时报告 `healthy=true, required=false, detail=disabled`；
- 聚合 `healthy` 只要求所有 required 组件健康。

Patchouli 的 `RUNTIME_STORAGE_HEALTH` 使用该聚合结果。Qdrant 不可用时，prepare 可以把 `storage_available=false` 放入 AgentRunContext，使 Alice 避免继续发出依赖长期存储的记忆操作。健康检查是降级信号，不是跨介质数据一致性证明。

## 4. 配置所有权

当前配置位于 `PatchouliConfig`：

- `storage`：Qdrant 地址、collection、向量维度、部署与启动参数；
- `perception.engine.max_resident_topics`：短期活跃话题上限；
- `lifecycle.archiver`：冷存储目录与压缩；
- `artifacts`：artifact 根目录、摘要内联长度和各 builder 开关。

Runtime 是唯一装配入口。Engine、Familiar 和应用服务不能自行重新读取配置并创建另一套 store，否则会破坏“同一进程只有一个 MemoryLibrary 状态图”的前提。

## 5. 当前限制与一致性边界

- 短期 store 只在内存中，异常退出不会自动恢复；
- archive/revive 是跨两个后端的顺序操作，没有事务、补偿日志或幂等 job；前一步成功而后一步失败时，可能暂时形成重复副本；
- MidTerm secondary 写入虽有接口，但当前没有原子提交和回滚语义；
- 长期记忆不会被普通 Retrieval 自动召回，revive 需要显式触发；
- 健康检查验证可访问性，不扫描缺失副本、悬空 ArtifactRef 或 archive/revive 中间态；
- Store 和 port 中仍有 Phase 命名与 future adapter 注释，它们是演化痕迹，不代表 Redis、图数据库或 SQL artifact index 已经实现。

后续若引入持久化短期状态、多后端复制或自动复活，必须先定义失败恢复与唯一真相，再修改本文件；不能只增加另一个 adapter 就宣称完成数据耐久性。
