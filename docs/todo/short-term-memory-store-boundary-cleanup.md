---
title: ShortTermMemoryStore Boundary Cleanup
status: todo
owner: patchouli
scope: short-term-memory-store-crud-boundary-and-storage-key-encapsulation
code_paths:
  - src/hivememory/patchouli/memory_library/stores.py
  - src/hivememory/patchouli/memory_library/ports.py
  - src/hivememory/patchouli/memory_library/adapters/short_term.py
  - src/hivememory/patchouli/memory_library/buffer.py
  - src/hivememory/engines/perception/
related_docs:
  - docs/patchouli/memory-library.md
  - docs/patchouli/perception.md
  - docs/architecture/workspace.md
  - docs/architecture/decisions/0001-data-model-mutability-and-boundary-projection.md
  - docs/architecture/decisions/0002-unique-identities-and-minimal-concurrency.md
last_reviewed: 2026-09-02
---

# ShortTermMemoryStore 职责边界与存储键封装

## 事项定位

这是一项针对短期记忆存储层的技术债收敛事项。当前 `ShortTermMemoryStore` 同时承担底层 CRUD、Topic 状态机、Interaction 应用、Compact、Settle、Evict、LRU 和资产绑定等职责；`WorkspaceTopicKey` 也从物理寻址细节扩散到了 Store、感知模型和流程契约。它们使存储层与业务编排层的边界变得不清晰，增加了后续修改和并发分析的理解成本。

本 TODO 记录问题证据、目标边界和分阶段解决方案。若收回存储键需要同时迁移多个跨系统契约，或职责拆分已经形成独立的可验收迁移，应在实施前升级为 Plan；本文件不替代该 Plan。

## 当前结论

每个 `MemoryStore` 都应是自身记忆层级的存储封装，而不是业务编排器。Store 可以封装一个或多个下层 Port，决定如何向这些 Port 组织输入、选择查询路径并维护该层的存储事实；上层负责解释结果、组合业务流程、推进状态机和决定生命周期。

`MidTermMemoryStore` 是当前应当参照的抽象：它把 primary/secondary Port 的访问封装在中期层内，`upsert` 向后端分发，`get/search` 选择访问路径，但不负责把查询结果解释成业务决策，也不负责 archive、generation 或其他生命周期流程。`ShortTermMemoryStore` 应保持同一层级的纯粹性。

因此，短期 Store 的目标职责是：封装短期 Topic 记录的创建、读取、替换、删除、按 Workspace 列表/计数和健康检查，并在需要时提供快照或复制边界。Interaction、Topic 状态机、Page Folding、settle、evict、LRU、binding 判断、reservation/flush 协议和队列协调不属于 Store 的职责。

`WorkspaceTopicKey` 只是短期存储的复合寻址键。它不能作为领域模型、Store 公共 API 或感知层流程参数向外泄露。上层应使用 `IdentityScope`（或其既有访问上下文）与全局唯一的 `topic_id` 表达访问目标；由 Port/adapter 在最底层完成实际键构造。`topic_id` 的全局唯一前提仍然有效，复合键只用于 Workspace 归属校验和物理隔离，不构成允许不同 Workspace 复用 Topic ID 的局部命名空间。

并发方面，状态机约束和状态转换原子性是两个不同问题。当前代码仍需要临界区保护检查与更新之间的间隙；但这不意味着 Store 应永久承载一个覆盖业务状态、可变 buffer、LRU 和生命周期的“大锁”。先完成职责拆分，再根据真实并发契约决定锁的最终归属；在当前单用户、单进程、低并发基线下，不应为理论场景继续增加新的 Controller、single-flight 或每个消费者专用的协调方法。

## 问题分析

### 1. Store 同时承担存储和业务编排

`ShortTermMemoryStore` 当前约 500 行，除基本 CRUD 外还包含以下业务能力：

- `apply_interaction`：同时追加 block、建立首次 `TopicAssetBinding`、写入模型信息并要求 `PROCESSING` 状态；
- `apply_compaction`、`clear_blocks`、`update_summary`：实现 Page Folding/Compact 的裁剪和摘要更新；
- `reserve_processing`、`release_processing`、`reserve_flushing`、`commit_flushing`、`abort_flushing`：实现感知层状态机和 flush/admission 协议；
- `freeze_and_evict`、`freeze_for_manual_settle`、`pop_buffer_by_key`：实现 automatic/manual settle 的冻结、提交和驱逐；
- `get_lru_topic`、`needs_eviction`、`get_last_active_topic`、`set_last_active_topic`：实现 Topic 池策略和最近访问选择；
- `list_asset_bindings`、`get_buffer_info`：暴露 binding 事实和面向业务/测试的派生信息。

这些方法不只是“把一条记录写入后端”，而是在 Store 内决定何时允许状态转换、如何组合多字段写入、何时清理 buffer 以及如何为队列生成准备快照。结果是每个上游消费者都需要一套特事特办的入口，Store 逐渐成为状态机和生命周期控制器。

现有文档已经显露出这一版混合设计：`docs/patchouli/memory-library.md` 说明 Topic 内部状态和结算规则由 Perception 负责、MemoryLibrary 维护存储事实；同一节又把 block、摘要、模型名和状态更新列为 ShortTermMemoryStore 的命名写入口。代码与文档因此共同保留了两套职责判断，需要通过本事项重新收敛。

### 2. 公开 API 存在重复和语义冲突

当前同时存在按身份寻址和按复合键寻址的入口，例如 `get_topic_data` 与 `get_topic_data_by_key`、`pop_buffer` 与 `pop_buffer_by_key`。两者的忙状态约束也不一致：普通 `pop_buffer` 可直接移除，而 `pop_buffer_by_key` 只接受 `IDLE`。调用者如果只更换参数形式，就可能改变生命周期语义。

同样的重复还出现在 `add_block` 与 `apply_interaction`、`clear_blocks`/`update_summary` 与 `apply_compaction`、`update_metadata(state=...)` 与各类 reserve/release 方法之间。低层方法在正式代码中缺乏稳定消费者，却因为测试直接调用而继续扩大 Store 的公共表面。这种“既有原语又有复合原语”的形态，使写入顺序、失败回滚和状态所有权难以从 API 名称判断。

### 3. `WorkspaceTopicKey` 向上泄漏

目前可以观察到以下泄漏路径：

- Store 暴露 `get_topic_data_by_key`、`pop_buffer_by_key`，并让多个 `reserve_*`、`freeze_*`、`commit_*` 方法直接接收 key；
- `SemanticBuffer.topic_key` 和 `TopicData.topic_key` 将存储键重新暴露给调用方；
- `FlushEvent`、Perception 的 `TriggerManager`、`SemanticFlowPerceptionLayer` 以及相关接口以 `WorkspaceTopicKey` 作为领域流程参数；
- `ShortTermStoragePort` 以 `WorkspaceTopicKey` 作为所有 CRUD 方法的直接参数，迫使上层 Store 参与底层物理键构造；
- 中期层还存在 `get_by_key`/`delete_by_key`，`MemoryLibrary.archive(key)` 也接受 `WorkspaceMemoryKey`，说明这是值得后续统一审查的一类边界问题。

复合键本身并非错误：它可以作为 adapter 内部的安全索引。但一旦出现在领域快照、事件或 Store API 中，Workspace 身份就从访问边界变成了所有上游都必须理解的存储细节，后续替换后端、调整寻址方式或复用 Topic 领域模型都会被迫扩大改动范围。

### 4. 锁和状态机边界混在一起

当前 `ShortTermMemoryStore._lock` 同时保护：

- `IDLE -> PROCESSING/FLUSHING` 的检查与写入；
- `SemanticBuffer` 的 blocks、summary、binding、metadata 等可变字段；
- freeze 快照与 evict 的连续操作；
- `_last_active_topic_keys` 索引和 LRU 相关状态。

底层 `InMemoryShortTermStorage` 还使用另一把锁保护 map 和 Workspace 索引。由于 Port 返回的是可变 `SemanticBuffer`，adapter 的锁只保护取出对象的瞬间，Store 仍需保护对象内部字段，于是两层锁的边界和获取顺序变得复杂。

状态机可以表达“只有 IDLE 才能进入 PROCESSING”，但如果检查和写入之间没有原子临界区，两个异步任务仍可能同时通过检查。因此不能把当前锁简单判定为完全多余；它确实承担了现有复合操作的原子性。不过，锁的存在也不能成为继续把业务组合逻辑塞回 Store 的理由。职责收回后，锁应只保护实际拥有的存储容器或有证据的版本转换。

### 5. 可变 buffer 的存储边界不够清楚

Store 的读取方法目前会构造冻结的 `TopicData`，这是正确的方向；但 Port/adapter 之间仍传递真实可变 `SemanticBuffer`，并由 Store 直接修改其多个字段。这样一来，“Store 是存储事实封装”与“上层不得接触可变对象”的约束只在部分边界成立，快照何时生成、谁可以替换整条记录、谁负责派生字段，都不够明确。

## 当前实现证据

主要证据入口如下，后续迁移应以这些代码和测试重新核对，而不是仅按本 TODO 的目标模型改名：

| 证据 | 当前表现 | 暴露的问题 |
|:---|:---|:---|
| `src/hivememory/patchouli/memory_library/stores.py` | ShortTermMemoryStore 同时提供 CRUD、状态预约、复合 apply、freeze/evict、LRU、binding 和 info 方法 | 存储层与感知/生命周期业务混合 |
| `src/hivememory/patchouli/memory_library/ports.py` | ShortTermStoragePort 的 CRUD 参数直接是 `WorkspaceTopicKey` | 物理存储键进入 Store/Port 公共契约 |
| `src/hivememory/patchouli/memory_library/adapters/short_term.py` | adapter 维护 `WorkspaceTopicKey -> SemanticBuffer` map 和 Workspace 索引，并返回可变 buffer | 键构造和可变性边界未完全封装 |
| `src/hivememory/patchouli/memory_library/buffer.py` | SemanticBuffer 提供 `topic_key` 属性，并由 Store 直接改写字段 | 存储键和领域实体细节向外可见 |
| `src/hivememory/core/models/topic.py` | TopicData 提供 `topic_key` 属性并生成复合键 | 只读领域模型仍携带存储寻址概念 |
| `src/hivememory/engines/perception/trigger_manager.py`、`semantic_flow_perception_layer.py`、`interfaces.py` | 触发、folding、settle、swap-out 流程以 `WorkspaceTopicKey` 传递目标 | 感知契约依赖 Store 的寻址形式 |
| `docs/patchouli/memory-library.md` 与 `docs/patchouli/perception.md` | 一处强调 Perception 拥有状态和结算，一处保留 Store 命名写入口 | 当前设计理由和实现接口尚未完全收敛 |
| `tests/unit/patchouli/memory_library/`、`tests/integration/patchouli/`、`tests/unit/engines/perception/` | 大量测试直接构造 key、调用 `reserve_*`/`apply_*`/`freeze_*` | 测试既是现状证据，也是迁移时需要重写的契约依赖 |

## 目标职责边界

### ShortTermMemoryStore 应负责

- 通过一个短期存储 Port 保存和读取 Topic 记录；
- 创建、读取、替换/写回、删除单个 Topic；
- 按 Workspace 列表、计数和全量维护查询；
- 在必要时提供不可变快照或复制边界；
- 转发或聚合短期存储健康检查；
- 在存储层内部维护该 Port 所需的寻址、索引和 adapter 细节。

### ShortTermMemoryStore 不应负责

- Interaction 的领域校验、block 派生和 binding 判断；
- Compact、Page Folding 和摘要生成；
- settle、manual settle、automatic settle、evict 或 Topic 生命周期决定；
- LRU 策略和“何时需要驱逐”的业务判断；
- `PROCESSING`/`FLUSHING` 状态机及 reservation/flush/admission 协议；
- 记忆生成任务、队列提交、重试或跨 Store 协调；
- 将多个读取结果整合为上层业务结果。

这些行为应继续由 Perception、`SemanticFlowPerceptionLayer`、`TriggerManager`、`PerceptionFamiliar` 以及对应的应用/队列边界拥有；本事项不要求新增一个总 Controller 来替代它们。

## 目标 API 形状

目标是让 Store 的公共接口能够表达“本层存储事实”，而不是表达某个上游流程。具体命名可在实施 Plan 中结合现有调用方确定，建议形状如下：

```python
get(identity_scope, topic_id) -> TopicData | None
put(topic_data_or_snapshot) -> None
create(identity_scope, ...) -> TopicData
delete(identity_scope, topic_id) -> bool
list_by_workspace(identity_scope) -> list[TopicData]
list_all() -> list[TopicData]
count(identity_scope) -> int
check_health() -> StorageHealthComponent
```

这里的 `identity_scope + topic_id` 只是稳定的访问语义，不允许不同 Workspace 复用同一全局 Topic ID。若最终选择以 `WorkspaceIdentity` 作为存储层参数，也应保证 `WorkspaceTopicKey` 只在 adapter 内部构造和消费。

严格目标下，Store、领域模型、Perception 接口、事件 payload 和上层测试均不出现 `WorkspaceTopicKey`。`ShortTermStoragePort` 可以暂时保留内部 key-based 形态作为迁移过渡，但最终应将 key 构造下沉到 adapter，Port 对 Store 暴露 `workspace + topic_id` 或等价的稳定存储参数。任何兼容别名都必须是短期迁移 seam，不能形成第二套长期 API。

Store 返回的应是不可变 `TopicData`/`TopicSnapshot` 或明确的副本；上层若要追加 block、更新摘要或改变状态，应在自身领域层构建新的记录/快照，再通过 Store 的写回能力保存。是否使用整条记录替换、专用字段更新或版本检查，属于实施阶段的存储设计，不在本 TODO 中预先固定。

## 锁与并发处理原则

1. 不要在当前代码上直接删除 `ShortTermMemoryStore._lock`。在复合业务操作尚未移出前，它仍保护状态检查、可变 buffer 更新和快照/删除连续性；贸然删除会把已有状态机不变量变成真实竞态。
2. 完成职责拆分后，重新画出状态所有权：Store/adapter 只保护自身 map、索引和写回操作，Perception 负责其领域状态转换；不再由 Store 通过一把 RLock 同时覆盖业务生命周期和存储容器。
3. 如果短期存储继续采用当前单用户、单进程、低并发的单写者调用模型，应明确这一运行假设，不为假想并发增加 per-topic lock、Controller 或多层 receipt。
4. 如果未来出现可复现的同一 Topic 并发写入，优先在 Port/adapter 引入清晰的 revision/CAS 或单写者队列契约；保护应围绕实际被破坏的不变量，而不是为每一个上游消费者新增一个组合方法。
5. Port 不应把真实可变 `SemanticBuffer` 作为无边界的共享对象暴露出去。若后端需要可变内部实体，应在 adapter 内部拥有它，并以副本、快照或受控替换向 Store 交接。

## 分阶段解决方案

### 阶段 1：冻结职责和调用边界

- 在实施前确认 MemoryLibrary、Perception 和 Workspace 文档中的所有权描述，消除“状态由 Perception 负责”与“Store 提供业务写入口”的冲突；
- 把当前 Store 方法按 CRUD、领域状态、生命周期、索引/策略和测试 seam 分类，标记正式生产调用与仅测试调用；
- 明确 `WorkspaceTopicKey` 是内部寻址键，禁止新增新的 `by_key` Store 入口；
- 如果分类结果显示需要跨多个子系统同时迁移接口，则把本事项拆成独立 Plan，并在 Plan 中绑定验收范围。

### 阶段 2：收回存储键

- 移除或收敛 Store 层 `get_*_by_key`、`pop_*_by_key` 以及其他直接接收 key 的公开入口；
- 将 Perception 事件、接口和流程参数改为传递 `IdentityScope + topic_id` 或上层已经拥有的稳定目标对象；
- 逐步移除 `TopicData.topic_key`、`SemanticBuffer.topic_key` 等领域暴露属性；
- 让 Port/adapter 内部根据 Workspace 身份与全局 Topic ID 构造 `WorkspaceTopicKey`，并在一个位置完成归属校验；
- 同步迁移测试中的 key 构造，增加“上层契约不依赖复合键”的接口检查；
- 将 `MidTermMemoryStore.get_by_key`、`delete_by_key` 与 `MemoryLibrary.archive(key)` 作为同类边界问题另行评估，不在本事项中顺手扩大短期 Store 迁移范围。

### 阶段 3：让 ShortTermMemoryStore 回到 CRUD

- 保留 get/create/put/delete/list/count/health 等短期存储能力；
- 将 Interaction apply、Compact、settle、evict、LRU、binding 和状态预约迁回 Perception/领域层；
- 把 `add_block`、`clear_blocks`、`update_summary`、`update_title`、`update_metadata`、`update_model_used` 等零散写入口视为迁移 seam，逐步删除或降级为内部测试辅助，不再作为 Store 的长期公共能力；
- 对需要跨多个字段保持一致的操作，由上层先形成不可变快照/记录，再调用 Store 的单次写回，而不是为每一个消费者在 Store 内新增专用复合方法；
- 保留 Topic 资产绑定作为 Topic 事实的一部分，但由“本轮交互确认并形成 binding”的领域流程决定内容，Store 只保存和返回已形成的记录。

### 阶段 4：重新确定锁归属

- 在业务操作移出后，评估 Store 级 `_lock` 是否仍有必要；
- 如 adapter 仍是内存 map，应只保留保护 map/Workspace 索引的容器锁，并明确返回副本或快照；
- 对状态机真正需要的原子转换，在其领域所有者中采用单写者顺序或局部版本保护；
- 任何新增锁、CAS 或协调器都必须关联一个可复现竞态、明确不变量和测试，不以“理论上可能并发”作为唯一理由。

### 阶段 5：测试和文档收口

- Store 单元测试只验证 CRUD、Workspace 归属、快照边界和健康检查；
- Perception 测试验证 Interaction、Compact、settle、evict、LRU 和状态机行为；
- 增加 key 不泄漏到领域接口的静态/运行时契约测试；
- 增加可变对象隔离测试，确保修改调用方拿到的结果不会静默改变 Store 内事实；
- 更新 `docs/patchouli/memory-library.md`、`docs/patchouli/perception.md` 以及必要的 Workspace/Contracts 文档，使当前事实只保留一套职责描述；
- 完成后再根据最终代码决定是否形成 ADR，解释 Store 与领域层边界及锁归属的长期理由。

## 影响范围

直接影响短期存储实现、Port、内存 adapter、SemanticBuffer/TopicData 模型和 Perception 的触发/摄入接口。测试会受到较大影响，因为现有测试大量直接构造 `WorkspaceTopicKey` 并调用 Store 的状态机方法。若要一次性收回所有 key 参数，还会波及路由事件和跨系统 payload，因此需要先评估是否拆分为多个可验收切片。

对 Workspace 隔离的影响应保持收敛：Topic ID 仍按全局唯一值处理，Workspace 仍是访问校验和存储索引边界；本事项只是隐藏复合键的实现细节，不重新划分 cache、queue、scheduler 或 registry 的命名域，也不允许通过简化 API 取消归属校验。

## 明确非目标

- 不新增 Topic Controller 或万能协调器；
- 不改变 Topic 全局唯一 ID 规则和 Workspace 访问边界；
- 不引入跨 Store 事务、outbox、retry 或新的队列机制；
- 不在本 TODO 中实现多用户、多进程或分布式 Topic 并发模型；
- 不把 MemoryStore 变成业务结果整合器或生命周期编排器；
- 不删除 `WorkspaceTopicKey` 类型本身，只收回其可见范围；
- 不把 MidTerm 多 Port 的结果解释和业务合并逻辑下沉到 Store；
- 不因为当前锁复杂就直接删除锁，或以一次性大重构替代逐步验证。

## 完成条件

- ShortTermMemoryStore 的长期公共 API 只表达短期存储 CRUD、列表/计数、快照边界和健康检查；
- Interaction、Compact、settle、evict、LRU、binding 判断和 Topic 状态机均有明确的上层所有者，Store 不再通过专用方法编排这些流程；
- `WorkspaceTopicKey` 不再出现在 Store、领域模型、Perception 接口、事件 payload 或上层测试的稳定契约中，复合键仅在底层 adapter/内部索引使用；
- 不再存在语义冲突的 `by_key` Store 入口，兼容 seam 有明确迁移期限且不成为第二套真相；
- Store/adapter 的可变对象边界清晰，公开读取不会泄漏可修改的内部 buffer；
- 锁的每一处保留都有对应的状态所有权和不变量说明；没有证据支撑的 Store 级业务锁在拆分后被移除或下沉；
- CRUD、Workspace 隔离、Perception 生命周期和 key 封装测试均通过，且测试断言公开行为而不是内部 mock 调用次数；
- `docs/patchouli/memory-library.md`、`docs/patchouli/perception.md` 和相关入口文档与最终代码一致；
- 若迁移实际扩大为跨系统重构，已建立独立 Plan 并把本 TODO 标记为由该 Plan 承接。

## 相关事项

- [Patchouli MemoryLibrary](../patchouli/memory-library.md)：短期、中期、长期及 Artifact Store 的当前职责入口；
- [Patchouli Perception](../patchouli/perception.md)：Topic 状态、摄入、Compact、settle 和 evict 的领域责任入口；
- [Workspace 架构](../architecture/workspace.md)：Workspace 身份与资源隔离边界；
- [ADR-0001：按语义选择可变性，跨边界使用只读投影](../architecture/decisions/0001-data-model-mutability-and-boundary-projection.md)：快照与可变对象边界；
- [ADR-0002：全局唯一身份与按需并发保护](../architecture/decisions/0002-unique-identities-and-minimal-concurrency.md)：全局 ID 和按证据添加并发保护的原则。
