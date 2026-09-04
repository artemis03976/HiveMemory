---
title: Perception Topic Buffer Boundary Refactor
status: active
owner: patchouli
target: post-p7-perception-boundary-cleanup
scope: short-term-aggregate-storage-topic-lifecycle-trigger-orchestration-and-perception-engine-cleanup
code_paths:
  - src/hivememory/patchouli/memory_library/stores.py
  - src/hivememory/patchouli/memory_library/ports.py
  - src/hivememory/patchouli/memory_library/adapters/short_term.py
  - src/hivememory/patchouli/memory_library/short_term/
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/services/trigger_plan.py
  - src/hivememory/patchouli/services/topic_buffer.py
  - src/hivememory/patchouli/contracts/topic_management.py
  - src/hivememory/patchouli/control/memory_generation/models.py
  - src/hivememory/engines/perception/
  - src/hivememory/core/models/topic.py
  - src/hivememory/system/config/patchouli.py
  - src/hivememory/__init__.py
updates:
  - docs/patchouli/perception.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/README.md
  - docs/todo/short-term-memory-store-boundary-cleanup.md
  - docs/todo/topic-shutdown-per-topic-failure-isolation.md
related_docs:
  - docs/todo/short-term-memory-store-boundary-cleanup.md
  - docs/todo/topic-shutdown-per-topic-failure-isolation.md
  - docs/patchouli/perception.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/README.md
  - docs/architecture/decisions/0001-data-model-mutability-and-boundary-projection.md
  - docs/architecture/decisions/0002-unique-identities-and-minimal-concurrency.md
last_reviewed: 2026-09-04
---

# Perception Topic Buffer 边界重组计划

## 1. 背景：短期存储不是普通数据库表

上一轮边界清理把 `ShortTermMemoryStore` 收敛成 CRUD，并把状态转换、Topic Pool 和 settlement 迁到了 `TopicBufferService`。这解决了 Store 业务方法过多的问题，却产生了新的结构性问题：

- `TopicBufferService` 约 700 行，同时包含 CRUD 转发、Topic Pool 查询、Buffer 状态机、TriggerPlan、compact、settle、evict、Relay 调用和 reservation；
- Engine 仍通过 `TopicBufferService`、`InteractionApplyJournal` 和 Patchouli 错误反向依赖上层服务，所谓“无状态 Engine”并未成立；
- Runtime 与 Engine 工厂存在两套 Topic Buffer 组装入口；
- 如果继续把所有状态逻辑放在一个外部服务中，下一步只能在 Store、TriggerManager、TopicBufferService 和 Familiar 之间反复搬运同一份复杂度。

根因不是某个类拆得不够细，而是此前把短期存储错误地套用了中期、长期存储的抽象。中期和长期主要保存 `MemoryAtom`，存储介质本身不拥有记忆生命周期；短期存储则同时承载两个必须联动的一致性层：

```text
ShortTermMemoryStore
├── Topic Pool：有限的驻留集合
│   ├── 当前有哪些 Topic 存在
│   ├── 最大驻留容量
│   ├── 访问顺序、LRU、idle 和 shutdown 候选
│   └── Topic 是否处于可写、压缩或结算预约状态
│
└── Topic Buffer：每个 Topic 的工作内容
    ├── blocks
    ├── state_summary
    ├── TopicAssetBinding
    ├── title / summary / model_used
    └── token 与时间信息
```

settle 必须同时冻结 Buffer 并最终移除 Pool 中的 Topic；compact 只修改 Buffer，但仍需要通过 Pool 中的状态预约阻止交错写入；manual delete、LRU 和 shutdown 也必须遵守同一组驻留与状态不变量。因此，短期 Store 不能退化为“只会把记录交给 adapter 的无状态 CRUD facade”。

本次修订采用新的底层判断：

> `ShortTermMemoryStore` 是 `Topic Pool + Topic Buffer` 的一致性聚合存储。它封装短期存储自身的状态机制和原子提交，但不解释业务触发原因，也不调用任何外部服务。

这不是把原来的万能 Store 原样恢复，也不是保留一个换名后的 `TriggerManager`。需要重新划分的是“存储状态机制”和“跨系统业务编排”之间的边界。

### 1.1 与旧版“复杂 Store”的区别

旧版 Store 的复杂度主要来自“按业务原因拼装流程”：`apply_interaction`、`freeze_and_evict`、`reserve_flushing`、`commit_flushing` 等入口同时知道触发来源、领域状态、快照字段以及队列前后的处理顺序。调用方只要换一种触发原因，就可能进入另一套状态约束；Store 也因此被迫解释业务意图。

本计划收回的不是所有状态相关代码，而是收回短期存储为维持自身一致性所必需的最小协议。`append_interaction`、`begin/complete/abort_compact` 和 `begin/complete/abort_settlement` 的区别在于它们分别保护不同的存储不变量：Interaction 要一次性写入 Buffer 事实，compact 要保护锁外折叠后的条件写回，settlement 要冻结快照并在完成时联动删除 Pool 与 Buffer。它们不接收 `TriggerReason`，不调用外部服务，也不决定何时被触发。

因此，Store 的方法数量可能仍多于普通数据库 facade，但每个入口都能从“短期存储协议”本身解释，而不是为 automatic、manual、idle、LRU、shutdown 各复制一套业务方法。触发矩阵集中由 Patchouli 侧的无状态 `TopicTriggerPolicy` 解释；`PerceptionFamiliar` 只调用该策略并在具体业务场景应用结果。Pool/Buffer 的事实与原子提交集中由 `ShortTermMemoryStore` 维护；两者不会再通过另一个状态 Controller 互相转发。

## 2. 目标与非目标

### 2.1 目标

1. 将 `ShortTermMemoryStore` 明确定义为短期聚合存储边界，同时封装 Topic Pool 与 Topic Buffer 的驻留、一致性和生命周期机制。
2. Store 的公开方法围绕稳定的存储协议组织，而不是按照 `MANUAL_SETTLE`、`IDLE_TIMEOUT`、`LRU_EVICTION` 等触发来源复制入口。
3. 保留统一的 `TriggerReason -> TriggerPlan` 决策矩阵，但将其固定为 Patchouli 侧的无状态 `TopicTriggerPolicy`；`PerceptionFamiliar` 调用策略并应用计划，不在自身内嵌矩阵，也不由 Store 或 Engine 持有。
4. 让所有需要跨 `await` 的本地状态转换通过 Store 的 reservation 协议完成：`begin -> 锁外外部工作 -> complete/abort`。
5. 让 `PerceptionFamiliar` 成为现有 Patchouli 业务门面，负责路由、Journal、调用 `TopicTriggerPolicy`、Relay/Generation 协调、错误投影和调度入口；不新增另一个 Topic 状态控制组件。
6. 让 `engines/perception/` 只提供 block 构造、事件归并、token 估算和纯折叠算法，不导入 Patchouli service、Store、Journal、队列或 Patchouli 错误。
7. 让 `WorkspaceTopicKey` 继续封装在 adapter/内部索引，不进入 Store 以上的稳定契约；`topic_id` 仍按全局唯一值处理。
8. 统一 `TriggerReason`、`BufferState.SETTLING`、`MemoryPerceptionEngine`、`create_perception_engine` 和 `TopicMaterializeTask.from_topic_data()` 等既有命名目标，并将 `TriggerReason` 从 Engine 模型归属中收回到 Patchouli 生命周期策略/交接契约。
9. compact 仍强制 `retain_recent_blocks >= 1`；summary-only Topic 仍被视为有内容，且 settle 的 no-material 结果仍是正常生命周期完成。
10. 通过内部模块拆分控制 Store 的代码规模，但不把这些内部模块暴露成新的跨层状态所有者。

### 2.2 非目标

- 不保留或重建 `TopicBufferService`、`TriggerManager` 这类独立的 Topic 状态控制器；矩阵代码只允许由无状态的 `TopicTriggerPolicy.resolve()` 提供，且不得放入 `engines/perception/`；
- 不把 `ShortTermMemoryStore` 做成只允许七个 CRUD 方法的普通数据库 facade；短期存储的状态预约和 Pool/Buffer 联动属于其自身职责；
- 不把 Relay、Generation queue、InteractionApplyJournal、HTTP、SSE、Gateway 或 WorkspaceAssetStore 生命周期下沉到 Store；
- 不让 Store 解释 `TriggerReason`、决定用户业务意图或生成 API 响应；
- 不通过 `put` 或通用 callback 允许调用方绕过 Store 的状态不变量；
- 不新增跨系统 Topic Controller、Workspace Controller、single-flight、content revision、per-topic lock 或新的队列重试机制；
- 不改变 `topic_id` 的全局唯一规则、Workspace 访问校验或 `IdentityScope` 语义；
- 不在本计划中设计多用户、多进程、分布式 Topic 并发或持久化短期 Buffer；
- 不把 `TopicMaterializeTask` 的字段转换复制到多个调用方；
- 不把完整 Topic、SemanticBuffer 或 MemoryLibrary 设计复制到其他当前事实文档。

## 3. 当前实现证据与缺口

### 3.1 TopicBufferService 的边界错误

当前 [topic_buffer.py](../../src/hivememory/patchouli/services/topic_buffer.py) 同时包含：

- `create_topic`、`get_topic`、`touch_topic`、`count_topics`、`list_topics` 等 CRUD 转发；
- Topic Pool 的 LRU、idle、shutdown 候选查询；
- `PROCESSING`、`SETTLING` 状态转换和领域锁；
- Interaction block、asset binding、metadata 的写回；
- TriggerPlan、矩阵和 `handle_trigger`；
- Relay 调用、compact reservation 和写回；
- settle reservation、Generation admission 前后的删除/恢复；
- manual delete、空 Topic 清理和结果模型。

它既不是存储层，也不是单纯的业务门面，而是把短期存储自身机制和外部系统流程绑定在了一起。继续在此类中增删方法不能解决边界问题。

### 3.2 Engine 仍反向依赖 Patchouli

当前 `semantic_flow_perception_layer.py` 仍持有 Store/TopicBufferService、InteractionApplyJournal、配置与 Patchouli 错误，并实现 Topic 路由、retry、状态预约和 compact 触发。`engines/perception/__init__.py` 还会延迟导入并组装 Patchouli `TopicBufferService`。

因此当前类虽然被描述为“无状态”，实际上仍然是一个带有外部状态和业务流程的 Perception Service。重命名而不移除这些依赖，只会把边界问题隐藏在 `MemoryPerceptionEngine` 名称之下。

### 3.3 ShortTermMemoryStore 当前过度收缩

当前 [stores.py](../../src/hivememory/patchouli/memory_library/stores.py) 只有 CRUD 和快照复制。它无法表达：

- Topic Pool 与 Buffer 必须一起删除；
- settle 在 admission 完成前不能移除 Topic；
- compact 的锁外生成与锁内条件写回；
- `PROCESSING`/`SETTLING` 期间拒绝新的 Interaction；
- binding、blocks、token 和 metadata 的一致写入；
- Pool 查询与 Buffer 状态之间的安全联动。

因此上层只好通过另一个 Service 拼接多次 CRUD 调用，并在外部维护本应属于短期存储的状态不变量。

## 4. 目标架构

```text
Patchouli Runtime
  ├─ MemoryLibrary
  │   └─ ShortTermMemoryStore
  │       ├─ Topic Pool（有限驻留、索引、候选查询）
  │       ├─ Topic Buffer（内容快照和状态）
  │       ├─ reservation 与本地原子提交
  │       └─ ShortTermStoragePort -> adapter（内部 WorkspaceTopicKey）
  │
  ├─ PerceptionFamiliar
  │   ├─ 路由与当前 Topic 选择
  │   ├─ InteractionApplyJournal / retry
  │   ├─ 调用 TopicTriggerPolicy / 触发矩阵
  │   ├─ Relay 与 Generation admission
  │   ├─ 用户结果、maintenance、shutdown 报告
  │   └─ 调用 ShortTermMemoryStore 的存储协议
  │
  └─ MemoryPerceptionEngine（无状态）
      ├─ InteractionPayload -> LogicalBlock
      ├─ TurnEvent -> Action / Trace reduction
      ├─ token estimate
      └─ 纯 folding / 摘要算法能力
```

### 4.1 ShortTermMemoryStore：短期聚合存储

`ShortTermMemoryStore` 是 MemoryLibrary 的短期层的封装者。它可以由多个内部实现模块组成，但外部仍只有一个 Store 入口和一张 Topic 状态图。

#### Store 负责

- 以 `IdentityScope + topic_id` 访问 Topic；在最底层构造和校验 `WorkspaceTopicKey`；
- 创建、读取、替换普通 Topic 快照，以及按 Workspace/全量查询；
- 维护有限驻留集合、访问时间和活跃索引；
- 提供 LRU、idle、shutdown 候选的只读查询；
- 保证 Topic Pool 与 Topic Buffer 的增删同步；
- 执行 `append_interaction`，一次性保存 block、binding、token、时间和模型信息；
- 只保存由本轮交互确认的 `TopicAssetBinding` 事实；未被用户实际使用的 WorkspaceAsset 保持 orphan，不因存在或被上传而建立关联；
- 执行 `begin/complete/abort_compact`，保护 compact 的本地状态和条件写回；
- 执行 `begin/complete/abort_settlement`，冻结快照并在完成后联动移除 Topic；
- 执行独立的 IDLE Topic 删除，以及真正空 Topic 的清理；
- 暴露健康检查和不可变 `TopicData` 快照边界。

#### Store 不负责

- 解释 `TriggerReason` 或生成 `TriggerPlan`；
- 调用 Relay、LLM、Generation queue、Bus 或 EventBus；
- 读取或写入 InteractionApplyJournal；
- 决定 settle 是由 manual、idle、LRU 还是 shutdown 触发；
- 构造 HTTP/API 响应或向用户解释 admission 错误；
- 把多个 Topic 的结果整合成业务报告；
- 保存 WorkspaceAsset 内容或决定资源是否被用户使用。

Store 的“状态管理”是短期存储自身的存储协议，不是上层业务原因的解释器。它知道某个 Topic 是否处于 `SETTLING`，但不知道它为什么进入该状态，也不知道外部队列是什么。

### 4.2 Store API：按存储协议而非触发来源组织

长期公共 API 由普通访问、Pool 查询和少量跨 `await` 的存储协议组成。下面是目标形状；实际命名可在实施时根据现有调用点微调，但不得重新按触发来源复制方法。

```python
# 普通存储访问
get(scope, topic_id, *, touch=True) -> TopicData | None
create(scope, *, topic_title, topic_summary, topic_id=None) -> TopicData
put(topic_data) -> None
delete(scope, topic_id) -> bool          # 只允许可删除的 IDLE Topic
list_by_workspace(scope, *, include_empty=True) -> list[TopicData]
list_all() -> list[TopicData]
count(scope) -> int
check_health() -> StorageHealthComponent

# Topic Pool 查询
select_lru_candidate(scope, *, exclude_ids=()) -> str | None
list_idle_candidates(timeout_seconds) -> tuple[TopicCandidate, ...]
list_shutdown_candidates() -> tuple[TopicCandidate, ...]

# Buffer/Pool 联动的存储协议
append_interaction(scope, topic_id, block, *, asset_bindings=(), model_used=None) -> TopicData
begin_compact(scope, topic_id, *, retain_recent_blocks) -> CompactionReservation | None
complete_compact(reservation, *, state_summary) -> TopicData | None
abort_compact(reservation) -> bool
begin_settlement(scope, topic_id) -> SettlementReservation | None
complete_settlement(reservation) -> bool
abort_settlement(reservation) -> bool
delete_if_empty(scope, topic_id) -> bool
```

`begin_settlement()` 的参数只表达存储目标，不接收 `TriggerReason`；原因与用户/维护语义由 `PerceptionFamiliar` 保留，并通过 `TopicTriggerPolicy.resolve()` 决定后续流程。

这些协议的共同约束如下：

1. `put` 不能成为绕过生命周期状态机的后门。普通写回只能保存合法的非预约快照；`PROCESSING`/`SETTLING` 的进入、完成和恢复由对应协议控制。
2. `append_interaction` 在 Store 内完成状态检查、binding 去重、block 追加和相关字段更新；它不接收 `TriggerReason`，也不负责判断是否应该 compact。
3. `begin_compact` 在同步临界区内把 Topic 置为 `PROCESSING`，返回不可变折叠输入；Relay 在锁外执行；`complete_compact` 只在预约仍有效时写回摘要、保留 block 并恢复可写状态；失败调用 `abort_compact` 保留原内容。
4. `begin_settlement` 在同步临界区内把 Topic 置为 `SETTLING`，返回冻结的 `TopicData`；外部完成 admission 后，`complete_settlement` 才同时删除 Buffer 和 Pool；明确失败调用 `abort_settlement` 恢复 `IDLE`。
5. `delete`/`delete_if_empty` 只执行存储删除，不知道这次删除对应 manual delete、LRU 还是其他业务原因；非 `IDLE` Topic 不能被普通删除绕过。

`SettlementReservation`、`CompactionReservation` 是短生命周期的进程内存储协议对象。它们携带内部预约标识、Workspace、topic_id 和冻结快照/期望前缀，不持久化，不跨进程恢复，也不携带 Queue 或 Relay handle。

Store reservation 不携带 `TriggerReason`，因为同一个 settlement 存储协议不应因来源是 manual、idle、LRU 还是 shutdown 而改变。触发原因由 `PerceptionFamiliar` 与 `TopicTriggerPolicy` 在业务层保留，并在需要时附加到 `TopicMaterializeTask` 或 `SettlementOutcome`；Store 只处理预约标识、冻结快照和状态提交。

Topic 的唯一性仍以全局 `topic_id` 为前提：`create` 在所有 Workspace 范围内检查 ID 冲突，同一 ID 即使携带不同 `IdentityScope` 也必须被拒绝；读取和列表则按 scope 做归属校验与可见范围限制。`WorkspaceTopicKey` 只服务于这一校验和底层索引，不构成允许跨 Workspace 复用 ID 的命名空间。

### 4.3 TopicTriggerPolicy 与 PerceptionFamiliar

`TriggerPlan` 仍然是七种 `TriggerReason` 的唯一决策矩阵，但它不属于 Store，也不属于 `engines/perception/`。它固定放在 `patchouli/services/trigger_plan.py`，由无状态的 `TopicTriggerPolicy.resolve()` 作为唯一解析入口；不得再建立一个持有状态的 TriggerManager，也不得在 `services/perception.py` 或其他入口复制矩阵。

这里需要明确区分“策略翻译”和“计划执行”两件事：`TopicTriggerPolicy` 只回答“这个原因需要 settle、compact、evict 中哪些动作”，不读取 Topic、不调用 Store、不调用 Relay 或队列，也不选择目标话题。`PerceptionFamiliar` 是策略的使用者，负责把计划应用到具体 Topic，并统一协调 Store reservation、锁外外部调用和结果投影。若一个对象只做纯矩阵查表，就不应以 `TriggerManager` 命名；`Manager` 这个名称保留给有状态控制器会误导后续实现。

`TriggerReason` 的语义来自 Patchouli Topic 生命周期：它同时覆盖 token overflow、idle/LRU/shutdown 维护和三个 manual 入口，不是感知算法输出。因此它不得继续由 `engines/perception/models.py` 作为 Engine 模型的所有者；阶段 4 将枚举收口到 `patchouli/services/trigger_plan.py`，交接模型只引用这一份定义，Engine 只接收与自身算法有关的输入。

```python
@dataclass(frozen=True, slots=True)
class TriggerPlan:
    """触发原因解释后的三动作值对象，不持有 Store 或外部依赖。"""

    settle: bool = False
    compact: bool = False
    evict: bool = False

    def __post_init__(self) -> None:
        # settle 的结果必然是 Topic 生命周期结束，因此必须同时声明 evict。
        if self.settle and not self.evict:
            raise ValueError("settle=True requires evict=True")
        if not (self.settle or self.compact or self.evict):
            raise ValueError("TriggerPlan must contain at least one action")


TRIGGER_PLANS = {
    TriggerReason.TOKEN_OVERFLOW: TriggerPlan(compact=True),
    TriggerReason.IDLE_TIMEOUT: TriggerPlan(settle=True, evict=True),
    TriggerReason.LRU_EVICTION: TriggerPlan(settle=True, evict=True),
    TriggerReason.SHUTDOWN: TriggerPlan(settle=True, evict=True),
    TriggerReason.MANUAL_SETTLE: TriggerPlan(settle=True, evict=True),
    TriggerReason.MANUAL_COMPACT: TriggerPlan(compact=True),
    TriggerReason.MANUAL_DELETE: TriggerPlan(evict=True),
}


class TopicTriggerPolicy:
    """无状态的触发计划解析器，不持有 Topic 或任何外部依赖。"""

    @staticmethod
    def resolve(reason: TriggerReason) -> TriggerPlan:
        return TRIGGER_PLANS[reason]
```

`PerceptionFamiliar` 是已有的 Patchouli 业务门面，不是新增控制层。它可以保留一个统一的内部 `handle_trigger()` 用例，但该方法先调用 `TopicTriggerPolicy.resolve()`，再只做以下工作：

- 根据 `TriggerReason` 从 `TopicTriggerPolicy.resolve()` 取得 `TriggerPlan`；
- 选择目标 Topic 或消费 Store 返回的 Pool 候选；
- 调用 Store 的 `begin/complete/abort` 存储协议；
- 在 Store 预约完成后于锁外调用 Relay 或 Generation queue；
- 把 Store 结果投影成 `TopicSettleResult`、`TopicEvictionResult`、maintenance 结果或 shutdown 报告。

Familiar 的内部流程只按动作计划分派，不按七个原因复制流程分支；示意如下：

```python
plan = self._trigger_policy.resolve(reason)
if plan.settle:
    return await self._settle_candidate(target, reason)
if plan.compact:
    return await self._compact_topic(target)
if plan.evict:
    return await self._evict_topic(target)
```

`MANUAL_SETTLE` 与 `SHUTDOWN` 因而共享同一条 settlement 路径，差异只保留在目标选择和失败结果投影；`MANUAL_DELETE` 不会误进入 settlement。

Familiar 不直接 `get -> model_copy -> put` 拼装状态，也不直接修改 `TopicData.state`、blocks、summary 或 bindings。这样“统一解释”仍然存在，但状态所有权不再依赖一个额外的 TopicBuffer/TriggerManager 类。

### 4.4 MemoryPerceptionEngine

`MemoryPerceptionEngine` 直接提供无状态能力：

- `InteractionPayload + IdentityScope -> LogicalBlock`；
- `TurnEvent -> ActionReducer/TraceReducer` 派生值；
- token 估算和纯 folding 判断；
- 可注入的纯摘要/折叠算法。

它不得持有 Store、TopicData、InteractionApplyJournal、领域锁、Queue 或 Patchouli Service，也不得实现 `route_and_ingest`、`settle_topic`、`swap_out_topic` 等 Topic 用例。Interaction retry、路由和状态协议由 Familiar + Store 承接。

`TopicMaterializeTask` 是 Perception 到 Generation 的交接契约，字段转换和 `worth_saving` 过滤统一由 `TopicMaterializeTask.from_topic_data(...)` 完成，不在 Engine 或 Familiar 中逐字段拼装。

Settlement task 只携带已经形成的 asset ref/binding 快照，不复制或持有 WorkspaceAsset 的真实内容。后续创建 artifact 时再通过 ref 反查 `WorkspaceAssetStore`；asset 内容的读取和资源生命周期不属于 ShortTermMemoryStore。binding 的清理随 Buffer 的最终删除处理，若清理时机仍有缺口，另行作为技术债追踪，不在本次边界重组中增加跨 Store 控制器。

## 5. 状态、预约与锁设计

### 5.1 两层存储的一致性边界

Topic Pool 与 Topic Buffer 必须共享同一个 Store 聚合边界：

```text
Store lock 内：
  读取 Pool/Buffer -> 检查状态 -> 形成新快照或 reservation ->
  更新 Buffer 与 Pool 索引

Store lock 外：
  Relay、LLM、Generation admission、Bus 等外部调用，以及所有可能阻塞的 await

Store lock 内：
  校验 reservation 仍有效 -> 写回或同时删除两层存储
```

Store 可以使用一把聚合级 `RLock` 保护内存 adapter、Pool 索引和 reservation 状态。adapter 自身的锁只保护其底层 map 与快照复制；不得再由 `TopicBufferService` 或 Familiar 叠加第二把“领域锁”。

当前运行基线是单用户、单进程、低并发。锁只用于保证同步检查与写回、Pool/Buffer 联动和跨 `await` 后的预约确认；不因为理论上的多用户并发新增全局协调器、revision 或 per-topic lock。

### 5.2 Interaction

`append_interaction` 是一个同步的存储协议：

```text
MemoryPerceptionEngine.build_block()
  -> ShortTermMemoryStore.append_interaction()
      -> 检查 Topic 为 IDLE
      -> 一次性追加 block、binding、token、metadata
      -> 返回新的 TopicData
  -> Familiar 根据 token 阈值决定是否另行 begin_compact()
```

由于追加本身不跨 `await`，正常单进程调用不需要把 `reserve_processing/release_processing` 暴露为上游流程。若将来出现可复现的跨线程并发写入，应以具体不变量和测试为依据扩展 Store 内部协议，而不是为每个消费者增加一套预约方法。

### 5.3 Compact

```text
Store.begin_compact()
  -> IDLE -> PROCESSING
  -> 返回折叠前缀与 previous state_summary

Familiar / Relay（锁外）
  -> 生成新摘要

Store.complete_compact()
  -> 校验预约与折叠前缀仍有效
  -> 写入 state_summary、保留至少一个 block、重算 token
  -> PROCESSING -> IDLE
```

Relay 失败或预约失效时调用 `abort_compact` 或返回 no-op，不能覆盖其他写入，也不能留下永久 `PROCESSING`。

### 5.4 统一 settle

所有 `settle=True` 的原因都使用同一套存储协议和用户时序：

```text
Store.begin_settlement()
  -> IDLE -> SETTLING
  -> 冻结 TopicData

Familiar（锁外）
  -> TopicMaterializeTask.from_topic_data(...)
  -> Generation admission

明确接纳或没有可生成材料
  -> Store.complete_settlement()
  -> Buffer 与 Topic Pool 一起删除

明确拒绝或异常
  -> Store.abort_settlement()
  -> SETTLING -> IDLE，内容完整保留
```

`MANUAL_SETTLE`、`IDLE_TIMEOUT`、`LRU_EVICTION` 和 `SHUTDOWN` 的差异只影响目标选择、admission 失败的结果投影和维护报告；不再存在 automatic/manual 两套状态协议。`MANUAL_DELETE` 使用普通 IDLE 删除，不构造 settlement payload；`MANUAL_COMPACT` 只走 compact 协议。

settlement 不在 Generation admission 之前删除 Topic。Familiar 只以队列提供的 admission/receipt 作为完成依据；队列的持久化、重试、幂等和裸提交失败处理属于队列机制本身，不在本计划中再套一层补偿队列。若 admission 明确拒绝且尚未形成 receipt，调用 `abort_settlement` 保留内容；成功 receipt 之后的 Generation 后续失败属于 Generation 自身的终态，不重新打开 Topic，也不由 Store 补偿。

## 6. 代码组织与迁移边界

### 6.1 目标文件组织

```text
src/hivememory/patchouli/memory_library/
  stores.py                         # MemoryLibrary 对外 Store 入口
  ports.py                          # Port 契约，不暴露复合 key
  adapters/short_term.py            # adapter，唯一构造 WorkspaceTopicKey
  short_term/
    pool.py                         # 内部 Topic Pool 索引与候选查询
    reservations.py                 # Compact/Settlement reservation 值对象
    mutations.py                    # 纯 TopicData 变换和局部不变量

src/hivememory/patchouli/services/
  perception.py                     # PerceptionFamiliar 与跨系统业务编排
  trigger_plan.py                   # TriggerPlan 与唯一决策矩阵（无状态）

src/hivememory/engines/perception/
  memory_perception_engine.py       # 无状态 MemoryPerceptionEngine
  interfaces.py                     # Relay 等仍有用途的最小协议
  models.py                         # 纯感知模型；不得继续承载 Topic 生命周期交接模型
  relay_controller.py               # 无状态摘要能力
```

`topic_buffer.py` 不再作为独立的 Topic 状态所有者。迁移期间可以保留短期兼容文件，但生产代码不得继续从中组装 Store、Relay 和业务流程；完成后应删除该文件及其兼容导出。内部 `short_term/` 模块不对外公开，不形成第二个 MemoryLibrary 或 Topic Controller。

### 6.2 方法迁移映射

| 当前入口 | 目标归属 | 迁移说明 |
|:---|:---|:---|
| `TopicBufferService.create/get/touch/count/list` | `ShortTermMemoryStore` | 直接成为短期聚合存储 API，不再经过 Service 转发 |
| `TopicBufferService` 的 Pool/LRU/idle/shutdown 查询 | `ShortTermMemoryStore` 内部 Pool | Pool 查询属于短期驻留机制，返回只读候选 |
| `TopicBufferService.apply_interaction` | `ShortTermMemoryStore.append_interaction` | Store 内一次性保存 block、binding、token 和 metadata；不接收 TriggerReason |
| `reserve_processing/release_processing` | `append_interaction` 或 Store 内部同步临界区 | 正常 interaction 不再向上游暴露独立 processing gate |
| `TopicBufferService._execute_compact` | `begin/complete/abort_compact` + Familiar Relay 调用 | Store 只做预约和写回，Relay 在锁外执行 |
| `begin/complete/abort_settlement` | 同名 ShortTermMemoryStore 存储协议 | Store 冻结快照并联动 Pool/Buffer，Familiar 负责 admission |
| `delete_if_idle` / `discard_if_empty` | `delete` / `delete_if_empty` | Store 执行本地删除；业务原因由 Familiar 解释 |
| `TriggerReason`、`TriggerPlan` / `TRIGGER_PLANS` | `services/trigger_plan.py` | 从当前 Engine/混合入口收口为 TopicTriggerPolicy；只查表，不持有状态，不调用 Store |
| `TopicBufferService.handle_trigger` | `PerceptionFamiliar.handle_trigger` | 统一业务入口调用 Store 协议，不建立新的控制组件 |
| `TriggerManager._build_settle_payload*` | `TopicMaterializeTask.from_topic_data` | 删除重复字段转换 |
| `SemanticFlowPerceptionLayer` | `MemoryPerceptionEngine` | 只保留无状态 block/token/folding 能力 |
| `route_and_ingest`、Journal retry、Topic 路由 | `PerceptionFamiliar` + Store | Engine 不再承接 Topic 用例 |
| `create_perception_layer` | `create_perception_engine` | 始终构造真实 Engine，删除 no-op 分支 |
| `BasePerceptionLayer`、`NullPerceptionLayer` | 删除 | 不保留兼容别名或第二套接口 |

### 6.3 `put` 与快照写回边界

普通 `put` 只用于创建或替换合法的非预约 Topic 快照。任何改变 `PROCESSING`/`SETTLING`、完成预约、恢复预约或同时删除 Pool/Buffer 的操作，都必须经过 Store 内部协议。Familiar 不得通过 `model_copy()` 后直接 `put()` 模拟状态转换。

这项约束是为了避免“Store 看似是 CRUD，调用方实际上可以随意改变生命周期”的隐性第二套状态机。若测试需要构造特殊状态，应使用 Store 的测试辅助或公开预约协议，而不是绕过生产不变量。

## 7. 实施阶段

### 阶段 1：冻结新边界和依赖图

- 以本计划替代此前“Store 纯 CRUD、TopicBufferService 状态所有者”的目标模型；
- 盘点 Store、TopicBufferService、TriggerManager、PerceptionFamiliar、Engine 的生产调用点和测试调用点；
- 固定 `ShortTermMemoryStore = Topic Pool + Topic Buffer` 的所有权判断；
- 固定 Store 存储协议、reservation 生命周期和 `put` 约束；
- 固定 `TopicTriggerPolicy` 只属于 Patchouli 侧的无状态业务策略；Engine 不依赖 Patchouli 生命周期策略，也不拥有 `TriggerReason`。

### 阶段 2：重组 ShortTermMemoryStore 内部

- 把当前 adapter 的可变 Buffer、Pool 索引和全局 Topic ID 归属检查收敛到一个 Store 聚合边界；
- 建立 `short_term/pool.py`、`reservations.py`、`mutations.py` 等内部模块，控制 `stores.py` 体积；
- 实现 `append_interaction`、compact reservation、settlement reservation 和受约束删除；
- 移除 Store 以上稳定契约中的 `WorkspaceTopicKey` 与所有 `by_key` 入口；
- 补充 Pool/Buffer 联动、快照隔离和全局 Topic ID 唯一性测试。

### 阶段 3：迁移 Trigger 与 PerceptionFamiliar

- 将 `TriggerReason`、`TriggerPlan` 与唯一矩阵迁移到 `patchouli/services/trigger_plan.py`，形成无状态 `TopicTriggerPolicy`；
- 把 `TopicBufferService.handle_trigger` 的业务协调并入现有 `PerceptionFamiliar`，保留一个统一触发入口；Familiar 调用策略，不在类内复制矩阵；
- Familiar 只调用 Store 存储协议，不直接改写 TopicData；
- 统一 automatic/manual/idle/LRU/shutdown settle 时序，保持 admission-before-evict；
- 保留 manual delete、manual compact 与 settle 的独立动作语义。

### 阶段 4：收敛并重命名无状态 Engine

- 将 `semantic_flow_perception_layer.py` 重命名为 `memory_perception_engine.py`，实现 `MemoryPerceptionEngine`；
- 删除 Engine 对 Store、Journal、Patchouli errors、TopicBufferService 和队列的依赖；
- 将 `TriggerReason` 与 `TopicMaterializeTask` 从 `engines/perception/models.py` 的 Engine 模型归属中移出，避免 Engine 模型拥有 Patchouli Topic 生命周期语义；如交接模型需要序列化原因，改由 Patchouli 生命周期/Generation 契约引用唯一枚举；
- 把 block 构造、事件归并、token 估算和纯 folding 算法留在 Engine；
- 删除 `BasePerceptionLayer` 与 `NullPerceptionLayer`；
- 将工厂重命名为 `create_perception_engine`，删除 `enable=False` 静默 no-op 分支和旧导出。

### 阶段 5：删除旧混合入口并收口文档

- 删除 `TopicBufferService`、旧 `TriggerManager` 和相关兼容导出；
- 清理旧 Store 复合方法、`by_key` 入口、旧感知层导入和测试 seam；
- 更新 Runtime 组装，确保单进程只有一个 MemoryLibrary/ShortTermMemoryStore 状态图；
- 按最终实现更新 `docs/patchouli/perception.md`、`docs/patchouli/memory-library.md` 和 `docs/patchouli/README.md`；
- 修订或归档已经被本计划替代的旧 Todo 结论：ShortTermMemoryStore 纯 CRUD，以及 automatic SHUTDOWN 先 evict 后 admission；保留仍有效的 key 封装和逐 Topic 失败隔离要求；
- 验收完成后归档本 Plan，仍未解决的并发、持久化或跨系统问题另建 Todo/Plan。

## 8. 测试与验收

### 8.1 ShortTermMemoryStore 单元测试

- CRUD、创建/替换/删除、Workspace 访问校验和全局 Topic ID 唯一性；
- Topic Pool 容量、LRU、idle、shutdown 候选及 busy 候选跳过；
- Pool 与 Buffer 同步删除，空 Topic 和 summary-only Topic 的判空语义；
- `append_interaction` 的 block/binding/token/metadata 一次性写入与 binding 去重；
- `begin/complete/abort_compact` 的状态预约、锁外生成边界、异常恢复和至少保留一个 block；
- `begin/complete/abort_settlement` 的冻结快照、admission-before-evict、失败恢复和两层存储联动；
- `put` 无法绕过 `PROCESSING`/`SETTLING` 生命周期约束；
- adapter 可变对象与 Store 返回的不可变 `TopicData` 之间互不泄漏；
- `WorkspaceTopicKey` 只存在于 adapter/内部索引，Store 以上公开接口不暴露复合 key。

### 8.2 TriggerPlan 与 PerceptionFamiliar 测试

- 七种 `TriggerReason` 的唯一矩阵、三动作独立性和 `settle => evict` 不变量；
- `TopicTriggerPolicy` 只做纯矩阵解析，不读取 Topic、不调用 Store/Relay/队列；
- `PerceptionFamiliar.handle_trigger` 调用唯一策略并执行对应 Store 协议，矩阵不在多个入口重复；
- automatic、manual、idle、LRU 和 shutdown 共享 begin/admission/complete/abort；
- manual delete 只删除、不生成记忆；manual compact 只压缩、不 settle、不删除；
- admission 明确失败时 Topic 保留并恢复 `IDLE`，无材料时正常删除；
- shutdown 逐 Topic 失败隔离，异常不伪装成 skip；
- `TopicMaterializeTask.from_topic_data()` 的字段映射、binding 快照和 no-material 语义。

### 8.3 MemoryPerceptionEngine 与结构测试

- Engine 测试只验证 block、事件、token 和纯 folding 算法，不注入 Store、Journal 或 Patchouli Service；
- 静态搜索确认 `engines/perception/` 生产代码不导入 `hivememory.patchouli.*`；
- 静态搜索确认不存在 `TopicBufferService`、`TriggerManager`、`BasePerceptionLayer`、`NullPerceptionLayer` 和 `SemanticFlowPerceptionLayer` 的生产导出；
- 唯一工厂为 `create_perception_engine`，唯一引擎为 `MemoryPerceptionEngine`；
- `TriggerReason` 不再由 `engines/perception/models.py` 定义或导出，Engine 包不承载 manual/idle/LRU/shutdown 等生命周期语义；
- 只有一份 `TriggerPlan`/矩阵定义，不存在 `evict_timing`、`defer_evict` 等第二套时序字段；
- Runtime/Familiar/Engine 不直接修改 `TopicData.state`、blocks、summary 或 bindings；
- Runtime 只组装一个 `MemoryLibrary.short_term`，不存在另一套 Topic 存储实例。

### 8.4 集成与回归测试

使用真实 `PerceptionFamiliar + ShortTermMemoryStore + MemoryPerceptionEngine + deterministic Relay` 验证：

- Active/Passive interaction 共享同一 Store 写入协议；
- TOKEN_OVERFLOW 只 compact；settle 原因只走统一 settlement；
- LRU、idle、shutdown 和 manual settle 的 admission-before-evict；
- 两个 Workspace 不串扰，跨 Workspace 复用同一全局 Topic ID 被拒绝；
- receipt 成功后的 Generation 失败不会重新打开 Topic；
- `TopicAssetBinding` 只记录用户实际使用的 asset ref，orphan asset 不进入 Topic 事实；
- 现有 API、local route、shutdown 顺序和结果模型不发生未计划的变化。

## 9. 风险与待决事项

### 风险

- Store 重新承接短期状态机制后，若不拆分内部模块，`stores.py` 仍可能膨胀；必须把 Pool、reservation 和纯变换拆成私有模块，但不把它们变成新公共层；
- `put` 约束收紧会影响直接构造特殊状态的旧测试；测试应迁移到 Store 协议，而不是放宽生产 API；
- 将 `TopicBufferService` 业务协调并入 Familiar 可能使 Familiar 的方法数量增加；应按“路由/交互、触发策略、settle admission、maintenance”组织私有方法，但不得再引入第二个状态所有者；
- Relay/Generation 在锁外执行要求 reservation 能够阻止同一 Topic 的交错写入；Store 的状态预约和完成校验必须有针对性测试；
- Engine 重命名会影响 Runtime、顶层导出和大量旧测试；应集中迁移，不保留长期兼容别名；
- shutdown、idle 和 LRU 的逐 Topic 失败隔离必须保持，Store 只返回明确的状态结果，失败投影仍由 Familiar 完成。
- 现有 `short-term-memory-store-boundary-cleanup.md` 与 `topic-shutdown-per-topic-failure-isolation.md` 分别保留“Store 纯 CRUD”和“automatic settle 先 evict 后 admission”的旧结论；它们在实现收口时必须被修订或归档，不能继续作为与本计划竞争的实施依据。

### 待决事项

- `TopicMaterializeTask` 在 Patchouli/Generation 交接契约中的具体文件位置，需在阶段 4 结合现有导入面确定；`TriggerReason` 的唯一所有者已固定为 `patchouli/services/trigger_plan.py`，无论交接模型最终如何拆分，生命周期枚举与交接模型均不得由无状态 Engine 所有，字段转换方法和唯一真相不变；
- `short_term/` 内部模块的具体命名可以在阶段 2 根据现有 adapter 代码确定，但不得改变 Store 是唯一短期聚合入口的结论；
- 当前低并发基线不引入 revision/CAS；若测试发现同步 Store 协议仍无法保护真实竞态，另建针对性 ADR/Todo，不在本计划中预防性扩张。

## 10. 完成条件

- `ShortTermMemoryStore` 明确封装 Topic Pool 与 Topic Buffer，Pool/Buffer 的生命周期联动和本地状态预约不再由外部 Topic Service 承担；
- Store 公开 API 按存储协议组织，不按触发来源复制方法，不暴露 `WorkspaceTopicKey`，也不允许 `put` 绕过状态机；
- `TopicBufferService` 与 `TriggerManager` 不再作为生产状态所有者存在；统一 `TopicTriggerPolicy`/`TriggerPlan` 仍只有一份，由 Familiar 调用并应用 Store 协议；Engine 包中不存在恢复版 TriggerManager；
- `PerceptionFamiliar` 负责业务编排、Journal、Relay/Generation admission、结果投影和维护入口，不直接拼装 Topic 状态快照；
- `engines/perception/` 只包含无状态感知能力，唯一实现为 `MemoryPerceptionEngine`，不反向依赖 Patchouli；
- `BufferState.SETTLING`、`TriggerReason`、`create_perception_engine` 和 `TopicMaterializeTask.from_topic_data()` 的命名与调用链完成收口；
- automatic/manual/idle/LRU/shutdown settle 共用同一 admission-before-evict 时序，manual delete 和 manual compact 保持独立；
- compact 至少保留一个最近 block；summary-only Topic 不被误判为空；
- Store、PerceptionFamiliar、Engine、Runtime 和 adapter 的单元、集成、结构契约测试通过；
- `docs/patchouli/perception.md`、`docs/patchouli/memory-library.md` 和 `docs/patchouli/README.md` 已根据最终实现更新；
- 本 Plan 完成后进入 `docs/archive/plans/`，未完成的并发、持久化或跨系统事项另行追踪。

## 11. 修订记录

本节记录本 Plan 的模型修订，不把仍在实施中的目标冒充当前事实。

### 11.1 2026-09-03：从“TopicBufferService 状态所有者”改为“ShortTermMemoryStore 聚合存储”

本次修订废止此前以下目标：

- `TopicBufferService` 作为 Topic Buffer 与 Topic Pool 的唯一状态所有者；
- `ShortTermMemoryStore` 只保留 CRUD，禁止承接短期状态机制；
- 以外部 TopicBufferService 作为 Engine 与 Store 之间的长期边界。

新的冻结判断为：

1. 短期存储天然包含有限 Topic Pool 与 Topic Buffer 两层，二者需要在 settle、evict、compact 和容量管理中保持一致；
2. `ShortTermMemoryStore` 负责本地状态预约、Pool/Buffer 联动和存储原子提交，但不负责 TriggerReason、Relay、Queue、Journal 或 API；
3. `PerceptionFamiliar` 作为已有业务门面承接统一触发编排，不新增 `TriggerManager` 或其他状态控制组件；
4. `MemoryPerceptionEngine` 必须真正无状态，所有 Topic 路由、retry 和生命周期用例移出 Engine；
5. 内部模块拆分只用于控制代码体积，不形成新的公开状态所有者。

此前阶段 1 的调用点盘点和失败测试清单仍可作为迁移证据，但其中关于“状态执行迁入 TopicBufferService”和“领域锁归 TopicBufferService”的结论均由本修订替代。后续实施与验收以本版本的目标架构、Store 协议和完成条件为准。

### 11.2 2026-09-04：补充存储协议、资产引用和队列边界

- 明确本计划与旧版“复杂 Store”的差异：Store 只实现维持 Pool/Buffer 一致性所需的存储协议，不按触发原因复制业务入口；`TriggerPlan` 固定为 `services/trigger_plan.py` 的无状态策略。
- 明确 settlement task 仅携带已确认的 asset ref/binding 快照，真实附件由后续 artifact 创建流程反查 `WorkspaceAssetStore`；orphan asset 不建立 Topic 事实关联。
- 明确 admission-before-evict 的失败边界：Topic 不得在 admission 前删除；队列的持久化、重试、幂等和裸提交失败由队列机制负责，本计划不增加补偿队列；receipt 之后的 Generation 失败不重新打开 Topic。
- 更新复合键和低并发锁的说明，继续禁止 `by_key` 上浮、per-topic lock、revision/CAS 及额外 Controller。

### 11.3 2026-09-04：明确 TriggerPolicy 与 Engine 的分层

- 不恢复 `engines/perception/trigger_manager.py`。触发矩阵不是感知算法能力，而是 Patchouli Topic 生命周期策略；统一由 `patchouli/services/trigger_plan.py::TopicTriggerPolicy.resolve()` 提供。
- `PerceptionFamiliar` 作为策略使用者，按 `TriggerPlan` 的三类动作调用 Store 协议并协调锁外 Relay/Generation；它不在类内复制矩阵，也不把策略执行下沉到 Engine。
- `TriggerReason` 与 `TopicMaterializeTask` 不再由 Engine 模型长期所有；`TriggerReason` 固定收口到 `trigger_plan.py`，交接模型迁移到 Patchouli/Generation 契约，Engine 只保留纯感知模型和算法输入。
- Store 的 `begin_settlement()` 不接收 `TriggerReason`；触发来源只存在于业务上下文和必要的交接/结果模型中。
