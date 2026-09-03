---
title: Perception Topic Buffer Boundary Refactor
status: active
owner: patchouli
target: post-p7-perception-boundary-cleanup
scope: perception-topic-buffer-state-ownership-trigger-execution-and-engine-cleanup
code_paths:
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/services/topic_buffer.py
  - src/hivememory/patchouli/memory_library/stores.py
  - src/hivememory/patchouli/memory_library/ports.py
  - src/hivememory/patchouli/memory_library/adapters/short_term.py
  - src/hivememory/engines/perception/
  - src/hivememory/core/models/topic.py
  - src/hivememory/system/config/patchouli.py
  - src/hivememory/__init__.py
updates:
  - docs/patchouli/perception.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/README.md
related_docs:
  - docs/todo/short-term-memory-store-boundary-cleanup.md
  - docs/patchouli/perception.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/README.md
  - docs/architecture/decisions/0001-data-model-mutability-and-boundary-projection.md
  - docs/architecture/decisions/0002-unique-identities-and-minimal-concurrency.md
last_reviewed: 2026-09-03
---

# Perception Topic Buffer 边界重组计划

## 1. 背景与问题

`ShortTermMemoryStore` 职责边界收敛后，原先的复杂度没有消失，而是集中转移到了 `engines/perception/trigger_manager.py`。当前 `TriggerManager` 同时承担触发矩阵解释、Topic 状态转换、Interaction 写入、资产绑定、Compact、settle、evict、快照构造和锁保护，已经成为一个有状态的 Topic 领域控制器。

这与项目既有分层不一致：`PerceptionFamiliar` 是面向上层 API、队列和结果的业务门面；`engines/` 下的 Perception 代码应当是无状态的底层能力实现。当前 `SemanticFlowPerceptionLayer` 同样持有 Store、InteractionApplyJournal、领域锁和生命周期方法，因而也不是纯 Engine。这个名称还是多个历史感知层实现逐步合并后遗留的命名，已经不能准确表达当前唯一的记忆感知引擎。另一方面，`PerceptionFamiliar` 直接读取 Store、选择 LRU、扫描 idle 和执行 shutdown，又把部分 Topic 池策略泄漏到了上层业务门面。

当前工厂还会在配置关闭感知时返回 `NullPerceptionLayer`，以空实现阻断整条能力链；但 Perception 已经是系统不可或缺的组成部分，这条可选分支只会制造第二套接口行为和静默丢弃数据的风险。

本计划不是删除统一触发解释，也不是把每个 `TriggerReason`（现有实现中的 `FlushReason`）分散回各个调用方。三类操作仍然独立：`settle`、`compact`、`evict`；`settle=True` 必须伴随 `evict=True`，但 `evict=True` 可以独立存在。七种触发原因（包括三个 manual 触发）继续通过一张唯一决策矩阵解释。需要重组的是矩阵解释与 Topic Buffer 状态执行之间的边界。

## 2. 目标与非目标

### 2.1 目标

1. 保留一个统一的 `TriggerReason -> TriggerPlan` 解释入口，矩阵继续表达 `settle/compact/evict` 三个独立动作；所有 `settle=True` 的原因共用同一条 settle 时序。
2. 在 Patchouli 内建立唯一的 Topic Buffer 领域服务，承接 `TopicData` 状态变换、Topic Pool 管理、快照冻结、binding 写入和矩阵计划执行。
3. 让 `engines/perception/` 只提供无状态的 block 构造、事件归并、token 估算和摘要/折叠算法，不持有 Store、Topic 状态、Journal、锁或队列。
4. 让 `PerceptionFamiliar` 只负责上层业务适配、当前 Topic 选择、generation admission、公开结果和运行时调度入口，不直接读写短期 Store 或判断 Buffer 状态。
5. 让 Store 继续保持 CRUD 与快照存储职责，`WorkspaceTopicKey` 仍只在短期 adapter 的内部寻址实现中出现。
6. 将 automatic 与 manual settle 统一为 `begin -> admission -> complete/abort`：材料冻结后才等待 Generation admission，接纳成功后结束 Topic 生命周期，明确失败则保留内容并恢复可重试状态；manual delete 仍为不写记忆的独立 evict，compact 仍至少保留一个 block。
7. 将状态锁收敛到真正拥有 Topic Buffer 状态的 Patchouli 领域服务；外部摘要生成和队列提交不得在领域锁内执行。
8. 清除历史感知层抽象：删除 `BasePerceptionLayer` 与 `NullPerceptionLayer`，将合并后的唯一实现统一命名为 `MemoryPerceptionEngine`，并使工厂始终装配真实感知能力。
9. 将触发原因统一命名为 `TriggerReason`，将结算预约状态统一命名为 `BufferState.SETTLING`，删除 `FlushReason` 与 `BufferState.FLUSHING` 的生产语义。
10. 将 `TopicData -> TopicMaterializeTask` 的字段转换、`worth_saving` 过滤和 no-material 判断收口到 `TopicMaterializeTask.from_topic_data()`，不在服务或调用方重复拼装。

### 2.2 非目标

- 不新增跨系统 Topic Controller、Workspace Controller、single-flight 或新的队列/重试机制；
- 不把 `TriggerManager` 拆成 `SettlementController`、`CompactionController`、`EvictionController` 等多个有状态控制组件；
- 不删除或分散决策矩阵，不让 Familiar、Engine 或各个入口自行解释 `TriggerReason`；
- 不改变 `topic_id` 的全局唯一规则、Workspace 归属校验或 `IdentityScope` 语义；
- 不恢复 Store 中的业务复合方法，也不把状态机重新下沉到 `ShortTermMemoryStore`；
- 不在本计划中设计多用户、多进程、分布式 Topic 并发或持久化短期 Buffer；
- 不引入 `content_revision`、跨 Store 事务或新的领域幂等记录，除非实施过程中出现可复现的不变量破坏并另行记录决策；
- 不把 Generation queue admission、Artifact promotion 或 shutdown 总体顺序改成另一套协议；
- 不把完整 Topic、SemanticBuffer 或 MemoryLibrary 设计复制进本计划以外的当前事实文档。

## 3. 当前实现证据与缺口

### 3.1 `TriggerManager` 的混合职责

当前 `src/hivememory/engines/perception/trigger_manager.py` 同时包含：

- `DECISION_MATRIX` 和 `resolve_topic()`：解释触发原因并组合动作；
- `settle_and_evict()`、`prepare_manual_settle()`、`commit_manual_settle()`、`abort_manual_settle()`：实现 automatic/manual settle 生命周期；
- `reserve_processing()`、`release_processing()`：实现 Topic 状态机预约；
- `apply_interaction()`：追加 block、判断并建立 `TopicAssetBinding`、更新模型信息；
- `_compact_path()`：调用 RelayController、裁剪 blocks、写回 summary；
- `delete_if_idle()`、`_set_state()`：直接执行 Store 读改写；
- `_build_settle_payload*()`：在服务外侧逐字段构造跨异步边界的 generation task，重复承担模型转换细节。

因此它不是无状态“触发解释器”，而是一个同时知道存储、领域实体、Relay 和状态锁的执行器。`resolve_topic()` 还使用 `MANUAL_SETTLE` 特殊异常分支绕开通用矩阵，进一步说明策略和执行协议被挤在同一个入口。

### 3.2 Engine 层的状态泄漏

`src/hivememory/engines/perception/semantic_flow_perception_layer.py` 当前持有 `ShortTermMemoryStore`、`InteractionApplyJournal`、`_domain_lock` 和 `TriggerManager`，并暴露 `settle_topic`、`prepare_settlement`、`swap_out_topic` 等有状态操作。它同时构造 `LogicalBlock`、处理 retry journal、管理 Topic 路由并执行 Page Folding，超出了无状态底层能力的范围。

`interfaces.py` 中的 `BasePerceptionLayer` 也把 `ingest_payload`、block 构造、Topic 管理、settle、evict 和 manual settle 协议放在同一个抽象接口中；`semantic_flow_perception_layer.py` 还通过 `NullPerceptionLayer` 提供一套静默 no-op 实现，工厂在 `enable=False` 时返回它。只要这两个历史抽象继续存在，调用方就会被迫把领域状态服务当作 Engine 使用，并且系统会保留“感知已关闭但请求仍被成功吞掉”的第二套行为。

`engines/perception/__init__.py` 与包顶层 `hivememory/__init__.py` 目前还公开导出旧类名和 `create_perception_layer`，Runtime、Familiar 及测试通过这些入口建立依赖；名称收敛必须覆盖这些导出，而不是只移动实现文件。

### 3.3 Familiar 与领域状态重叠

`src/hivememory/patchouli/services/perception.py` 当前直接使用 Store 完成：

- LRU 候选选择、容量判断和候选重试；
- idle 扫描和 shutdown 遍历；
- 当前 Topic 的 touch 与本地 active 索引；
- 空 Topic 清理。

Familiar 同时还负责 generation queue admission、manual settle 的提交窗口和公开结果映射。这样既无法让它保持单纯的 API 业务适配，也无法让一个组件完整拥有 Topic Pool 状态。

## 4. 目标架构

```text
PerceptionFamiliar
  └─ 上层 API / queue admission / 公开结果 / 调度入口
       ↓
Patchouli TopicBufferService
  ├─ TopicData 状态和 Topic Pool 的唯一领域所有者
  ├─ TriggerPlan 执行（不解释策略来源）
  ├─ Interaction / TopicAssetBinding 写入
  ├─ Compact / Settle / Evict / LRU / idle / shutdown
  └─ ShortTermMemoryStore CRUD
       ↓
ShortTermMemoryStore
  └─ ShortTermStoragePort / adapter（内部 WorkspaceTopicKey）

MemoryPerceptionEngine（无状态 Perception Engine）
  ├─ InteractionPayload -> LogicalBlock
  ├─ TurnEvent -> Action / Trace reduction
  ├─ token estimate
  └─ 无状态的摘要或折叠算法能力
```

### 4.1 `TopicBufferService`

在 `src/hivememory/patchouli/services/topic_buffer.py` 建立唯一的有状态 Topic Buffer 领域服务。它不是跨系统 Controller，也不创建新的总线层；它是 Patchouli 内部负责短期 Topic 领域状态的单一所有者。

它负责：

- 持有 `ShortTermMemoryStore`、Relay/无状态 Perception Engine 依赖和一把领域锁；
- 以 `IdentityScope + topic_id` 读取/写回 `TopicData`；
- 创建和确认 Topic、维护活跃池、选择 LRU、扫描 idle、提供 shutdown 快照列表；
- 对 `PROCESSING`、`SETTLING` 和 `IDLE` 执行状态转换；
- 原子地把 Interaction block、首次 binding 和相关 metadata 写回一个新快照；
- 根据统一 `TriggerPlan` 执行 compact、settle 和 evict；
- 冻结 `TopicMaterializeTask`，但不提交 generation queue；
- 返回结构化的领域结果，由 Familiar 决定是否提交或如何投影。

它不负责：HTTP、SSE、local/global bus、generation admission、API 响应模型、Gateway 命令解析、WorkspaceAssetStore 生命周期和跨 Store 协调。

### 4.2 统一触发策略与矩阵

统一矩阵保留在 `topic_buffer.py` 内，以同一个 `TriggerPlan` 表达策略结果。计划不新增 `topic_buffer_models.py`；`TriggerPlan`、矩阵和 Topic 执行结果类型与 `TopicBufferService` 放在同一文件，避免为少量内部模型制造碎片模块。跨边界共享的 `TriggerReason` 则继续作为纯协议枚举放在现有感知模型模块中，供矩阵和 `TopicMaterializeTask` 共同引用。

本计划按消息中的文件约束理解为：只建立一个集中承载 Topic Buffer 领域服务、策略和内部模型的 `topic_buffer.py`，不另行创建重复的 `topic_buffer_policy.py` 或 `topic_buffer_models.py`。如代码树中后续已经存在同名兼容文件，也不得再形成第二份矩阵或模型真相源。

矩阵的三列仍为独立动作，但 `settle=True` 不再区分 automatic 与 manual 的驱逐时机：它们都进入同一个 settle 协议。只有 `MANUAL_DELETE` 这样的 `evict=True`、`settle=False` 原因才绕过 settle。

| 原因 | settle | compact | evict | 实际执行 |
|:---|:---:|:---:|:---:|:---|
| `TOKEN_OVERFLOW` | 否 | 是 | 否 | `PROCESSING` 预约，锁外生成摘要，锁内写回 |
| `IDLE_TIMEOUT` | 是 | 否 | 是 | 统一 settle，admission 成功后结束 Topic |
| `LRU_EVICTION` | 是 | 否 | 是 | 统一 settle，admission 成功后释放 Topic 池容量 |
| `SHUTDOWN` | 是 | 否 | 是 | 统一 settle，逐 Topic 记录结果 |
| `MANUAL_SETTLE` | 是 | 否 | 是 | 统一 settle，成功后结束 Topic 并返回用户结果 |
| `MANUAL_COMPACT` | 否 | 是 | 否 | manual `PROCESSING` 预约，完成后回到 `IDLE` |
| `MANUAL_DELETE` | 否 | 否 | 是 | 只删除 Topic，不生成记忆 |

`TriggerPlan` 只表达矩阵中的三类动作，不携带 `evict_timing`、`defer_evict` 或其他把时序差异重新编码为第二套协议的字段。`settle=True` 必须满足 `evict=True`；其统一时序由 TopicBufferService 的 settle 执行协议保证，而不是由计划模型表达。

### `TriggerPlan` 数据模型示例

计划中的 `TriggerPlan` 是不可变的内部策略值对象。它只描述“这次触发需要哪些动作”，不持有 Store、锁、队列或 Topic 快照，也不负责执行动作：

下面示例中的 `TriggerReason` 是计划采用的新名称；现有实现的 `FlushReason` 只作为迁移来源，代码片段只展示计划值对象及其唯一映射，不重复定义该枚举。

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TriggerPlan:
    """TriggerReason 解释后的三动作计划；不承载执行时序或外部副作用。"""

    settle: bool = False
    compact: bool = False
    evict: bool = False

    def __post_init__(self) -> None:
        # settle 的结果必然是 Topic 生命周期结束，因此必须同时声明 evict。
        if self.settle and not self.evict:
            raise ValueError("settle=True requires evict=True")
        if not (self.settle or self.compact or self.evict):
            raise ValueError("TriggerPlan must contain at least one action")


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

实际实现可以用位置参数或带默认值的关键字参数，但应保持上述不变量和单一矩阵来源。`resolve_trigger_plan(reason)`（或等价的同文件纯函数）只从 `TRIGGER_PLANS` 返回计划；`TopicBufferService.handle_trigger()` 是唯一的计划执行入口，调用方不再根据 `TriggerReason` 复制分支。

### 统一 settle 用户时序

从用户可观察角度，settle 表示“结束 Topic，并把已冻结内容交给记忆生成链路”，而不是单纯删除一个 buffer。所有 `settle=True` 的触发原因都遵循以下协议：

```text
begin_settlement
  -> IDLE 进入 SETTLING
  -> 冻结 TopicData / TopicMaterializeTask
  -> 在领域锁外等待 Generation admission
       ├─ 明确接纳成功 -> complete_settlement -> 删除 SETTLING Topic
       ├─ 没有可生成材料 -> complete_settlement -> 删除 SETTLING Topic
       └─ 明确拒绝/异常 -> abort_settlement -> SETTLING 回到 IDLE，内容保留
```

`SETTLING` 是一个短生命周期的领域预约状态，不代表普通 flush、compact 或 eviction，也不作为持久化历史状态长期保留。目标实现应直接使用 `BufferState.SETTLING`，删除 `BufferState.FLUSHING`，避免同一状态存在两套命名和解释。

统一时序带来以下用户语义：

- settle 开始后 Topic 不再接受新的 Interaction；
- `TopicAssetBinding`、blocks 和 `state_summary` 在 begin 阶段一起冻结；
- queue 明确接纳后，Topic 才从活跃池移除；
- 没有可生成材料是正常成功，不是失败或 busy；
- admission 明确失败时，Topic 保留原内容并恢复可重试状态；
- queue 在已经返回成功 receipt 后的生成失败属于 Generation 自身的后续终态，不重新打开 Topic。

内部结果可以统一为以下形状，再由 Familiar 投影为不同的公共报告：

```python
from dataclasses import dataclass
from enum import Enum

class SettlementStatus(str, Enum):
    ACCEPTED = "accepted"
    NO_MATERIAL = "no_material"
    REJECTED = "rejected"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class SettlementOutcome:
    topic_id: str
    status: SettlementStatus
    removed: bool
    generation_task_id: str | None = None
    reason: TriggerReason | None = None
```

`BUSY` 仍通过 `TopicBusyError` 显式表达，不混入正常 settlement outcome。`ACCEPTED` 与 `NO_MATERIAL` 都表示 Topic 已经结束；`REJECTED` 表示 admission 未完成且 Topic 仍保留；`NOT_FOUND` 表示目标已被其他生命周期操作移除。该结果表示的是队列交接和 Topic 生命周期，不表示记忆已经生成或写入中期存储。

### 4.3 `TopicMaterializeTask` 转换边界

TopicBufferService 只负责冻结 `TopicData` 并提供本次操作的 `IdentityScope` 与 `TriggerReason`；它不应在服务方法中逐字段拼装生成任务。字段映射、`worth_saving` 过滤和“没有可生成材料”的判断收口在 `TopicMaterializeTask` 自身的类方法中。`identity_scope` 仍由调用方显式传入，因为 `TopicData` 只保存 Workspace 归属，不保存本次执行者身份：

```python
class TopicMaterializeTask(BaseModel):
    @classmethod
    def from_topic_data(
        cls,
        topic_data: TopicData,
        *,
        identity_scope: IdentityScope,
        reason: TriggerReason,
    ) -> "TopicMaterializeTask | None":
        """从冻结 TopicData 构造生成交接任务；无可保存 block 时返回 None。"""

        # 结算任务只携带值得保存的 block，字段转换不泄露到调用方。
        blocks = tuple(
            block for block in topic_data.blocks if block.worth_saving is not False
        )
        if not blocks:
            return None

        return cls(
            topic_id=topic_data.topic_id,
            identity_scope=identity_scope,
            topic_title=topic_data.topic_title,
            topic_summary=topic_data.topic_summary,
            blocks=blocks,
            state_summary=topic_data.state_summary,
            asset_bindings=topic_data.bindings,
            reason=reason,
        )
```

后续若 `TopicData` 或 `TopicMaterializeTask` 增加字段，只修改该转换方法及其模型测试；`begin_settlement()`、Familiar 和队列适配层不再复制字段映射细节。`TriggerManager._build_settle_payload*` 迁移后应删除，不能与 `from_topic_data()` 并存为第二套转换真相源。

### 4.4 无状态 `MemoryPerceptionEngine`

`engines/perception/` 下的实现应收敛为无状态能力。合并后的唯一感知实现命名为 `MemoryPerceptionEngine`，文件名为 `memory_perception_engine.py`；最终 Engine 不得持有 Store、Journal、TopicData 可变状态、领域锁或队列，也不再依赖历史的 `SemanticFlowPerceptionLayer` 名称。

Engine 的稳定能力包括：

- 从 `InteractionPayload` 和 `IdentityScope` 构造不可变 `LogicalBlock`；
- 从 `TurnEvent` 归并 `ActionReducer`/`TraceReducer` 所需的派生值；
- 计算 token 估算和是否达到 compact 阈值所需的纯判断；
- 为 TopicBufferService 提供纯摘要/折叠算法依赖。

Topic 状态、binding 去重和 `TopicData.model_copy()` 不属于 Engine 算法，应由 TopicBufferService 或同文件中的纯领域变换函数完成；`TopicMaterializeTask` 的字段转换统一由 `from_topic_data()` 负责。

不再保留 `BasePerceptionLayer` 继承层，也不新增另一个同义的抽象基类；`MemoryPerceptionEngine` 直接提供上述无状态能力。`NullPerceptionLayer` 同步删除，感知工厂始终装配真实引擎，不能通过 no-op 实例静默吞掉 Interaction。`SemanticFlowPerceptionConfig.enable` 同步删除，不再作为关闭感知的开关；含有旧字段或禁用值的配置必须在迁移时显式处理，不能静默回退。若 `interfaces.py` 继续保留，它只承载 `BaseRelayController` 等仍有实际用途的最小协议，不再导出 `BasePerceptionLayer`。

### 4.5 PerceptionFamiliar

Familiar 继续是上层业务门面，负责：

- 注入并调用唯一的 `MemoryPerceptionEngine`，不再依赖 `BasePerceptionLayer` 或 `NullPerceptionLayer`；
- 接收 Active/Passive 已接纳的 Interaction payload；
- 选择或传递用户明确指定的当前 Topic ID；
- 调用 TopicBufferService 完成交互与 Topic 用例；
- 将返回的 settlement task 提交到 Generation queue；
- 对所有 `settle=True` 计划执行统一的 `begin -> admission -> complete/abort` 外部时序；
- 将领域结果投影为 `TopicSettleResult`、`TopicEvictionResult` 和 shutdown 报告；
- 处理 local route、上层异常和调度 callback。

Familiar 不再直接访问 `ShortTermMemoryStore`，不读取 `BufferState`，不选择具体 LRU 候选，不遍历 TopicData，也不复制决策矩阵。当前 Topic 选择可以保留为 API 语义适配，但 Topic Pool 中“哪个 Topic 应当驱逐”由 TopicBufferService 决定。manual、idle、LRU 和 shutdown 只在目标选择、错误呈现和报告格式上不同，不再拥有两套 settle 提交与驱逐时序。

## 5. 状态与锁设计

### 5.1 领域操作的原子边界

TopicBufferService 的复合操作遵循：

```text
领域锁内：读取快照 -> 检查状态 -> 形成 reservation/新快照 -> Store 写回或删除
领域锁外：Relay 摘要生成、Generation queue admission、EventBus 调用
领域锁内：确认 reservation 仍然有效 -> 写回最终快照或完成/回滚本次变换
```

同一 Topic 的当前保护依赖 `PROCESSING`/`SETTLING` 状态预约和领域锁，不新增 per-topic lock。摘要生成不能在领域锁内执行；如果摘要生成失败，必须通过既有状态恢复路径回到 `IDLE`，不能留下永久 busy。

Store/adapter 的锁只保护单次 CRUD、内部 map、Workspace 索引和快照复制。TopicBufferService 的锁保护跨多个 CRUD 调用组成的一次领域状态转换。两者不能通过让 Store 暴露 callback/mutate API 重新合并。

### 5.2 统一 settle 协议

`IDLE_TIMEOUT`、`LRU_EVICTION`、`SHUTDOWN` 和 `MANUAL_SETTLE` 虽然触发来源不同，但都使用同一个 `begin -> admission -> complete/abort` 协议。`TopicBufferService` 提供以下领域原语：

1. `begin_settlement(identity_scope, topic_id, reason)`：在领域锁内确认 Topic 为 `IDLE`，将其置为 `SETTLING`，并通过 `TopicMaterializeTask.from_topic_data(...)` 冻结包含 blocks、`state_summary`、`TopicAssetBinding` refs 的交接任务；
2. Familiar 在锁外把可选 task 提交到 Generation queue。Topic 在 `SETTLING` 期间不能接受新的 Interaction；
3. queue 明确接纳成功，或 begin 阶段判断没有可生成材料时，调用 `complete_settlement()`，只删除仍处于本次 settling 状态的 Topic；
4. queue 明确拒绝或 admission 抛出异常时，调用 `abort_settlement()`，将 Topic 恢复为 `IDLE`，保留原始 blocks、summary 和 bindings；
5. queue 已返回成功 receipt 后的后续生成失败由 Generation 自身处理，不重新打开 Topic，也不由 TopicBufferService 补偿。

该协议不由 `TriggerPlan` 直接提交队列，也不允许 Familiar 绕过 TopicBufferService 直接写 Store。`settle=True` 的计划不再有“先驱逐”或“等待驱逐”的两种实现时序；Topic 只有在交接成功或无材料正常完成时才结束生命周期。

### 5.3 触发来源差异只影响协调与结果投影

统一 settle 协议下，触发来源的差异仅限于目标选择、调用入口和结果呈现：

- manual settle 由用户指定 Topic；admission 明确失败时恢复 Topic 并向用户返回可重试错误；成功或无材料时返回 `TopicSettleResult`；
- idle 扫描由 scheduler 提供候选；admission 明确失败时保留 Topic，维护轮次记录失败并等待下一次调度，不在 queue 外增加 retry；
- LRU 由 TopicBufferService 选择最久未访问的 IDLE 候选；admission 失败时不释放容量，调用方得到 backpressure/稍后重试语义；
- shutdown 逐 Topic 执行同一协议；未完成 admission 的 Topic 不计入已完成清理，失败被记录到 shutdown report；
- `MANUAL_DELETE` 仍是独立的 `evict=True`、`settle=False` 路径，不构造 generation task；`MANUAL_COMPACT` 仍是独立的 compact-only 路径。

Familiar 负责把这些统一的 `SettlementOutcome` 投影为用户 API 结果、maintenance 结果或 shutdown 报告，但不得为不同来源重新实现 begin/complete/abort。

## 6. 代码组织与迁移边界

### 6.1 目标文件组织

```text
src/hivememory/engines/perception/
  memory_perception_engine.py        # MemoryPerceptionEngine，无状态记忆感知能力
  interfaces.py                      # Relay 等仍需的最小协议，不含 PerceptionLayer 基类
  models.py                          # Engine/跨边界模型及 TopicMaterializeTask 转换
  relay_controller.py                # 无状态摘要能力

src/hivememory/patchouli/services/
  perception.py                      # PerceptionFamiliar，上层业务门面
  topic_buffer.py                    # TopicBufferService + TriggerPlan + 矩阵 + 内部结果

src/hivememory/patchouli/memory_library/
  stores.py                          # ShortTermMemoryStore CRUD
```

不得新增 `topic_buffer_models.py`，也不得再维护第二份 `DECISION_MATRIX`、`TriggerPlan` 或 Topic 执行结果模型。若策略需要独立测试，可测试 `topic_buffer.py` 导出的纯策略函数，而不以拆分文件作为测试边界。

### 6.2 方法迁移映射

| 当前入口 | 目标归属 | 迁移说明 |
|:---|:---|:---|
| `TriggerManager.DECISION_MATRIX` / `resolve_topic` | `TopicBufferService` 同文件内的纯策略与 `handle_trigger` | 保留统一解释，不再位于 Engine |
| `TriggerManager.reserve_processing/release_processing` | `TopicBufferService` | 状态预约只由 Topic 领域所有者执行 |
| `TriggerManager.apply_interaction` | `TopicBufferService.apply_interaction` | block/binding/metadata 作为单次领域写回 |
| `TriggerManager._compact_path` | `TopicBufferService.compact` | reservation、锁外摘要、锁内写回 |
| `TriggerManager.*settle*` / `delete_if_idle` | `TopicBufferService` | 收敛为 `begin_settlement`、`complete_settlement`、`abort_settlement` 与独立 `evict`，所有 settle 来源共用同一协议 |
| `TriggerManager._build_settle_payload*` | `TopicMaterializeTask.from_topic_data` | 将字段映射、worth_saving 过滤和 no-material 判断收口到数据模型；删除服务外侧的重复转换逻辑 |
| `semantic_flow_perception_layer.py::SemanticFlowPerceptionLayer` | `memory_perception_engine.py::MemoryPerceptionEngine` | 完成历史命名收敛；不保留类级别别名或旧模块导出 |
| `FlushReason` 与 `FlushEvent.reason` | `TriggerReason` 与同一事件模型的 `reason` 字段 | 将“刷新”这一过窄的历史命名改为统一触发原因；保留现有枚举字符串值，避免任务/事件载荷发生无关变化；事件载体是否改名不在本次范围内 |
| `BufferState.FLUSHING` | `BufferState.SETTLING` | 结算预约只使用 `SETTLING`；删除旧枚举成员及其 `flushing` 序列化值，统一使用 `settling` |
| `SemanticFlowPerceptionLayer._build_block` | `MemoryPerceptionEngine` | 重命名并收敛为无状态 block 构造，不再直接访问 Store 或 Journal |
| `SemanticFlowPerceptionLayer.route_and_ingest` | Familiar + TopicBufferService | 路由与 Topic 状态移出 Engine，旧方法不在新引擎中保留 |
| `create_perception_layer` 及顶层导出 | `create_perception_engine` | 始终创建 `MemoryPerceptionEngine`；删除 `enable=False` 的 no-op 分支以及旧类和基类导出 |
| `PerceptionFamiliar._maybe_evict_lru` | TopicBufferService 的池操作 | Familiar 只消费结果并提交 task |
| `PerceptionFamiliar.scan_idle_buffers_once` | TopicBufferService 扫描 + Familiar 提交 | 调度入口可留在 Familiar |
| `PerceptionFamiliar.flush_all_for_shutdown` | TopicBufferService 批量计划 + Familiar drain | 保持逐 Topic 报告和失败语义 |
| `ShortTermMemoryStore` | 保持 CRUD | 不回迁业务状态机或生命周期逻辑 |

### 6.3 兼容与删除策略

1. 先建立 TopicBufferService、`MemoryPerceptionEngine` 和 `create_perception_engine` 的新入口，更新 Runtime 装配、local route binding、公开导出和测试引用。
2. 将所有调用点从 `SemanticFlowPerceptionLayer`、`BasePerceptionLayer` 和 `NullPerceptionLayer` 迁移到新入口；最终不保留这些类名或 no-op 兼容别名。
3. 完成调用迁移后删除或明确标记旧 `engines/perception/trigger_manager.py`，避免它继续成为矩阵或状态逻辑的第二真相源。
4. 在新模块稳定后删除旧的 `semantic_flow_perception_layer.py`（由重命名后的 `memory_perception_engine.py` 取代），并从 `interfaces.py` 移除 `BasePerceptionLayer` 定义；`interfaces.py` 如仍被 `BaseRelayController` 使用则保留文件，但不得重新引入感知层基类。
5. 统一 settlement executor 后，删除 automatic `settle_and_evict` 与 manual `prepare/commit/abort` 的分叉实现；旧方法如需暂留，只能转发到同一 `begin/complete/abort` 协议。
6. 迁移期间不改变公开 local route 名称、Topic 管理 HTTP 结果模型和 generation task payload；公共协议只在最终事实文档收口时更新。

## 7. 实施阶段

### 阶段 1：冻结契约和依赖图

- 盘点所有 `TriggerManager`、`SemanticFlowPerceptionLayer`、`BasePerceptionLayer`、`NullPerceptionLayer`、`PerceptionFamiliar` 和 Store 调用点；
- 固定七种触发原因的矩阵、`settle => evict` 不变量、所有 settle 统一 `begin -> admission -> complete/abort` 顺序和 compact `retain_recent_blocks >= 1` 约束；
- 固定目标命名为 `TriggerReason` 与 `BufferState.SETTLING`，列出 `FlushReason`、`BufferState.FLUSHING` 及其导入/序列化引用的迁移清单；
- 确认 `MemoryPerceptionEngine` 的无状态调用契约、TopicBufferService 领域接口和 Familiar 业务接口的依赖方向；
- 禁止在迁移期间新增新的 `by_key`、Store 复合方法或额外状态控制器。

### 阶段 2：建立 TopicBufferService

- 在 `patchouli/services/topic_buffer.py` 内实现 `TriggerPlan`、矩阵、策略校验、统一 settlement 状态预约、快照变换和 `SettlementOutcome` 结果模型；
- 在现有感知数据模型中实现 `TopicMaterializeTask.from_topic_data()`，集中处理 TopicData 快照到交接任务的字段转换、worth_saving 过滤和 no-material 结果；
- 迁移 Interaction/binding、compact、automatic settle、manual settle、delete/evict、LRU、idle、shutdown 的状态执行；
- 使用领域锁保护复合状态转换，保证 Relay 和 queue 调用在锁外；
- 让所有 `settle=True` 触发原因都采用 admission-before-evict：明确接纳或无材料才完成删除，明确失败则恢复 Topic；
- 将所有 Store 访问收口到该服务，保留 `IdentityScope + topic_id` 入口。

### 阶段 3：收敛并重命名无状态 Engine

- 将 `engines/perception/semantic_flow_perception_layer.py` 重命名为 `memory_perception_engine.py`，并将实现类重命名为 `MemoryPerceptionEngine`；
- 从新引擎移除 Store、Journal、领域锁和 Topic 生命周期方法；
- 将 block 构造、事件归并、token 估算和纯 folding 算法留在 Engine；
- 删除 `BasePerceptionLayer` 与 `NullPerceptionLayer`，将工厂重命名为 `create_perception_engine`，同步清理 `enable=False` no-op 分支、顶层导出和相关配置语义；
- 由 Runtime/Familiar 直接注入 `MemoryPerceptionEngine`，不通过旧类名或兼容基类转发。

### 阶段 4：收敛 PerceptionFamiliar

- Familiar 改为注入 TopicBufferService 与无状态 Engine，不再持有 ShortTermMemoryStore；
- 将 `_maybe_evict_lru`、idle、shutdown 和空 Topic 清理改为调用领域服务；
- 保留 generation admission、统一 settle 外部三阶段时序、不同来源的结果投影和 scheduler callback；
- 删除 Familiar 内复制的状态判断、矩阵分支和 TopicData 遍历。

### 阶段 5：删除旧混合入口并收口文档

- 删除或冻结 `TriggerManager` 的状态执行实现，确保只剩一个矩阵和一个 Topic Buffer 状态所有者；
- 清理兼容 seam、旧导入和不再使用的 Store 复合 API；
- 按最终代码更新 `docs/patchouli/perception.md`、`docs/patchouli/memory-library.md` 与 `docs/patchouli/README.md`；
- 将本 Plan 在验收后移入 `docs/archive/plans/`，并把仍未完成的边界问题留在对应 Todo，而不是写入当前事实文档。

## 8. 测试与验收

### 8.1 单元测试

- 纯策略测试覆盖七种 `TriggerReason`、三列动作、`settle => evict` 和 `TriggerPlan` 不变量；
- TopicBufferService 状态测试覆盖 `IDLE/PROCESSING/SETTLING` 的合法与非法转换、binding 幂等追加、快照写回和 busy 隔离；
- Compact 测试验证 Relay 调用在锁外、异常后状态恢复、至少保留一个最近 block、无可折叠前缀时 no-op；
- settlement 测试验证空 Topic、summary-only Topic、worth_saving 过滤，以及所有 settle 来源共用 `begin -> admission -> complete/abort`；
- settlement 失败测试验证明确 admission 拒绝时 Topic 对所有触发来源均保留并恢复 `IDLE`，无材料时正常移除，成功 receipt 后的 Generation 失败不重新打开 Topic；
- `TopicMaterializeTask.from_topic_data()` 测试验证字段映射、Workspace/执行者作用域传递、binding 快照、worth_saving 过滤和 no-material 返回值；
- Topic Pool 测试验证 LRU、idle、shutdown、busy 候选跳过和全局唯一 Topic ID；
- 无状态 `MemoryPerceptionEngine` 测试只验证 block/事件/token/摘要计算，不注入 Store 或领域锁；
- 工厂与导出测试验证 `create_perception_engine` 始终返回 `MemoryPerceptionEngine`，不再存在 `NullPerceptionLayer`，且旧 `BasePerceptionLayer`/`SemanticFlowPerceptionLayer` 符号、模块路径和 `create_perception_layer` 导出均已清理；
- Store 测试只验证 CRUD、Workspace 归属、快照复制和健康检查，不再测试 TriggerPlan 或生命周期组合。

### 8.2 集成测试

使用真实 `PerceptionFamiliar + TopicBufferService + ShortTermMemoryStore + MemoryPerceptionEngine + deterministic Relay` 验证：

- Active/Passive interaction 都通过同一 TopicBufferService 写入 Topic；
- TOKEN_OVERFLOW 只 compact，不 settle、不 evict；
- automatic idle/LRU/shutdown 与 manual settle 按矩阵使用同一 settle 时序，并返回各自稳定的 task/skip/失败报告；
- manual settle 的用户错误、automatic 维护失败和 shutdown 未完成项只是结果投影差异，不产生第二套 Topic 生命周期；
- manual compact 与 manual delete 不误触发其他动作；
- 两个 Workspace 的同名/不同名 Topic 不串扰，Topic ID 全局唯一约束仍有效；
- Familiar 不直接接触 Store 可变对象或 WorkspaceTopicKey。

### 8.3 结构契约检查

- 静态搜索确认 `engines/perception/` 不导入 `ShortTermMemoryStore`、`InteractionApplyJournal` 或队列控制器；
- 静态搜索确认不存在 `BasePerceptionLayer`、`NullPerceptionLayer` 或 `SemanticFlowPerceptionLayer` 的生产导出，唯一感知引擎及装配入口分别为 `MemoryPerceptionEngine` 与 `create_perception_engine`；
- 静态搜索确认触发原因统一命名为 `TriggerReason`，生产代码不再导出 `FlushReason`；BufferState 的结算状态统一为 `SETTLING`，不再保留 `FLUSHING`；
- 静态搜索确认只存在一份决策矩阵和 `TriggerPlan` 定义，且不存在 `evict_timing`、`defer_evict` 等替代时序字段；
- 静态搜索确认 `WorkspaceTopicKey` 不出现在 Store 以上的稳定契约中；
- 检查 TopicBufferService 之外没有直接修改 `TopicData.state`、blocks、summary 或 bindings 的生产代码；
- 检查 `topic_buffer.py` 是唯一新增 Topic Buffer 领域文件，不产生 `topic_buffer_models.py` 或平行 policy/model 真相源。

## 9. 风险与待决事项

### 风险

- 当前测试大量直接实例化 TriggerManager 并断言内部方法调用，迁移会造成较大测试重写量；测试应改为断言公开领域结果和 Topic 终态；
- `BasePerceptionLayer`、`NullPerceptionLayer`、旧 `SemanticFlowPerceptionLayer` 名称及 `create_perception_layer` 工厂已被 Runtime、测试及顶层导出多处引用，直接清理会带来较大的迁移面；应集中更新调用点和工厂装配，并以缺失旧符号作为收口验收条件，不保留长期兼容别名；
- 把 Relay 调用移出锁后，需要确保 reservation 能够阻止同一 Topic 被并发写入；当前单用户低并发基线下以状态预约和领域锁完成，不预先增加 revision 字段；
- shutdown、idle 和 LRU 都涉及逐 Topic 失败处理，必须保持已有“异常不伪装成 skip”和逐 Topic 隔离语义；
- 若 TopicBufferService 继续膨胀为新的万能类，应按“单 Topic 状态变换 / Topic Pool 管理 / TriggerPlan 执行”三个内部区域组织方法，但不拆成一组新的跨层 Controller。

### 待决事项

- 无新增待决事项；`TopicMaterializeTask.from_topic_data()` 的转换职责和 `TriggerReason`/`SETTLING` 命名已在本计划中固定，实施时只需完成调用点迁移。

## 10. 完成条件

- `engines/perception/` 中的生产类不持有 Store、Topic Buffer、Journal、领域锁或队列状态，只提供无状态算法能力；
- 唯一感知实现为 `MemoryPerceptionEngine`，`BasePerceptionLayer`、`NullPerceptionLayer`、`SemanticFlowPerceptionLayer` 及 `create_perception_layer` 均已移除，感知工厂不会再静默禁用能力；
- `TopicBufferService` 是 Patchouli 内部唯一的 Topic Buffer 状态和 Topic Pool 所有者；
- `TriggerPlan` 与七种触发原因的决策矩阵只有一个定义，三类动作仍可独立解释，`settle => evict` 约束得到测试保证；
- `PerceptionFamiliar` 不直接读写 Store，不判断 BufferState，不选择 LRU 候选，不复制生命周期矩阵；
- `ShortTermMemoryStore` 保持 CRUD/快照/健康检查职责，没有回迁业务状态机；
- automatic settle、manual settle、manual compact、manual delete、idle、LRU 和 shutdown 的用户可观察行为与现行设计一致；
- Relay 和 Generation queue 均不在 Topic 领域锁内调用；busy、abort、skip 和异常结果具有稳定语义；
- `WorkspaceTopicKey` 仍被封装在短期 adapter/内部索引，不向 Familiar、Engine 或跨边界模型泄漏；
- 单元、集成和结构契约测试通过，并完成旧 TriggerManager、旧感知层类名及兼容入口清理；
- `docs/patchouli/perception.md`、`docs/patchouli/memory-library.md`、`docs/patchouli/README.md` 已根据最终实现更新，旧 Plan/Todo 链接不再把 TriggerManager 描述为当前状态所有者；
- 实施完成后本 Plan 进入 `docs/archive/plans/`，若仍有未落地的并发、持久化或跨系统问题，分别创建新的 Todo/Plan，不在本文件继续扩展。

## 11. 阶段实施记录

本节记录各阶段的实施证据与冻结结论，只作为本 Plan 的实施工作依据，不构成当前事实描述。

### 11.1 阶段 1：冻结契约和依赖图（2026-09-03 完成）

#### 11.1.1 基线状态

- 基线提交：`0ff5a17`（`refactor/short-memory-store-cleanup` 分支）。
- 全量测试基线：1997 passed，69 failed，49 errors（默认排除 `live_llm`/`e2e`/`slow`）。
- 全部失败与错误均为上一轮 ShortTermMemoryStore 边界清理遗留的过期测试，与本次重构无关但必须随本次迁移一并重写：

| 测试文件 | 失败/错误数 | 遗留原因示例 |
|:---|:---:|:---|
| `tests/unit/patchouli/memory_library/test_memory_library.py` | 3 failed / 34 error | 引用已删除的 `create_buffer`、`max_resident_topics` 构造参数等旧 Store API |
| `tests/integration/patchouli/test_perception_flush_chain.py` | 26 failed | 旧 `FlushEvent(topic_key=...)` / 旧 settle 签名 |
| `tests/unit/engines/perception/test_trigger_manager.py` | 23 failed | `settle_and_evict(topic_key, reason)` 旧签名（现为 `identity_scope, topic_id, reason`） |
| `tests/unit/patchouli/memory_library/test_binding_and_reservation.py` | 15 error | 引用已删除的 Store 预约/绑定方法 |
| `tests/unit/patchouli/services/test_perception.py` | 6 failed | mock 的 `AutomaticSettleResult`/旧调用链 |
| `tests/integration/patchouli/test_topic_access_chain.py` | 5 failed | 旧 key/签名 |
| `tests/integration/patchouli/test_asset_binding_lifecycle.py` | 3 failed | 旧 Store API |
| `tests/unit/patchouli/control/test_interaction_submission.py` | 2 failed | 旧调用链 |
| `tests/integration/patchouli/test_workspace_interaction_retry.py` | 1 failed | 旧签名 |

#### 11.1.2 旧符号定义处与迁移目标

| 符号 | 定义位置 | 目标 |
|:---|:---|:---|
| `TriggerManager`、`DECISION_MATRIX`、`resolve_topic`、`settle_and_evict`、`prepare/commit/abort_manual_settle`、`reserve_processing`、`release_processing`、`apply_interaction`、`_compact_path`、`delete_if_idle`、`_set_state`、`_build_settle_payload*` | `src/hivememory/engines/perception/trigger_manager.py` | 状态执行迁入 `TopicBufferService`（阶段 2）；`_build_settle_payload*` 由 `TopicMaterializeTask.from_topic_data()` 取代；最终删除该文件（阶段 5） |
| `SemanticFlowPerceptionLayer` | `src/hivememory/engines/perception/semantic_flow_perception_layer.py` | 重命名为 `memory_perception_engine.py::MemoryPerceptionEngine`（阶段 3） |
| `NullPerceptionLayer` | 同上 | 删除（阶段 3） |
| `BasePerceptionLayer` | `src/hivememory/engines/perception/interfaces.py` | 删除；`BaseRelayController` 保留在原文件（阶段 3） |
| `FlushReason`、`FlushEvent`、`AutomaticSettleResult` | `src/hivememory/engines/perception/models.py` | `FlushReason` 重命名为 `TriggerReason`（枚举字符串值不变）；`AutomaticSettleResult` 由 `SettlementOutcome` 取代；`FlushEvent` 载体是否改名不在范围内 |
| `BufferState.FLUSHING`（序列化值 `flushing`） | `src/hivememory/core/models/topic.py:21` | 重命名为 `SETTLING`（序列化值 `settling`），删除 `FLUSHING` |
| `create_perception_layer` | `src/hivememory/engines/perception/__init__.py` | 重命名为 `create_perception_engine`，删除 `enable=False` no-op 分支 |
| `SemanticFlowPerceptionConfig.enable` | `src/hivememory/system/config/patchouli.py:51` | 删除该开关（阶段 3），配置加载需显式处理旧字段 |
| `TopicMaterializeTask` | `src/hivememory/engines/perception/models.py` | 保留为跨边界模型，新增 `from_topic_data()` 类方法 |

#### 11.1.3 生产代码调用点盘点

| 调用点 | 引用内容 | 迁移阶段 |
|:---|:---|:---:|
| `src/hivememory/__init__.py:159-177、355-372` | 顶层导出 `SemanticFlowPerceptionLayer`、`BasePerceptionLayer`、`NullPerceptionLayer`、`FlushEvent`、`FlushReason`、`TriggerManager`、`DECISION_MATRIX`、`create_perception_layer` | 阶段 3 |
| `src/hivememory/engines/perception/__init__.py` | 包导出与工厂；`enable=False` 时返回 `NullPerceptionLayer`（:97-98）；`__all__` 中残留未导入的 `InteractionArtifactBuilder` 项 | 阶段 3 |
| `src/hivememory/engines/perception/interfaces.py` | `BasePerceptionLayer` 八个抽象方法（settle/prepare/commit/abort/ingest/route/prepare_topic/swap_out） | 阶段 3 |
| `src/hivememory/engines/perception/semantic_flow_perception_layer.py` | 持有 Store、Journal、`_domain_lock`、`TriggerManager`；`route_and_ingest`/`ingest_payload` 含 journal retry 协议与 `_compute_apply_digest`；`_maybe_fold_pages` 阈值判断；`settle_topic`/`prepare_settlement`/`commit_settlement`/`abort_settlement`/`swap_out_topic`/`discard_if_empty` | 阶段 3/4 |
| `src/hivememory/patchouli/services/perception.py` | `PerceptionFamiliar` 持有 `memory_library.short_term`（:61）；直接 Store 调用（:115、:164、:180、:242）；LRU 候选选择与 IDLE 过滤（:189-220）；idle 扫描（:303-326）；shutdown 遍历（:328-366）；`_maintenance_scope` 重建访问作用域 | 阶段 4 |
| `src/hivememory/patchouli/runtime/core.py:398-407、541-547` | `_build_perception_layer` 调用 `create_perception_layer`；`PerceptionFamiliar` 注入 `perception_layer` | 阶段 3/4 |
| `src/hivememory/patchouli/control/interaction_apply_journal.py:9` | 仅类型引用 `TopicMaterializeTask` | 不迁移 |
| `src/hivememory/patchouli/control/interaction_submission.py:197-207` | 适配 `PerceptionFamiliar.apply_interaction`（签名不变） | 不迁移 |
| `src/hivememory/patchouli/runtime/route_bindings.py:99-112`、`system.py:94、171` | 绑定 Familiar 公开入口（`prepare_topic`/`evict_topic`/`discard_if_empty`/`manual_settle_topic`/`apply_interaction`/`scan_idle_buffers_once`） | 公开路由名不变 |
| `src/hivememory/prompts/assembler.py:11` | `PerceptionContextConverter`（无状态门面，不在重构范围） | 不迁移 |

#### 11.1.4 Store 消费方盘点

| 消费方 | 用法 | 处置 |
|:---|:---|:---|
| `PerceptionFamiliar`（services/perception.py） | `get`/`count`/`list_by_workspace`/`list_all` + LRU、idle、shutdown 遍历 | 阶段 4 收口到 `TopicBufferService`，Familiar 不再持有 Store |
| `RetrievalFamiliar`（services/retrieval.py:97、117） | `get`、`list_by_workspace`（话题池只读投影） | 保留（读模型） |
| `PatchouliRuntime._build_memory_library`（runtime/core.py:332） | 构造与注入 | 保留（装配） |
| `MemoryLibrary.short_term`（memory_library/library.py） | 持有与健康检查 | 保留 |

`WorkspaceTopicKey` 目前只出现在 `adapters/short_term.py` 内部，上一轮清理已把它收敛到 adapter，本计划维持该封装。

#### 11.1.5 `FlushReason` / `BufferState.FLUSHING` 迁移清单

生产代码（静态搜索确认的全部引用）：

- `core/models/topic.py:21`：`FLUSHING = "flushing"` 定义 → `SETTLING = "settling"`；
- `engines/perception/models.py`：`FlushReason`/`FlushEvent` 定义 → `TriggerReason`（保留七个枚举字符串值：`token_overflow`/`idle_timeout`/`lru_eviction`/`shutdown`/`manual_settle`/`manual_compact`/`manual_delete`），`TopicMaterializeTask.reason` 字段类型随迁；
- `engines/perception/trigger_manager.py`、`semantic_flow_perception_layer.py`、`interfaces.py`、`__init__.py`：随文件迁移/删除收口；
- `hivememory/__init__.py`：顶层导出改名为 `TriggerReason`；
- `patchouli/services/perception.py`：`FlushReason.LRU_EVICTION`/`IDLE_TIMEOUT`/`SHUTDOWN` 三处用法改为 `TriggerReason`；
- `patchouli/control/interaction_apply_journal.py`：仅 `TopicMaterializeTask` 类型引用，无枚举引用。

序列化影响核查：短期 buffer 为进程内存储（`InMemoryShortTermStorage`），`FLUSHING`/`flushing` 无持久化数据依赖；`flushing` 字符串不出现在配置、server 模型或队列 payload 中，重命名安全。`TriggerReason` 枚举字符串值保持不变，因此 `TopicMaterializeTask`/`FlushEvent` 的既有 payload 不受影响。

测试侧引用（随对应阶段重写）：`tests/conftest.py`（`FlushRecorder`/`FlushEventRecorder`/perception fixture）、`tests/unit/agent_runtime/mtp/test_write_chain.py` 与 `test_update_chain.py`（仅断言 `FlushReason.__members__` 不含 `MTP_*`，需跟随新名称）、11.1.1 表列出的全部过期测试，以及 `tests/unit/patchouli/memory_library/test_buffer.py:191`（断言 `FLUSHING.value == "flushing"`）。

#### 11.1.6 冻结契约

以下契约为后续阶段的实施基准，实施中不得偏离或另立第二份定义：

1. **决策矩阵**：七种触发原因与 `TriggerPlan(settle, compact, evict)` 的映射固定为 Plan §4.2 表格；`TOKEN_OVERFLOW`/`MANUAL_COMPACT` 仅 compact，`MANUAL_DELETE` 仅 evict，其余四种 settle 原因均为 `settle=True, evict=True`。
2. **不变量**：`settle=True` 必须 `evict=True`；`TriggerPlan` 至少含一个动作；compact 的 `retain_recent_blocks >= 1`。
3. **settle 统一时序**：所有 `settle=True` 原因共用 `begin_settlement -> 锁外 admission -> complete_settlement/abort_settlement`；`SETTLING` 期间拒绝新 Interaction；接纳成功或无材料才删除 Topic；明确拒绝/异常恢复 `IDLE` 并保留内容；queue 成功 receipt 后的生成失败不重开 Topic。
4. **消除特殊分支**：现行 `resolve_topic` 中 `MANUAL_SETTLE` 的 ValueError 分支随统一协议消失；`MANUAL_SETTLE` 与 automatic settle 共用同一 begin/complete/abort 原语。
5. **命名**：`TriggerReason`、`BufferState.SETTLING`、`TopicBufferService`、`MemoryPerceptionEngine`、`create_perception_engine`、`TriggerPlan`、`SettlementOutcome`（状态枚举 `SettlementStatus`：`ACCEPTED`/`NO_MATERIAL`/`REJECTED`/`NOT_FOUND`）。
6. **迁移禁令**：不新增 `by_key` Store 入口、Store 复合方法、额外状态控制器、per-topic lock；不新增 `topic_buffer_models.py` 或第二份矩阵/`TriggerPlan`；`TriggerPlan` 不携带 `evict_timing`/`defer_evict` 等时序字段。
7. **锁边界**：领域锁（RLock）归 `TopicBufferService`；Relay 摘要生成与 generation queue admission 在锁外；Store/adapter 锁只保护单次 CRUD 与内部索引。

#### 11.1.7 目标依赖方向确认

```text
PerceptionFamiliar (patchouli/services)
  -> TopicBufferService (patchouli/services/topic_buffer.py)
       -> ShortTermMemoryStore (patchouli/memory_library) -> adapter
  -> MemoryPerceptionEngine (engines/perception, 无状态)
TopicBufferService -> MemoryPerceptionEngine（摘要/折叠纯算法）
```

- 允许方向：`patchouli -> engines/perception`（Familiar/Service 引用 Engine、Relay 协议与感知模型）；
- 必须消除的反向依赖：现行 `trigger_manager.py`/`semantic_flow_perception_layer.py` 导入 `patchouli.errors`、`patchouli.control`、`patchouli.memory_library`；阶段 3 完成后 `engines/perception/` 生产代码不得导入 `patchouli.*`（`TopicBusyError` 的使用随状态执行迁入 `TopicBufferService`，journal retry 协议归 Familiar）；
- `engines/perception/` 阶段 3 后只依赖 `core.models`、`core.protocol.models`、`system.config`（fold 阈值配置经注入或参数传入）、`utils.token_estimator`、`i18n` 与自身模块；
- `interaction_apply_journal` 留在 `patchouli/control`，retry 协议（journal 检查、digest 等价性校验、journal 记录）由 Familiar 与 TopicBufferService 的调用顺序承接；
- `context_converter.py` 为无状态渲染门面，不属于本次边界重构范围。
