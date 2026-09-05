---
title: Data Model and Mutability Boundaries
status: current
owner: project
scope: domain-models-snapshots-runtime-state-and-boundary-projection
code_paths:
  - src/hivememory/core/models/
  - src/hivememory/core/protocol/
  - src/hivememory/system/runtime/workspace/
  - src/hivememory/gateway/workflow/
  - src/hivememory/patchouli/memory_library/
  - src/hivememory/agent_runtime/pending_atom/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
related_decisions:
  - docs/architecture/decisions/0001-data-model-mutability-and-boundary-projection.md
related_docs:
  - docs/architecture/workspace.md
  - docs/architecture/boundaries.md
related_inventories:
  - docs/governance/baselines/data-model-phase-i-inventory.md
last_reviewed: 2026-09-05
---

# 数据模型与可变性边界

HiveMemory 的数据模型首先服务于一个长期目标：把记忆从对话日志中的片段变成可以检索、阅读、演化和追溯的知识资产。这个目标既要求模型表达语义，也要求系统知道谁有权修改一份状态、何时形成稳定快照，以及跨子系统传递的对象是否仍会被原所有者改写。

因此，本项目不采用“所有模型一律冻结”的形式主义规则。MemoryAtom、话题 buffer、Gateway 请求状态和 Agent 执行进度处于不同生命周期；把它们都改成不可变对象，不会自动建立正确的所有权。当前设计按模型角色选择可变性：值对象、事件和读取快照倾向于递归不可变；实体、聚合与请求级运行状态可以受控可变；跨边界时则投影为与内部实体脱钩的 DTO 或快照。Workspace 的身份坐标遵循同一原则：`WorkspaceIdentity` 与 `IdentityScope` 是不可变值对象，资源实体仍由各自领域 Store 所有，具体归属与寻址规则见 [Workspace 架构](./workspace.md)。

本文描述当前已经成立的边界，并记录仍然存在的引用泄漏与冻结深度问题。后续规范化工作见[数据模型可变性治理](../governance/data-model/mutability.md)。

## 1. MemoryAtom：语义事务与冰山结构

HiveMemory 不把固定 token 切片或整段 session 直接当作长期记忆。MemoryAtom 试图表达一个相对自包含、能够被独立判断和复用的“语义事务”：它可以是一条事实、一个代码资产、一项用户偏好，也可以是一段需要保留来源的经验结论。

这个粒度不是“只保存最终答案”的机械清洗规则。原始过程里可能包含决定结论可信度的证据、失败尝试和约束；系统应把适合工作上下文的精炼内容与可追溯的原始材料分开，而不是为了降低 token 成本删除证据。由此形成了延续至当前模型的冰山结构：

- `index` 是水面上的检索视图，保存 title、summary、tags、type 和向量等用于发现记忆的信息；
- `payload` 是 Agent 真正消费的主体内容，保存结构化正文及内容格式；
- `artifacts` 保存或引用原始输入、生成产物与 provenance，使精炼结论仍能回到证据；
- `meta` 保存身份、可见性、版本、生命力、验证状态和时间等治理信息；
- `relations` 为版本与知识关系提供结构位置，但当前并非所有关系字段都已有完整业务行为。

Index 与 Payload 分离，解决的是“适合检索的摘要”和“适合阅读的内容”并不相同。Artifacts 再与 Payload 分离，则是为了避免上下文默认携带所有原始材料，同时保留检查、审计与未来重新解释的可能。这个结构只说明数据职责，不保证当前检索只嵌入 index，也不意味着所有 artifact、relation、版本回档或来源反查能力都已完整实现；这些事实以 Patchouli 当前文档为准。

Memory type 是系统对“这份资产应如何被使用”的结构化提示，tags 则是开放的语义描述。两者正交：type 不能退化为任意标签，tags 也不能替代代码对类型的行为约束。旧设计中的固定类型表已经演进，当前枚举和实际消费者以 `core/models/memory.py` 与 Patchouli/Alice 文档为准。

## 2. 模型角色与目标策略

| 角色 | 当前/目标策略 | 传播边界 |
|:---|:---|:---|
| Value Object | 字段和受支持的嵌套数据递归不可变 | 可以跨边界共享 |
| Domain Event | 创建后不修改；变化产生新事件 | 可以跨边界共享 |
| Snapshot / Read Model | 与源实体脱钩、递归只读 | 推荐作为读取结果 |
| Public Request / Result DTO | 默认只读；流式累积必须明确例外 | 只传契约字段 |
| Entity / Aggregate Root | 由唯一所有者受控修改或版本化更新 | 不传播内部引用 |
| Runtime State / Builder | 请求内可变、生命周期有限 | 不跨请求长期持有 |
| Persistence Model | 服从存储映射需要 | 经 mapper/投影进入领域边界 |

“冻结”不是单一保证。本文使用以下四种明确口径：

- `mutable`：对象可在其生命周期内修改；
- `controlled mutable`：只有声明的所有者可以修改；
- `shallow frozen`：只能阻止替换外层字段，内部对象仍可能可变；
- `deep immutable`：受支持的整个对象图都不能原地修改。

字段使用 Pydantic `frozen=True` 或 frozen dataclass，只能证明字段引用不能重新赋值。若内部仍是 list、dict、`Any` 或可变 Pydantic model，就不能称为递归不可变。

## 3. 当前不可变基础设施

`core/models/immutable.py` 提供 `FrozenDict`、`freeze_value()` 与 `freeze_mapping()`。当前递归范围是 JSON 风格 mapping 和 sequence：嵌套 mapping 转为 `FrozenDict`，list/tuple/set/frozenset 转为 tuple，标量直接保留。

这个工具不会自动冻结自定义对象或任意 Pydantic model。把可变对象放进 `Any` 字段后再调用 `freeze_value()`，仍然可能保留同一引用；`FrozenDict` 也只承诺拒绝常见写操作，不承诺可哈希。公共模型必须根据实际嵌套类型判断保证，不能只看构造函数是否调用了 freeze helper。

当前形成较完整不可变链路的模型包括：

- `ActorIdentity`；
- `TurnEvent`、`AgentAction`、`TraceItem`、`TurnRecord`；
- `TopicData`、`TopicSnapshot` 与相关展示值对象；
- Gateway 的公共 decision / command outcome 及多项私有分析结果；
- PendingAtom 领域中的 `WriteFocus`、`UpdateFocus`、`RuntimeScope`、`PendingAtomSnapshot` 和 `PendingAtomMaterializeTask`。

它们通常以重新构造、tuple 聚合或 `model_copy()` 生成新值。Pydantic 的 `model_copy(update=...)` 不会完整重复构造时的字段验证，因此带有业务语义的变化仍应逐步收敛到命名方法或明确的转换器。

## 4. 受控可变与快照投影

### 4.1 Workspace 坐标与资源键

短期 Topic 的复合键仅在 adapter 内部使用；上层通过 `IdentityScope + topic_id` 访问。

`IdentityScope` 只表达一次操作中的 actor 和 Workspace 事实，不把请求、任务或 trace 状态塞入公共身份模型。Topic、Memory、Artifact 和 WorkspaceAsset 的资源访问在相应领域边界组合 WorkspaceIdentity 与资源 ID；其中 `topic_id` 仍按全局唯一身份生成和校验，Workspace 归属由底层存储边界校验。物理复合键本身不复制实体，也不改变由 Patchouli 或 System Store 负责的生命周期。

`WorkspaceAssetStore` 是 System 进程内唯一的 working set，资产的 opaque ref、representation 与 lease 只在该 Store 的生命周期内有效。它不属于通用缓存、队列或事件对象图；关闭时由 System 在 Patchouli drain 完成后清空。共享 runtime 传递 scope 时只携带领域 payload，不因模型中有 Workspace 字段而自动形成一套按 Workspace 分区的可变状态。

### 4.2 Gateway 请求状态

短期 Topic 的 `WorkspaceTopicKey` 不属于公共领域模型或 Store/Perception 契约；这些边界统一使用 `IdentityScope + topic_id`，复合键只由 adapter 构造。

`GatewayExecutionState` 是有意可变的请求级状态，仅由 `GatewayWorkflow` 创建和持有。Step 通过 `GatewayStepResult` 提交更新，由 workflow 校验字段、提交顺序和 finalize 边界；最终只投影为公共 `GatewayProcessResult`，不把内部 state、fallback 细节或 snapshot 暴露给下游。

`GatewayStepResult.updates` 当前只以 `MappingProxyType` 冻结顶层 mapping，嵌套值是否只读依赖各 Step 自己的输出约束。它是私有提交信封，不是通用深度不可变容器。

### 4.3 Topic 实体与读取模型

短期存储 adapter 直接存储冻结的 `TopicData`（含 blocks、摘要、token 计数），Store 与 Port 只交换不可变 `TopicData` 或 `TopicSnapshot` 快照；执行占用不建模为记录字段，由 TopicWorkingSet 的 lease 表达。

“不可变快照 + 单写者编排”是当前最清晰的聚合边界：调用方可以观察话题，但不能通过读取结果改写 Store 内部状态（frozen 模型 + adapter 原样返回存储实例）。新增 append、settle、evict 或 summary update 行为时，应继续由 Patchouli 所有者执行，不能把可变实体直接返回给 public route。当前实现中 Perception 通过不可变 `TopicData` 快照完成领域更新（读取 → model_copy → 整条替换写回），互斥由 TopicWorkingSet 的 lease 保证。

### 4.4 PendingAtom 状态机

`PendingAtom` 与 `PendingAtomSettlement` 是可变状态载体，`PendingAtomSnapshot` 和 materialize task 是对外只读投影。可变性本身用于表达运行期间的 pending、redirect、settled、failed、expired 等迁移；真正需要治理的是写权限仍主要依靠模块约定，部分调用方仍可能直接修改字段。

当前设计要求 PendingAtom Runtime 成为状态迁移的唯一所有者。未来治理应把合法迁移收敛为命令或领域方法，而不是先把状态对象机械冻结。

### 4.5 可变累积后冻结

Alice 在请求内用 `ExecutionProgress` 等对象累积事件，Perception 在交互完成后构造 tuple 化的 `TurnRecord` 与 `LogicalBlock`。Builder/Accumulator 在有限生命周期内可变，完成后产出稳定快照，是合理边界；累积器不应跨 run 传播，也不需要为了形式统一而冻结。

## 5. 当前仍然可变或仅浅层冻结的区域

以下对象尚不能宣称具有统一的递归不可变保证：

| 区域 | 代表对象 | 当前风险/理由 |
|:---|:---|:---|
| 记忆领域 | `MemoryAtom` 及 meta/index/payload/artifacts/relations | 多层 list/dict 与模型可被直接修改；聚合写入口尚未统一 |
| 通用协议 | `RetrievalResponse`、`AgentRunContext`、`AgentRunResult`、`InteractionPayload` | 公共 DTO 与运行结果仍共享可变 list/model |
| Alice Runtime | frame、progress、generation result | 请求级累积状态有意可变，但所有权标记不统一 |
| 应用服务结果 | `StreamPrelude`、`PreparedAgentRun`、`PassiveIngressOutcome` 等 | frozen 外壳包裹可变模型、list 或 dict |
| Gateway Step | `GatewayStepResult.updates` | 只冻结顶层 mapping |

浅层冻结外壳可以减少误替换，却不能作为跨异步边界的数据隔离。随着缓存、并发任务或事件分发增加，共享引用可能让历史结果被后来写回。治理时应根据语义选择：把内容投影成真正只读 DTO，或把它明确限制为请求内临时句柄。

## 6. 当前约束

在完整治理项目启动前，新增代码至少遵守以下约束：

1. public route 不直接返回 Store、Repository 或 Runtime 持有的可变实体；
2. 新增值对象、事件和快照时，序列默认用 tuple，JSON mapping 用 `FrozenDict`；
3. frozen 模型只要包含 list、dict、`Any` 或可变 model，就按 shallow frozen 描述；
4. 新增可变状态必须说明唯一所有者和生命周期，不能被另一子系统长期持有；
5. 有业务含义的状态变化优先使用命名方法或所有者命令，避免扩散裸字段写入；
6. Builder、Accumulator、执行帧和持久化实体不因“统一风格”被强行冻结；
7. snapshot 构造后不得随源实体继续变化；
8. 边界投影不得依靠“用 frozen dataclass 包住实体引用”伪装隔离。

## 7. 尚未成立的旧设计断言

历史项目文档曾提出来源置信度阶梯、用户输入自动 `confidence=1.0/immutable`、Git-like 完整版本栈、自动历史 GC、严格/创意检索阈值、用户标签永久保留、relations 驱动知识图谱等方案。这些构想解释了项目为何重视 provenance、验证状态和记忆演化，但当前代码没有把它们完整实现为全局策略。

它们不能作为当前事实，也不能从字段存在反推行为已经落地。若未来重新采用其中任何一项，需要先明确威胁模型、所有权、冲突处理、可逆性和测试，再进入 Plan/ADR；尤其不能把“用户说过”自动等同于全局客观真相，也不能让模型自行维护的置信度替代来源证据。

## 8. 设计矛盾检查

修改数据模型时，应检查：

1. 这个对象是值、事件、快照、实体还是请求级运行状态？其可变性是否与角色一致？
2. 若对象可变，谁是唯一写入者，合法变化通过什么入口发生？
3. public route、缓存、事件或异步任务是否传播了内部实体引用？
4. `frozen=True` 是否只冻结外壳，嵌套 list/dict/自定义模型还能否被修改？
5. snapshot 是否与源对象真正脱钩，源对象后续变化会不会回写历史读结果？
6. copy-on-write 是否绕过字段验证或领域状态机？
7. 新字段只是为未来设计预留，还是已经有消费者、持久化与生命周期行为？

## 9. 验证入口与相关文档

主要验证入口：

- `tests/unit/gateway/test_phase3b_contracts.py`；
- `tests/unit/patchouli/memory_library/test_memory_library.py`；
- `tests/unit/agent_runtime/pending_atom/test_runtime.py`；
- `src/hivememory/core/models/immutable.py`、`interaction.py`、`topic.py`、`memory.py`、`pending.py`；
- `src/hivememory/core/protocol/models.py`；
- `src/hivememory/gateway/workflow/state.py`、`steps.py`。

相关设计：[系统边界与所有权](./boundaries.md)、[MemoryLibrary](../patchouli/memory-library.md)、[PendingAtom](../alice/pending-atom.md)、[数据模型边界 ADR](./decisions/0001-data-model-mutability-and-boundary-projection.md)、[Phase I 数据模型与边界清单](../governance/baselines/data-model-phase-i-inventory.md)与[后续治理主题](../governance/data-model/mutability.md)。
