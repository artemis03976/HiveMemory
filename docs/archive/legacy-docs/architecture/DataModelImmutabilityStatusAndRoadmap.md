---
title: Legacy Data Model Immutability Status and Roadmap
status: archived
owner: project
scope: legacy-data-model-status-and-governance-roadmap
archived_at: 2026-07-29
superseded_by:
  - docs/architecture/data-model.md
  - docs/architecture/decisions/0001-data-model-mutability-and-boundary-projection.md
  - docs/governance/data-model/mutability.md
---

> 本文混合了阶段性代码现状、架构裁定和未来治理步骤，现已拆分归档。当前模型边界见[数据模型与可变性边界](../../../architecture/data-model.md)，长期裁定见 [ADR-0001](../../../architecture/decisions/0001-data-model-mutability-and-boundary-projection.md)，未排期工作见[数据模型治理](../../../governance/data-model/mutability.md)。

# 项目级不可变数据模型现状与规范化规划

**文档状态**: Current State / Technical Debt Proposal

**基线版本**: Gateway Phase 3F（提交 `ad63478`）

**最后更新**: 2026-07-18

**适用范围**: `core/models/`、`core/protocol/`、`gateway/`、`patchouli/`、`alice/`、`agent_runtime/`、`system/`

**目的**: 记录当前不可变模型的实际保证、已知边界和后续项目级规范化方向。本文件不是要求当前阶段立即完成全量迁移的实施规范。

---

## 1. 背景与结论

Gateway Phase 3 为保证请求级决策过程可推理、可隔离，建立了一组不可变输入、快照、Step 结果和公共输出。同期，话题只读模型与单轮交互模型也收敛到了 `core`，因此当前实现已经不只是 Gateway 的局部约定，而是形成了覆盖以下链路的“不可变岛”：

```text
Identity
  ├─ TurnEvent -> AgentAction / TraceItem -> TurnRecord -> LogicalBlock
  ├─ TopicSnapshot / TopicData
  └─ Gateway Context / State Snapshot -> GatewayDecision / Command Outcome
```

但系统尚未统一定义以下模型角色及其写权限：

- 可变实体（Entity）与聚合根（Aggregate Root）；
- 不可变值对象（Value Object）与事件（Event）；
- 只读快照（Snapshot / Read Model）；
- 跨子系统传输对象（DTO / Protocol）；
- 请求级工作状态与状态机运行时。

因此，当前方案适合作为 Gateway 和 Topic 读取边界的阶段性实现，不应直接扩张为“所有模型一律冻结”。后续需要单独立项，统一模型分类、冻结深度、唯一写入所有者和变更 API。

`model_copy()`、tuple 替换和重新构造本身不是技术债。它们是不可变值对象正常的 copy-on-write 手段。真正的技术债是：调用方目前需要自行理解哪些对象可以复制、哪些对象可以原地修改、嵌套对象是否仍然可变，以及由谁负责提交合法变更。

---

## 2. 本文使用的不可变术语

### 2.1 字段冻结

Pydantic `ConfigDict(frozen=True)` 或 `@dataclass(frozen=True)` 禁止对模型字段重新赋值，例如禁止 `model.status = ...`。

字段冻结不等于递归不可变。如果字段仍然引用 `list`、`dict` 或可变 Pydantic model，调用方依然可以修改其内部内容。

### 2.2 递归不可变

从模型根节点可达的业务数据都不能被原地修改。当前项目通过以下组合实现这一目标：

- Pydantic `frozen=True`；
- 序列字段使用 tuple；
- JSON 风格 mapping 使用 `FrozenDict`；
- 嵌套模型本身也采用不可变模型。

当前 `freeze_value()` 只递归处理 mapping 和 list/tuple/set/frozenset。自定义类、任意 Pydantic model 或其他可变对象会原样保留，因此 `Any` 字段不能仅凭经过 `freeze_value()` 就宣称整个对象图递归不可变。

### 2.3 受控可变

模型允许变化，但只由一个明确的聚合所有者通过受控方法修改。典型例子是：

- `GatewayExecutionState` 只由 `GatewayWorkflow` 持有；
- `SemanticBuffer` 只应由短期记忆 Store 管理；
- Alice 的执行帧只在一次 Agent run 内累积运行状态。

受控可变不是冻结失败，而是状态实体和工作状态应有的实现方式。问题在于当前只有部分模块明确执行了所有权约束。

### 2.4 浅层冻结外壳

冻结 dataclass 如果持有 list、dict 或可变模型，只能保证外层字段引用不会被替换，不能保证字段内容不被修改。本文将这类模型称为“浅层冻结外壳”，不将其计入递归不可变模型。

---

## 3. 当前代码现状

### 3.1 Core 不可变基础设施

`src/hivememory/core/models/immutable.py` 提供：

- `FrozenDict`：保留 dict 序列化兼容性，拒绝常见原地写操作；
- `freeze_value()`：递归冻结 JSON 风格映射和序列；
- `freeze_mapping()`：将 mapping 转换为递归 `FrozenDict`。

当前保证的边界：

- mapping 的嵌套 mapping 会继续转换为 `FrozenDict`；
- list、tuple、set、frozenset 统一转换为 tuple；
- 标量直接保留；
- 自定义可变对象直接保留；
- `FrozenDict` 的目标是防止修改，不承诺可哈希。

### 3.2 Core 递归不可变领域模型

| 模型 | 当前角色 | 当前保证 | 主要写入方式 |
| --- | --- | --- | --- |
| `Identity` | 值对象 | frozen；字段为标量 | 重新构造 |
| `TurnEvent` | 领域事件 | frozen；`tool_args` 递归冻结 | `model_copy()` 生成新事件 |
| `AgentAction` | 单轮动作值对象 | frozen；结果为 tuple；`tool_args` 递归冻结 | Reducer 重新聚合 |
| `TraceItem` | 派生摘要值对象 | frozen；字段为标量 | Reducer 重新生成 |
| `TurnRecord` | 单轮内容快照 | frozen；事件、动作、摘要均为 tuple | 一次性构造 |
| `TopicLastTurn` | 展示值对象 | frozen；字段为标量 | 从 Topic 数据投影 |
| `TopicSnapshot` | 话题列表快照 | frozen；字段为标量或冻结值对象 | 从 `TopicData` 投影 |
| `LogicalBlock` | 感知层逻辑块 | frozen；嵌套冻结 `TurnRecord` | Perception 一次性构造 |
| `TopicData` | 话题完整只读模型 | frozen；blocks 为 tuple | 从 `SemanticBuffer` 投影 |

`TopicData.recent_blocks()` 返回 tuple，不向调用方暴露内部可变容器。

### 3.3 Pending Atom 的混合策略

Pending Atom 领域目前同时存在冻结值对象和可变状态句柄：

- 已冻结：`WriteFocus`、`UpdateFocus`、`RuntimeScope`、`PendingAtomMaterializeTask`、`PendingAtomSnapshot`；
- 当前可变：`PendingAtom`、`PendingAtomSettlement`；
- `RuntimeScope.with_action()` 使用 `model_copy()`，`for_child()` 重新构造子 scope；
- `PendingAtomSnapshot` 是对外只读视图，`PendingAtom` 是运行期状态机载体。

这种划分方向合理，但仍需补齐“只有 PendingAtom Runtime 可以执行状态迁移”的项目级约束，避免调用方直接写 `status` 或 `settlement`。

### 3.4 Gateway 公共协议

`src/hivememory/core/protocol/gateway.py` 中以下模型已冻结：

- `RetrievalPlan`；
- `CommandExecutionResult`；
- `GatewayDecision`；
- `GatewayCommandOutcome`；
- `GatewayDecisionOutcome`。

其中：

- `GatewayDecision.search_keywords` 使用 tuple；
- `GatewayDecision` 只嵌套冻结模型或枚举/标量；
- `CommandExecutionResult.data` 与 `client_action` 使用 `FrozenDict` 递归处理 JSON 风格值；
- Gateway 最终只向下游暴露 `GatewayProcessResult` 判别联合，不暴露内部 state、snapshot、fallback 或事件。

该部分可以视为当前最完整的跨子系统不可变协议实现。

### 3.5 Gateway 私有模型和执行状态

以下 Gateway 私有模型已冻结：

- `CandidateTopics`；
- `UserQueryAnalysisContext`；
- `UserQueryAnalysisResult`；
- `TopicRoutingResult`、`InterceptorResult`；
- command 定义、解析结果与注册项；
- `GatewayStateSnapshot`；
- `GatewayStepResult`；
- `GatewayWorkflowStep` 及固定拓扑节点。

`GatewayExecutionState` 是有意保留的可变 dataclass：

- 仅由 `GatewayWorkflow` 创建和持有；
- `_apply_step_result()` 是唯一 Step 写入口；
- 禁止覆盖初始化字段；
- 拒绝未知字段；
- finalize 后拒绝继续提交；
- `finalize()` 显式投影为公共 Gateway outcome。

已知限制：`GatewayStepResult.updates` 只使用 `MappingProxyType` 冻结顶层 mapping，嵌套值是否不可变依赖各 Step 的输出契约。它适合作为私有提交信封，但不能作为通用递归不可变容器。

### 3.6 Patchouli 的实体与只读投影

`SemanticBuffer` 是可变 Pydantic model，包含：

- 可变 blocks list；
- topic 标题、摘要和状态；
- token 计数；
- 更新时间和访问时间。

Store 内部通过原地写入维护其生命周期，对外读取时转换为冻结 `TopicData` 或 `TopicSnapshot`。这是当前最清晰的“可变实体 + 不可变读取模型”实现。

该边界仍主要依靠模块约定：未来需要确保 Store 不直接返回 `SemanticBuffer`，并将 append block、touch、flush、clear、summary update 等变更收敛为 Store 或聚合方法。

### 3.7 浅层冻结的服务结果

以下模型使用 frozen dataclass，但内部仍含可变对象，当前只能视为浅层冻结：

- `StreamPrelude`：包含 `list[TopicSnapshot]`、`list[Any]`；
- `PreparedAgentRun`：包含可变 `AgentRunContext`、`dict` generation options 和 `StreamPrelude`；
- `NonStreamingChatAgentOutcome`：包含可变 `AgentRunResult`；
- `PassiveIngressOutcome`：可包含可变 `RetrievalResponse`、`InteractionPayload`。

这些模型能防止调用方替换外层字段，但不能形成跨子系统的递归只读保证。命名和文档中不应把 `frozen=True` 直接等同于完整不可变。

### 3.8 仍然可变的主要模型

当前尚未纳入统一冻结或写权限策略的主要业务模型包括：

| 区域 | 代表模型 | 当前特征 |
| --- | --- | --- |
| 记忆领域 | `MemoryAtom`、`MetaData`、`IndexLayer`、`PayloadLayer`、`Artifacts`、`RelationLayer` | 多层嵌套 list/dict，可被直接修改 |
| 通用协议 | `RetrievalRequest`、`RetrievalResponse`、`AgentRunContext`、`AgentRunResult`、`InteractionPayload`、`MTPExecutionResult` | 公共 DTO 与运行结果混合使用可变 list/model |
| Alice Runtime | `ExecutionFrame`、`ExecutionProgress`、`GenerationResult`、`StreamChunk`、`FrameExecutionResult` | 请求级累积工作状态，有意可变但所有权未形成统一标记 |
| 通用交互 | `StreamMessage` | mutable model，`tool_args` 为普通 dict |
| Pending 状态机 | `PendingAtom`、`PendingAtomSettlement` | 状态迁移载体，可直接字段写入 |
| 检索链路 | 检索请求、响应及部分中间结果 | 复制、追加与原地更新并存 |

这些模型不能简单批量改为 frozen。部分是运行状态或聚合根，应保留受控可变；部分是跨边界 DTO，适合迁移为递归不可变快照。

---

## 4. 当前常见变更模式

### 4.1 Copy-on-write

当前典型调用包括：

- Agent Runtime 使用 `TurnEvent.model_copy(update=...)` 更新事件状态；
- Alice orchestrator 复制事件以调整 sequence 或替换 CALL 相关事件；
- `RuntimeScope` 通过复制或重新构造派生 action/child scope；
- Gateway finalize 重新构造公共决策，而不是把私有分析结果直接泄漏给下游。

该模式适合值对象、事件和快照。需要注意 Pydantic v2 的 `model_copy(update=...)` 将 update 数据视为可信输入，不会像重新构造模型一样完整执行字段验证。因此后续应把有业务含义的复制收敛为模型方法或领域转换器。

### 4.2 可变累积后一次性冻结

Alice 在 `ExecutionProgress` 中使用 list 累积事件；Perception 收到完整交互后，一次性构造 tuple 化的 `TurnRecord` 和 `LogicalBlock`。

这是 Builder/Accumulator 到 Snapshot 的合理边界。后续应显式命名和记录该边界，而不是把累积器本身强行冻结。

### 4.3 可变实体投影为只读模型

Patchouli 在 Store 内维护 `SemanticBuffer`，读取时投影为 `TopicData`。这应成为后续聚合治理的参考模式：实体由单一所有者修改，跨边界只传不可变快照。

### 4.4 浅层冻结包装可变内容

部分应用服务 outcome 只冻结外层 dataclass，实际仍共享内部可变对象。这种方式只能防止意外替换，不能作为数据访问权限边界。后续需要根据调用时序选择：

- 将内部结果也转换为不可变 DTO；或
- 明确声明它是请求内临时句柄，并限制其传播范围。

---

## 5. 已确认的技术债

### 5.1 模型角色没有统一登记

同一个 `frozen=True` 目前可能表示值对象、事件、快照、请求信封或只是方便防误写的外壳。可变模型也没有统一标记它是聚合根、运行状态还是历史遗留 DTO。

### 5.2 唯一写入所有者只在局部成立

Gateway state 和 Topic Store 的所有权相对清晰，但 Memory、Pending Atom、Agent run DTO 等区域仍允许多个调用方直接修改字段或嵌套容器。

### 5.3 冻结深度不一致

项目同时存在：

- 标量字段 frozen model；
- tuple + frozen nested model；
- `FrozenDict` 递归 JSON 冻结；
- frozen dataclass 包含 list/dict；
- `MappingProxyType` 只冻结顶层 mapping。

如果接口只声明“返回 frozen model”，调用方无法据此判断真实写权限。

### 5.4 裸复制缺少领域语义

`model_copy(update={"status": ...})`、`model_copy(update={"sequence": ...})` 可以工作，但没有表达合法迁移、重排规则和字段组合不变量。调用点增多后，验证逻辑容易分散。

### 5.5 公共 DTO 可能泄漏可变引用

浅层冻结 outcome、可变 `RetrievalResponse` 和 `AgentRunResult` 会让跨层调用方共享同一个可变对象图。异步执行、事件分发或缓存引入后，这类引用泄漏会增加竞态和历史数据被回写的风险。

### 5.6 缺少项目级检查机制

当前测试主要验证单个模型或 Gateway workflow 不变量，尚未系统覆盖：

- 嵌套 list/dict 是否可被修改；
- public route 是否返回内部实体引用；
- 可变实体是否只有指定所有者写入；
- snapshot 构造后是否会随源对象变化；
- `model_copy()` 是否绕过了必要验证。

---

## 6. 主路径稳定前的临时约束

在系统达到可用状态、正式启动规范化项目前，只执行低成本防扩散约束，不进行全量重构：

1. 新增跨子系统 public route 时，不直接返回可变实体或 Store 内部对象。
2. 新增值对象、事件和快照时，序列默认使用 tuple，JSON mapping 使用 `FrozenDict`。
3. frozen dataclass/Pydantic model 如果包含 list、dict、`Any` 或可变 model，必须按“浅层冻结”看待，不宣称递归不可变。
4. 新增可变状态时，在模型文档中写明唯一所有者和生命周期，不允许跨子系统长期持有。
5. 新增有业务含义的状态变更优先使用命名方法，避免继续扩散裸 `model_copy(update=...)`。
6. 不为了统一形式冻结 Builder、Accumulator、执行帧或数据库实体。
7. 不在当前阶段引入兼容适配层；未来迁移按聚合一次更新定义和所有调用方。

这些约束用于控制新增债务，不要求立即清理现有调用点。

---

## 7. 项目级目标规范

### 7.1 按语义选择可变性

| 模型角色 | 目标策略 | 跨子系统传播 |
| --- | --- | --- |
| Value Object | 递归不可变 | 可以直接传播 |
| Domain Event | 递归不可变，创建后不修改 | 可以直接传播 |
| Snapshot / Read Model | 递归不可变，与源实体脱钩 | 推荐传播 |
| Public Request/Result DTO | 默认递归不可变；确有流式累积需求时明确例外 | 可以传播 |
| Entity / Aggregate Root | 受控可变或版本化不可变，由领域需求决定 | 不直接传播内部引用 |
| Runtime State / Builder | 请求内可变，单一所有者 | 不跨生命周期传播 |
| Persistence Model | 服从存储映射需求 | 通过 Mapper 转换为领域模型或快照 |

### 7.2 唯一写入所有者

每个可变聚合必须声明：

- 谁创建；
- 谁可以修改；
- 合法修改命令；
- 何时生成快照；
- 快照是否脱离源对象；
- 谁负责持久化和版本冲突处理。

调用方只能提交命令或调用聚合方法，不能直接修改嵌套字段。

### 7.3 冻结保证分级

未来公共文档和类型注释应使用明确等级：

- `mutable`：允许调用方在所有权范围内修改；
- `controlled mutable`：只有指定聚合所有者可修改；
- `shallow frozen`：只冻结外层字段引用；
- `deep immutable`：整个受支持对象图递归不可变。

不再使用含糊的“已冻结”同时描述上述不同保证。

### 7.4 领域变更 API

对值对象和事件，使用返回新对象的命名方法，例如：

- `TurnEvent.resequenced(sequence)`；
- `TurnEvent.with_status(status)`；
- `RuntimeScope.with_action(action_id)`；
- `GatewayDecision.with_retrieval_plan(plan)`（仅在确有消费者时增加）。

对聚合根，使用表达业务动作的方法，例如：

- `SemanticBuffer.append_block(...)`；
- `SemanticBuffer.touch(...)`；
- `PendingAtom.transition_to(...)`；
- `MemoryAtom.record_access(...)`。

这些名称只是目标语义示例，正式实现前仍需结合调用方和领域不变量设计，不能机械添加包装方法。

### 7.5 边界投影

跨子系统 route、事件总线、缓存和异步任务边界应发送：

- 递归不可变 DTO；或
- 已序列化数据；或
- 与内部实体完全脱钩的 Snapshot。

禁止把内部实体引用包装在 frozen dataclass 中后视为已完成隔离。

---

## 8. 后续规范化项目实施阶段

### Phase I：盘点与架构决策

目标：先明确语义和所有权，不修改业务行为。

工作项：

1. 建立业务模型清单，记录定义位置、角色、冻结等级、嵌套可变字段、创建者、写入者和消费者。
2. 记录所有跨子系统 public/local route、运行时事件和缓存边界的数据类型。
3. 为 Turn、Topic、Memory、Pending、Agent Run、Retrieval 分别确定聚合边界。
4. 编写 ADR，确定递归不可变支持范围、Pydantic/dataclass 使用规则和 persistence mapper 策略。
5. 建立性能基线，测量大 Topic、Memory 列表和长事件流的复制成本。

交付物：模型矩阵、聚合所有权图、ADR、迁移优先级和性能基线。

### Phase II：统一不可变原语与验证方式

目标：提供一致的实现手段，消除“看似冻结”的歧义。

工作项：

1. 明确 `FrozenDict` 支持的数据边界及序列化行为。
2. 决定是否需要公共 immutable model 基类；没有足够重复收益时不引入。
3. 为深度不可变 DTO 建立嵌套 list/dict、自定义对象和 `Any` 字段检查。
4. 约束 `model_copy(update=...)`：业务字段更新进入领域方法，配置和测试复制保留例外。
5. 建立 Snapshot 与 Entity 的命名、模块和导出规则。

交付物：基础类型、测试辅助函数、编码规范和 CI 初始检查。

### Phase III：Turn 与 Topic 聚合收敛

目标：优先治理已经形成不可变基础、且贯穿主聊天链路的模型。

工作项：

1. 将 TurnEvent 的 sequence/status 变更收敛为领域方法或 Reducer。
2. 明确 `ExecutionProgress` 是请求级 Builder，完成后只产出 `TurnRecord`。
3. 禁止 Perception/Patchouli 以外调用方构造或修改 Topic 内部实体。
4. 为 `SemanticBuffer` 增加完整领域变更 API，由 Store 统一持有。
5. 验证 `TopicData`/`TopicSnapshot` 与源 buffer 脱钩，读取不改变 LRU 或生命周期，除非 route 明确要求 touch。

交付物：稳定的 Turn 快照链、Topic 聚合写边界和无引用泄漏测试。

### Phase IV：Memory 与 Pending 状态机治理

目标：处理风险最高、业务语义最复杂的可变聚合。

工作项：

1. 决定 `MemoryAtom` 采用受控可变聚合还是版本化不可变聚合，不能先冻结再补生命周期语义。
2. 收敛 metadata 访问计数、生命力、验证状态、index/payload 更新和关系变更入口。
3. Repository 不返回可被任意调用方修改的持久化实体引用。
4. 将 `PendingAtom` 合法迁移、settlement 回填和失败处理集中到 Pending Runtime。
5. 对并发更新引入明确的 version/optimistic lock 规则。

交付物：Memory/Pending 聚合 API、Repository 边界、迁移测试和并发规则。

### Phase V：公共协议与应用结果收敛

目标：消除跨子系统的浅层冻结和可变 DTO 泄漏。

工作项：

1. 评估并迁移 `RetrievalRequest/Response`。
2. 将 `AgentRunContext/Result` 区分为请求构建状态和完成后只读结果。
3. 将 `InteractionPayload` 的集合字段迁移为 tuple，或在边界构造专用快照。
4. 收敛 `StreamPrelude`、chat outcome、passive outcome 的嵌套可变引用。
5. 明确 `StreamMessage` 是事件还是流式 Builder，并据此选择冻结策略。
6. 公共 route 返回值增加序列化往返和深度不可变测试。

交付物：递归不可变公共 DTO、明确的流式状态边界和应用服务结果契约。

### Phase VI：强制执行与旧入口清理

目标：让规范由工具和测试保证，不再依赖人工记忆。

工作项：

1. CI 检查新增 public DTO 中的可变容器和无约束 `Any`。
2. 依赖搜索确保外部模块不导入内部 Entity/Runtime State。
3. 删除已迁移聚合的直接字段写入、兼容属性和旧构造入口。
4. 增加架构测试，验证 route 返回类型和模块依赖方向。
5. 更新开发文档和代码评审清单。

交付物：架构检查、完整测试集、旧入口删除记录和稳定规范文档。

---

## 9. 推荐迁移优先级

按风险和收益排序：

1. **P0：跨子系统可变引用泄漏**。优先处理 public route、事件、缓存和后台任务共享的对象。
2. **P1：Turn / Topic 主链路**。现有基础最好，改造范围可控，能形成标准样板。
3. **P1：Pending 状态迁移**。直接字段写入可能破坏生命周期不变量。
4. **P2：MemoryAtom 聚合**。收益高但涉及持久化、索引和生命周期，应单独设计。
5. **P2：Agent Run / Retrieval / Interaction DTO**。在主路径接口稳定后统一收敛。
6. **P3：配置、展示和低风险内部模型**。按实际问题逐步治理，不追求形式统一。

---

## 10. 项目验收标准

规范化项目完成时应满足：

1. 所有核心业务模型均登记角色、冻结等级和唯一写入所有者。
2. 所有跨子系统公共结果要么递归不可变，要么有经过评审的明确例外。
3. Store/Repository 不向调用方泄漏内部可变实体引用。
4. 所有聚合状态变化通过命名 API，并集中校验业务不变量。
5. 生产代码中的裸 `model_copy(update=...)` 仅保留已登记的低风险场景。
6. Snapshot 构造后不随源 Entity 的后续修改变化。
7. 测试覆盖嵌套容器修改、非法状态迁移、跨用户/跨聚合写入和序列化往返。
8. 复制和投影性能满足长会话、大话题和批量检索基线。
9. 新规范不依赖旧模型重导出或长期兼容适配。

---

## 11. 非目标

后续规范化项目不应承担以下目标：

- 不追求所有 Python 对象都可哈希；
- 不冻结请求内 Builder、执行帧和必要的高频累积器；
- 不通过深拷贝掩盖不清晰的所有权；
- 不在缺少性能数据时将所有 list 机械替换为 tuple；
- 不把数据库 ORM/Pydantic 存储模型直接当作跨子系统领域协议；
- 不为旧写入方式建立长期双轨兼容层。

---

## 12. 当前决策

在 Gateway Phase 3F 完成后的当前阶段：

- 保留现有 Gateway、Turn 和 Topic 不可变模型；
- 保留 `GatewayExecutionState`、`SemanticBuffer`、Alice 执行状态的受控可变设计；
- 暂不开展全项目批量冻结；
- 以第 6 节临时约束控制新增技术债；
- 等主聊天、被动接入、Memory 生命周期和 Agent Runtime 主路径达到可用稳定状态后，再按第 8 节单独立项实施。
