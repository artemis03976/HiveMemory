---
title: Data Model Mutability Phase I Inventory
status: current
owner: project
scope: model-role-ownership-boundary-and-copy-baseline-inventory
code_paths:
  - src/hivememory/core/models/
  - src/hivememory/core/protocol/
  - src/hivememory/gateway/
  - src/hivememory/system/runtime/
  - src/hivememory/system/services/passive/
  - src/hivememory/patchouli/
  - src/hivememory/agent_runtime/
  - src/hivememory/alice/
  - src/hivememory/server/models/
related_docs:
  - docs/plans/data-model-mutability-governance.md
  - docs/plans/durability-d0-state-inventory.md
  - docs/plans/idempotency-i0-operations-inventory.md
  - docs/plans/identity-s0-threat-model-inventory.md
  - docs/plans/v0.6.1-local-work-queue-runtime.md
  - docs/architecture/data-model.md
  - docs/architecture/decisions/0001-data-model-mutability-and-boundary-projection.md
last_reviewed: 2026-08-07
---

# Phase I 数据模型与边界清单

本文是[数据模型可变性治理计划](./data-model-mutability-governance.md) **Phase I** 的交付物。Phase I 不修改模型实现，而是冻结当前模型角色、生命周期、冻结深度、所有权和传播边界，并为后续治理建立复制性能基线。

本清单与另外三项 v0.6.1 前置清单分工如下：

- [D0](./durability-d0-state-inventory.md)回答“状态保存在哪里、跨重启承诺到什么程度”；
- [I0](./idempotency-i0-operations-inventory.md)回答“哪个业务操作可以安全重放”；
- [S0](./identity-s0-threat-model-inventory.md)回答“模型属于谁、谁有权读取或执行”；
- 本文回答“跨边界实际传递的对象是什么、谁可以修改、是否与源状态脱钩、复制成本如何”。

Phase I 的四项任务：

1. 建立主要业务模型矩阵，登记定义位置、角色、生命周期、冻结等级、嵌套可变字段、创建者、写入者和消费者；
2. 记录 public/local route、RuntimeEvent、cache 和 task 边界承载的模型；
3. 绘制 Turn、Topic、Memory、PendingAtom、Retrieval 和 Agent Run 的所有权与投影关系；
4. 建立大 Topic、Memory 列表和长事件流的复制性能基线。

## 1. 范围、口径与调研方法

### 1.1 纳入范围

矩阵纳入满足至少一项条件的模型：

- 构成 Turn、Topic、Memory、PendingAtom、Retrieval 或 Agent Run 聚合；
- 通过 HTTP、GlobalSystemBus、子系统 local bus 或 RuntimeEvent 跨边界；
- 被 cache、registry、outbox、scheduler 或后台 task 长期持有；
- 计划成为 v0.6.1 Work Queue 的 payload、record、result 或领域投影。

配置模型、纯异常、只在一个函数内使用的临时解析对象和无状态算法结果不逐项列入；若它们被跨边界长期持有，则按其承载边界登记。

### 1.2 冻结等级

| 等级 | 本文口径 |
|:--|:--|
| `mutable` | 调用方可以直接修改字段或嵌套容器，且没有唯一写入者边界 |
| `controlled mutable` | 对象有意可变，但声明了唯一所有者和有限生命周期；外部仍可能拿到原始引用时需单独标风险 |
| `shallow frozen` | 外层字段不可重新赋值，但至少一个嵌套对象、list/dict/`Any` 或运行时句柄仍可变 |
| `deep immutable` | 在当前支持的数据范围内，对象图没有公开的原地修改路径；变化通过新对象、reducer 或 snapshot 产生 |

`frozen=True` 或 frozen dataclass 只证明外层字段不能重新赋值。`FrozenDict` 只递归冻结 JSON 风格 mapping/sequence，不冻结任意自定义对象；因此本文只在嵌套成员同样不可变时标为 `deep immutable`。

### 1.3 调研方法

- 按构造、写入、读取和序列化调用点静态追踪六条聚合链；
- 检查 Pydantic `ConfigDict`、dataclass frozen 标记、list/dict/`Any` 字段和 `model_copy(update=...)`；
- 检查 store/cache/registry 返回的是 snapshot、投影还是内部原始引用；
- 检查 Global/local route bindings、HTTP response mapper、RuntimeEvent ring buffer 和后台 task spec；
- 使用 [benchmark_data_model_phase_i.py](../../scripts/benchmark_data_model_phase_i.py) 对当前投影、完整深拷贝和候选队列序列化进行本地基准测试。

## 2. 全局结论摘要

1. **Turn、Topic 读取模型和 Gateway 决策已经形成可复用的不可变岛。** `Identity`、`TurnEvent`、`TurnRecord`、`LogicalBlock`、`TopicData`、`TopicSnapshot` 和 Gateway decision/result 主要由 frozen Pydantic、tuple 与递归 JSON 冻结构成，可安全共享既有不可变成员。
2. **Memory 是当前最大可变聚合。** `MemoryAtom` 的 meta/index/payload/artifacts/relations 全部可变，多个写路径直接修改嵌套字段；Global/local route、retrieval response 和 Koakuma cache 都传播同一个可变类型，没有独立 read model。
3. **PendingAtom 有清晰 Runtime 所有者，但原始引用仍会出境。** `PendingAtomRuntime.get()` / `all_atoms()` 返回内部实体，状态机只靠调用约定维持；`PendingAtomSnapshot` 与 `PendingAtomMaterializeTask` 已是良好投影。
4. **Agent Run 使用“可变执行 PCB → 可变公共 DTO → 不可变 TurnRecord”的混合链。** `ExecutionFrame` / `ExecutionProgress` 的受控可变合理；问题在于 `AgentRunContext`、`AgentRunResult` 和 `InteractionPayload` 仍含可变 list、`Any`、`AgentProfile` 与 `MemoryAtom` 引用。
5. **frozen 外壳不等于可靠队列载荷。** `SealedTurn` 包含可变 `InteractionPayload`；`PreparedAgentRun` 包含可变 `AgentRunContext` 和 dict；`MemoryGenerationTaskSpec` 包含可变 `GenerationRequest`；`MTPExecutionContext` 包含可变 `AgentProfile` 和 policy。它们都只能标为 `shallow frozen`。
6. **HTTP 层大多已投影，主要风险集中在进程内总线和共享容器。** Memory、Topic、Task 的 REST response 都重新构造 DTO；Global/local bus、RuntimeEvent、Koakuma cache、PendingAtom store、Memory task registry 和 Chat run registry 则传递或返回内部可变对象。
7. **当前高效 Topic/Turn 投影依赖共享不可变子对象。** 基准显示对整个对象图做深拷贝会比当前投影慢两个数量级，并产生显著额外内存；后续治理应继续采用“受控可变聚合 + 不可变成员共享 + 边界投影”，不应全局机械 deep-copy。
8. **v0.6.1 Queue 不能直接持久化现有业务对象。** Work payload 必须是带 `kind/schema_version` 的规范化快照；不得保存 coroutine、task、lock、service、任意 `Any` 或可变领域实体引用。

## 3. 主要模型矩阵

### 3.1 Identity、Gateway、Turn 与 Topic

| 模型（位置） | 角色 / 生命周期 | 冻结等级 | 嵌套可变字段 | 创建者 → 唯一写入者 | 主要消费者 / 现状 |
|:--|:--|:--|:--|:--|:--|
| `Identity`（[interaction.py](../../src/hivememory/core/models/interaction.py#L20)） | Value Object；request/run/work 生命周期 | deep immutable | 无；均为标量 | Server/Patchouli/Alice → 无写入者 | 全系统传播；适合作为 Queue identity snapshot 的基础，但 S0 要求统一来源与授权 |
| `GatewayDecision` / outcome（[gateway.py](../../src/hivememory/core/protocol/gateway.py#L88)） | 公共决策快照；单请求 | deep immutable | tuple + frozen 子模型 | GatewayWorkflow → 无写入者 | System/Patchouli/Passive；边界良好 |
| `CommandExecutionResult`（[gateway.py](../../src/hivememory/core/protocol/gateway.py#L68)） | 公共命令结果；单请求 | deep immutable（JSON-supported） | `FrozenDict[str, Any]`；自定义对象仍不受支持 | CommandDispatcher → 无写入者 | System/HTTP；新增 data 类型需限制为 JSON 值 |
| `ExecutionProgress`（[models.py](../../src/hivememory/agent_runtime/models.py#L23)） | Builder / PCB progress；单 frame | controlled mutable | `text_segments`、`turn_events` list | FrameFactory → AgentRuntime loop | 只应留在 frame 内；当前定位合理 |
| `TurnEvent` / `AgentAction` / `TraceItem`（[interaction.py](../../src/hivememory/core/models/interaction.py#L132)） | Event / Value Object；run 后长期共享 | deep immutable（JSON-supported） | `tool_args` 经 `FrozenDict` 递归冻结 JSON 值 | Agent loop / reducer → 无写入者；变化产出新对象 | Agent、Patchouli、Artifact；`model_copy(update=...)` 是已知转换入口 |
| `TurnRecord`（[interaction.py](../../src/hivememory/core/models/interaction.py#L272)） | 内容真相快照；interaction 生命周期 | deep immutable | tuple 中成员均为 frozen 模型 | Perception → 无写入者 | `LogicalBlock`、Artifact builder、Generation；边界良好 |
| `LogicalBlock`（[topic.py](../../src/hivememory/core/models/topic.py#L47)） | Topic 内不可变 value；直到结算/归档 | deep immutable | 持有 deep immutable `TurnRecord` | Perception → 无写入者 | SemanticBuffer、TopicData、Generation；可安全共享引用 |
| `SemanticBuffer`（[buffer.py](../../src/hivememory/patchouli/memory_library/buffer.py#L18)） | Topic Aggregate；活跃话题期 | controlled mutable | `blocks` list、state/summary/计数 | ShortTerm store → `ShortTermMemoryStore` 命名写方法 | Patchouli 内部；port 仍可返回实体，写权限主要靠模块约定 |
| `TopicData` / `TopicSnapshot`（[topic.py](../../src/hivememory/core/models/topic.py#L100)） | Read Model / Snapshot；单次读取 | deep immutable | `blocks` tuple 共享已冻结 `LogicalBlock` | ShortTerm store → 无写入者 | Gateway、Server、Generation；源 buffer 后续 append 不改变既有 tuple |

### 3.2 Retrieval、Memory、Artifact 与 Profile

| 模型（位置） | 角色 / 生命周期 | 冻结等级 | 嵌套可变字段 | 创建者 → 唯一写入者 | 主要消费者 / 现状 |
|:--|:--|:--|:--|:--|:--|
| `RetrievalRequest` / `QueryFilters`（[models.py](../../src/hivememory/core/protocol/models.py#L80)） | 请求 DTO；一次检索 | mutable | keywords/tags list；filters 可变 | Gateway/MTP → 语义上不应再写，但代码未冻结 | Retrieval；不适合直接作为持久化 work payload |
| `RetrievalResponse` / engine result（[models.py](../../src/hivememory/core/protocol/models.py#L128)） | Read Result；prepare/run 生命周期 | mutable | `memories: list[MemoryAtom]`，直接传播可变实体 | Retrieval → 无明确写入者 | AgentRunContext、cache、hit 记录；缺少 Memory read projection |
| `MemoryAtom` + meta/index/payload/relations（[memory.py](../../src/hivememory/core/models/memory.py#L219)） | Durable Aggregate；长期资产 | mutable | 多层 list/dict/`Any`/子模型全部可变 | Generation/HTTP create → Generation、Lifecycle、Memory service 多点直接写 | Qdrant、retrieval、cache、MTP、Artifact、HTTP；没有唯一写入口或只读投影 |
| `ArtifactRef` / `MemoryEventLog`（[artifact.py](../../src/hivememory/core/models/artifact.py#L24)） | 引用与生命周期记录；随 Memory/Artifact | mutable | event 的 `artifact_refs` list | Artifact builder/Lifecycle → 多调用点 append/替换 | MemoryAtom payload、Artifact 文件；标称 append-only 但模型不冻结 |
| `BaseArtifact` 及四类 Artifact（[artifact.py](../../src/hivememory/core/models/artifact.py#L41)） | Persistence Model / point-in-time record | mutable | list/dict/`Any`、嵌套 mutable ref/snapshot | Artifact builder/store → store 会回填 `content_hash` | 文件系统；“写入后不可变”目前只是文档约定，不是模型保证 |
| `MemoryVersionSnapshot`（[artifact.py](../../src/hivememory/core/models/artifact.py#L146)） | Memory 版本快照 | mutable | `tags: List[str]` | Artifact builder → 无语义写入者 | Version Artifact；快照本身仍可修改，后续需冻结或序列化脱钩测试 |
| `AgentProfile`（[agent.py](../../src/hivememory/core/models/agent.py#L15)） | 配置/权限 read model；跨多个 run 缓存 | mutable（read-mostly） | allowed lists + 惰性 `_verb_set/_tool_set` | MemoryAtom 解析 / fallback 常量 → 惰性缓存方法会写私有 set | Frame、Profile cache、policy；共享引用可能在 run 间传播权限变化 |
| `MemoryResponse` / list response（[memory.py](../../src/hivememory/server/models/memory.py#L17)） | HTTP DTO；单响应 | mutable，但与实体投影脱钩 | tags list | Server mapper → 无后续写入者 | REST；边界投影方向正确，尚未形成递归只读契约 |

### 3.3 PendingAtom、生成任务与 Agent Run

| 模型（位置） | 角色 / 生命周期 | 冻结等级 | 嵌套可变字段 | 创建者 → 唯一写入者 | 主要消费者 / 现状 |
|:--|:--|:--|:--|:--|:--|
| `WriteFocus` / `UpdateFocus` / `RuntimeScope`（[pending.py](../../src/hivememory/core/models/pending.py#L114)） | Value Object；pending/run 生命周期 | deep immutable | 无 | Alice MTP runtime → 无写入者 | PendingAtom、materialize task；边界良好 |
| `PendingAtom`（[pending.py](../../src/hivememory/core/models/pending.py#L223)） | Runtime Aggregate；pending 到 terminal/evict | controlled mutable，但原始引用泄漏 | `settlement` 为 mutable 模型 | PendingAtomRuntime → Runtime 状态机 | `get()` / `all_atoms()` 返回内部引用；调用方可绕过状态机直接写字段 |
| `PendingAtomSnapshot`（[pending.py](../../src/hivememory/core/models/pending.py#L250)） | Read Model；单次查询 | deep immutable | 无 | PendingAtomRuntime → 无写入者 | Resolver/compiler/view；应成为默认查询结果 |
| `PendingAtomMaterializeTask`（[pending.py](../../src/hivememory/core/models/pending.py#L193)） | 跨 Alice/Patchouli task spec | deep immutable | Identity + Focus 均冻结 | PendingAtomRuntime → 无写入者 | finalize/generation；可作为 versioned codec 的业务输入，但自身无 schema_version |
| `PendingAtomSettlement`（[pending.py](../../src/hivememory/core/models/pending.py#L165)） | 跨子系统结果 DTO | mutable | 当前均为标量/enum | Generation → 语义上无后续写入者 | Global event → PendingAtomRuntime；应冻结并增加 settlement/schema version |
| `GenerationContext` / `GenerationRequest`（[models.py](../../src/hivememory/engines/generation/models.py#L97)） | Generation 内部执行 DTO；单 task | mutable | turns list、trace list、`existing_memory: Any` | Coordinator → Generation pipeline | 不可持久化；validator 还会原地回填 identity |
| `TopicMaterializeTask`（[models.py](../../src/hivememory/engines/perception/models.py#L59)） | Perception → Generation task payload | mutable | blocks list | TriggerManager → Coordinator | 结算后跨异步边界；需规范化快照而非继续传播 list |
| `MemoryGenerationTaskSpec`（[memory_tasks.py](../../src/hivememory/patchouli/runtime/memory_tasks.py#L73)） | Patchouli task spec | shallow frozen | `request: GenerationRequest` 可变且含 `Any` | Coordinator → 无语义写入者 | TaskController/未来 Queue；当前不能直接 JSON 持久化 |
| `MemoryGenerationTask`（[memory_tasks.py](../../src/hivememory/patchouli/runtime/memory_tasks.py#L107)） | Domain runtime handle；task 生命周期 | controlled mutable，但 registry 返回原始引用 | `asyncio.Event`、`asyncio.Task`、状态/时间字段 | TaskController → TaskController/registry cancel | System/HTTP 读取后再投影；不可作为 WorkRecord 或持久化模型 |
| wait result / summary（[memory_tasks.py](../../src/hivememory/patchouli/runtime/memory_tasks.py#L142)） | Task 查询快照 | deep immutable | summary 用 tuple 包含 frozen result | TaskController → 无写入者 | Application/HTTP；方向良好 |
| `AgentRunContext`（[models.py](../../src/hivememory/core/protocol/models.py#L148)） | Patchouli → Alice 请求 DTO；单 run | mutable | RetrievalResponse、AgentProfile、generation option 外部另传 | Patchouli → 语义上无后续写入者 | Alice、PreparedAgentRun；携带可变 MemoryAtom/Profile 引用 |
| `ExecutionFrame` / `RunSession`（[models.py](../../src/hivememory/agent_runtime/models.py#L41)） | Runtime State / PCB；单 run/frame | controlled mutable | history/progress/frame registry/call records | Alice orchestration → AgentRuntime/RunExecutor | 合理的请求内可变状态；禁止进入公共 DTO、cache 或持久化 payload |
| `MTPExecutionContext`（[models.py](../../src/hivememory/agent_runtime/models.py#L84)） | 单指令上下文 | shallow frozen | AgentProfile、FrameExecutionPolicy 可变 | Agent loop → 无语义写入者 | MTP runtime；外壳冻结不能隔离权限对象变化 |
| `AgentRunResult`（[models.py](../../src/hivememory/core/protocol/models.py#L185)） | Alice 公共结果；finalize 生命周期 | mutable | `turn_events: list[Any]`、materialize list | Alice → System/Patchouli | 流式链还会 `model_dump/model_validate`；需明确稳定只读结果边界 |
| `InteractionPayload`（[models.py](../../src/hivememory/core/protocol/models.py#L209)） | Active/Passive → Perception 传输包 | mutable | 三个 list；无 schema_version / interaction_id | Patchouli finalize / Passive buffer → Perception | v0.6.1 最高优先级 payload 缺口；不能直接 durable enqueue |
| `SealedTurn`（[outbox.py](../../src/hivememory/system/services/passive/outbox.py#L38)） | Passive outbox item | shallow frozen | 直接持有 mutable InteractionPayload | Passive buffer → 仅 attempts 通过 replace 新建 | 原对象后续修改可改变“已封口”内容；当前不可称不可变 outbox item |
| `StreamPrelude` / `PreparedAgentRun`（[models.py](../../src/hivememory/patchouli/models.py#L14)） | 应用服务 outcome / run lease | shallow frozen | list、dict、mutable AgentRunContext/Profile | Patchouli → System/Alice | frozen 外壳只防字段替换，不防上下文被修改 |
| `ChatGenerationRun`（[control.py](../../src/hivememory/system/runtime/control.py#L46)） | System runtime handle；chat 生命周期 | controlled mutable | `asyncio.Task` + 状态字段 | ChatApplicationService → registry/control methods | 合理的运行时状态；查询需 snapshot，不能进入 Work payload |

## 4. 聚合所有权与投影关系

### 4.1 Turn / Topic

```text
Alice AgentRuntime owns ExecutionFrame + ExecutionProgress (controlled mutable)
  └─ terminal projection → AgentRunResult (currently mutable DTO)
       └─ Patchouli finalize → InteractionPayload (currently mutable DTO)
            └─ Perception constructs TurnRecord + LogicalBlock (deep immutable)
                 └─ ShortTermMemoryStore owns SemanticBuffer (controlled mutable)
                      ├─ full read projection → TopicData (deep immutable)
                      └─ menu projection      → TopicSnapshot (deep immutable)
```

裁定：执行累积器和 Topic 聚合继续受控可变；稳定边界复用 `TurnRecord` / `LogicalBlock` / `TopicData`。v0.6.1 应修复中间的 `AgentRunResult` / `InteractionPayload` 快照边界，不需要深拷贝整个 Topic。

### 4.2 Memory / Retrieval / Artifact

```text
Patchouli owns MemoryAtom semantics
  ├─ Generation/Lifecycle/Memory services mutate MemoryAtom and nested layers
  ├─ MidTerm adapter serializes to/from Qdrant
  ├─ Retrieval returns list[MemoryAtom]
  │    ├─ AgentRunContext keeps the same mutable atoms
  │    └─ KoakumaAtomCache keeps and returns the same mutable atoms
  ├─ HTTP maps selected fields → MemoryResponse (detached DTO)
  └─ Artifact builders snapshot selected fields → mutable Artifact persistence models
```

裁定：Memory 领域缺少 `MemoryReadModel`/versioned write command。Phase I 不决定“受控可变还是版本化不可变”，但确认 Global/local route 和 cache 不应长期暴露可任意修改的 `MemoryAtom` 原始引用。

### 4.3 PendingAtom

```text
PendingAtomRuntime owns PendingAtom state transitions
  ├─ _PendingAtomStore owns entity + intent/canonical indexes
  ├─ snapshot()      → PendingAtomSnapshot (deep immutable)
  ├─ tasks_by_run()  → PendingAtomMaterializeTask (deep immutable)
  └─ get/all_atoms() → PendingAtom raw reference (ownership leak)

Generation publishes PendingAtomSettlement (currently mutable)
  └─ PendingAtomRuntime validates intent and advances state
```

裁定：保留 Runtime 的唯一写入者设计；后续逐步把 raw-reference 查询改成 snapshot/专用命令，settlement 冻结并版本化。

### 4.4 Agent Run

```text
Patchouli prepares AgentRunContext
  └─ Alice creates RunSession + root ExecutionFrame
       ├─ AgentRuntime mutates ExecutionProgress
       ├─ sub-agent CALL adds frame/call records to RunSession
       └─ terminal projection → AgentRunResult
            └─ Patchouli finalize → Interaction + MemoryGenerationTaskSpec
```

裁定：`RunSession` / `ExecutionFrame` / `ExecutionProgress` 是合法的 run-local 可变状态；恢复时需要专用 checkpoint DTO，不能序列化 PCB 本身。公共结果和 Queue task spec 必须与这些运行时对象脱钩。

## 5. 边界承载清单

### 5.1 HTTP public boundary

| 边界 | 当前承载 | 是否传播内部引用 | 结论 |
|:--|:--|:--|:--|
| Memory REST | `MemoryResponse` / `MemoryListResponse` | 否；从 MemoryAtom 逐字段构造 | 投影方向正确；DTO 仍 mutable，后续 Phase V 决定 tuple/frozen 兼容 |
| Topic REST | `TopicSnapshotResponse` | 否；从 TopicSnapshot 再 `model_dump` | 安全；存在重复 DTO 层但不泄漏实体 |
| Memory Task REST | `MemoryTaskResponse.from_domain` | 否；从 mutable task handle 构造标量结果 | 安全；task registry 本身仍返回原始句柄给应用层 |
| Chat SSE | dict + server event models | 通过 `model_dump` 形成传输值 | HTTP 侧脱钩；Alice/System 内部仍传 mutable AgentRunResult |
| Passive REST | request model → `PassiveIngressEvent`；response model | Pydantic 重构，但模型本身 mutable | 传输值可控；入队前仍需 versioned snapshot |

### 5.2 GlobalSystemBus public routes

| 路由族 | 主要模型 | 边界评级 | 说明 |
|:--|:--|:--|:--|
| Gateway process | frozen Gateway result | 良好 | 公共决策已投影且依赖中立 |
| Topic list/get | tuple[`TopicSnapshot`] / `TopicData` | 良好 | 明确只读契约，Gateway 还做类型检查 |
| Memory create/list/get/update/retrieve | `MemoryAtom` / list[`MemoryAtom`] / `RetrievalResponse` | 高风险 | 跨 System/Patchouli/Alice 传播 mutable aggregate 原始引用 |
| Patchouli prepare/finalize | `PreparedAgentRun`、`AgentRunResult`、list[`MemoryGenerationTask`] | 高风险 | frozen 外壳或 mutable handle；没有统一 snapshot 边界 |
| Alice run | `AgentRunContext` → `AgentRunResult` | 中高风险 | 请求/结果均可变且包含 Memory/Profile 引用 |
| Memory task list/get | mutable `MemoryGenerationTask` | 中风险 | HTTP 会投影，但其他 Global route 调用方可修改句柄 |

### 5.3 Subsystem local routes

| 子系统 | 主要载荷 | 问题 |
|:--|:--|:--|
| Patchouli ingestion | `InteractionPayload` | list 可变、无 interaction/schema version；是 Queue Q2 的直接前置缺口 |
| Patchouli retrieval | `RetrievalRequest/Response`、`MemoryAtom` | 读结果和 cache 共享 mutable atom |
| Patchouli generation | `TopicMaterializeTask`、`GenerationRequest/Outcome`、`MemoryGenerationTaskSpec` | list/`Any`/实体引用与 runtime handle 混合，不能直接持久化 |
| Patchouli task control | `MemoryGenerationTask`、wait snapshot | 写侧 handle 可变；wait snapshot 良好 |
| Alice MTP/events | Focus、MaterializeTask、Settlement | Focus/Task 良好；Settlement 未冻结/未版本化 |

### 5.4 RuntimeEvent 与 event bus

| 边界 | 当前承载 | 结论 |
|:--|:--|:--|
| `RuntimeEvent` | mutable Pydantic + `data: dict[str, Any]` | ring buffer、replay 和 subscriber 共享 event 引用；`model_copy(update=...)` 是浅拷贝，嵌套 data 可继续变化 |
| Runtime publisher | 对 BaseModel 调 `model_dump(mode="json")` | 经过发布器的 payload 多数已转为值；直接构造 RuntimeEvent 的调用点仍可放任意对象 |
| Pending settlement event | mutable `PendingAtomSettlement` | Global/local event 直接传对象；订阅者失败隔离，但没有快照/版本保证 |
| Patchouli task/passive event | 以标量 kwargs 或新 dict 为主 | 风险较低；仍需维持“不记录正文”的数据策略 |

RuntimeEvent 是观测事实，不是业务真相；即使后续冻结，也不能替代 WorkRecord、operation result 或领域状态。

### 5.5 Cache、registry 与 task boundary

| 容器 / 工作项 | 持有对象 | 返回行为 | 所有权风险 |
|:--|:--|:--|:--|
| `KoakumaAtomCache` | mutable `MemoryAtom` | 返回同一原始引用 | 调用方可改 cache；同时存在 S0 的跨身份问题 |
| `_PendingAtomStore` | mutable `PendingAtom` | `get/all_atoms` 返回原始引用 | 可绕过 Runtime 状态机 |
| `AgentProfileCache` | mutable `AgentProfile` | 返回共享 profile 引用 | 私有 lazy set 和白名单 list 可跨 run 变化 |
| `RuntimeEventBus` | mutable `RuntimeEvent` | replay/subscriber 收到同一对象 | 观测历史可被后续引用修改 |
| `MemoryGenerationTaskRegistry` | mutable task + asyncio handle | list/get 返回原始引用 | 应由 WorkRecord snapshot 替代执行真相源 |
| `ChatGenerationRunRegistry` | mutable run + active asyncio task | get 返回原始引用 | 仅应由 Chat control 使用；未来查询需独立 snapshot |
| `SealedTurn` | frozen 外壳 + mutable InteractionPayload | outbox 持同一 payload 引用 | seal 后内容仍可能变化 |
| `MemoryGenerationTaskSpec` | frozen 外壳 + mutable GenerationRequest | runner 直接持引用 | enqueue 后外部修改可改变执行输入 |
| `MaintenanceTaskSpec/TaskRuntimeState` | mutable spec + callback + asyncio task | scheduler 内部持有 | 合法 scheduler runtime；不能升级为可持久化 WorkItem |

## 6. 复制性能基线

### 6.1 环境与方法

- 日期：2026-08-07；
- Python 3.12.9；Pydantic 2.10.3；Windows 11 10.0.26200；
- 每项预热一次，执行 7 次并报告中位数；
- 峰值内存使用 `tracemalloc` 对单次操作测量；
- 合成数据不访问 Qdrant、模型服务或文件持久化；
- 可复现入口：[scripts/benchmark_data_model_phase_i.py](../../scripts/benchmark_data_model_phase_i.py)。

结果是当前开发机上的方向性基线，不是跨硬件性能承诺，也暂不作为 CI 阈值。

### 6.2 结果

| 形状 | 操作 | 中位耗时 ms | 峰值 MiB |
|:--|:--|--:|--:|
| Large Topic | 当前 `TopicData` 投影（1,000 blocks） | 0.077 | 0.018 |
| Large Topic | 完整 deep copy（1,000 blocks） | 39.866 | 5.930 |
| Memory list | HTTP 投影（1,000 atoms） | 5.388 | 1.365 |
| Memory list | 完整 deep copy（1,000 atoms） | 59.968 | 5.291 |
| TurnEvent stream | `TurnRecord` 投影（10,000 events） | 0.373 | 0.078 |
| TurnEvent stream | 完整 deep copy（10,000 events） | 75.127 | 11.188 |
| Queue candidate | `InteractionPayload` JSON encode（2,000 events） | 3.343 | 0.921 |
| Queue candidate | `InteractionPayload` JSON decode（2,000 events） | 17.353 | 3.314 |

### 6.3 解读

1. TopicData 和 TurnRecord 的当前投影非常便宜，因为 tuple 只复制容器并复用已经冻结的 `LogicalBlock` / `TurnEvent`；这是应该保留的模式。
2. 对整个 Topic/Turn 对象图做完整 deep copy 分别增加约 40ms/75ms 和 6MiB/11MiB 峰值，不适合作为每次读取或每个事件发布的默认策略。
3. Memory HTTP 投影比完整 deep copy 低一个数量级，并且只暴露所需字段；后续应增强该投影的只读性，而不是把完整 MemoryAtom 深拷贝后继续出境。
4. 2,000 事件的 JSON 编解码成本在单机队列可接受范围内，但 decode 明显更昂贵；Work Queue 必须设置 payload 大小、单 work 事件数和 lane capacity，不能把无限长事件流当作普通 work item。
5. 基准支持“冻结叶子值 + 共享不可变引用 + 边界窄投影 + 持久化时规范化序列化”，不支持“所有模型一律 deep-copy/frozen”的治理方向。

## 7. v0.6.1 Queue admission 清单

在 Queue Q0/Q1 实现前，下列边界必须冻结：

1. **WorkItem / Work payload**
   - `kind`、`schema_version`、operation identity、identity snapshot、ordering key 和 payload bytes/value 必须稳定；
   - WorkItem 本身 deep immutable；WorkRecord 由 Store 单一所有者 controlled mutable；查询只返回 WorkRecord snapshot；
   - payload 只接受可规范化序列化的数据，不接受 coroutine、lock、event loop、service、`asyncio.Task` 或任意领域实体引用。
2. **Interaction Submission**
   - 不把现有 `InteractionPayload` 或 `SealedTurn` 直接写入 WorkStore；
   - 新建 versioned submission envelope/codec，入队时把 list 规范化为稳定快照并包含 `interaction_id` 与 Identity scope；
   - handler decode 后再构造当前 Perception 所需 DTO，外部对象后续变化不得影响已接受 work。
3. **Memory Generation**
   - 不持久化 `MemoryGenerationTask`、`GenerationRequest.existing_memory: Any` 或 `asyncio.Event/Task`；
   - 把 task spec 拆成可序列化、版本化的业务输入和进程内 handler context；MemoryAtom 只传稳定 ref/id/version 或必要快照；
   - `WorkRecord` 是执行状态真相源，`MemoryGenerationTask` 只做兼容领域投影。
4. **RuntimeEvent**
   - 事件 payload 必须在 emit 时转成安全值快照；不得把 WorkRecord、MemoryAtom、InteractionPayload 或异常对象原始引用放进 `data`；
   - 冻结事件不会改变其 best-effort 观测定位。
5. **Identity 与 cache**
   - Work payload 从第一版携带不可变 identity snapshot；handler 在实际所有者处重验；
   - cache key/record 查询必须包含 S0 定义的 scope，不能因采用 frozen DTO 而省略授权检查。

## 8. 后续治理优先级

### P0：v0.6.1 Queue 前置

1. 建立 versioned `InteractionSubmission` codec，消除 `SealedTurn -> InteractionPayload` 的浅冻结假象；
2. 建立 WorkItem/WorkRecord/WorkRecordSnapshot 的角色与冻结规则；
3. 把 Memory generation payload 中的 `GenerationRequest`、`Any` 和运行时句柄拆出持久化 spec；
4. 为 enqueue-after-mutation、unknown schema、snapshot round-trip 和 identity scope 增加测试。

### P1：Queue 接线同步处理

1. RuntimeEvent emit 时规范化 data，防止 ring buffer 历史被原始引用修改；
2. PendingAtom 对外查询优先返回 snapshot，raw entity 只留在 Runtime 私有路径；
3. Memory task list/get 逐步返回领域 snapshot，由 Controller/WorkStore 保持唯一状态真相；
4. AgentRunResult / PreparedAgentRun 明确只读 projection 或严格限制为单请求句柄。

### P2：后续聚合治理

1. 裁定 MemoryAtom 为受控可变聚合还是版本化不可变聚合，并收敛 write command/CAS；
2. 建立 Memory read model，Global route、retrieval 和 cache 不再传播可变实体；
3. 冻结 Artifact point-in-time record，并明确 Store 回填 hash 的构造阶段；
4. AgentProfile 改为稳定权限 snapshot，cache 使用版本/失效机制；
5. 统一 HTTP/public DTO 的递归只读规则并验证 JSON 兼容。

## 9. Phase I 完成判据

- 六条主要聚合链均已登记角色、生命周期、冻结等级、所有者与消费者；
- HTTP、Global/local route、RuntimeEvent、cache/registry 和 task 边界均有承载清单；
- 已区分真正的 deep immutable、受控可变与 frozen 外壳；
- 已给出大 Topic、Memory 列表、长事件流和候选 Queue payload 的可复现性能基线；
- 已形成 v0.6.1 Queue admission 清单和后续 P0/P1/P2 优先级；
- 本文只冻结现状与进入条件，不宣称 Phase II-VI 已实现。
