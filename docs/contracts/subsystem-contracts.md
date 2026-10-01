---
title: Subsystem Contracts
status: current
owner: system
scope: subsystem-public-contracts
code_paths:
  - src/hivememory/core/contracts/
  - src/hivememory/gateway/contracts/
  - src/hivememory/patchouli/contracts/
  - src/hivememory/alice/contracts/
  - src/hivememory/workspace/contracts/
  - src/hivememory/core/protocol/
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-10-01
---

# 子系统公共契约

本文定义 System、Gateway、Patchouli、Alice 跨边界可观察的输入、输出和不变量，以及任务进程与执行者之间的 CPU 端口。路由字符串与事件名的完整清单见[routes-and-events.md](./routes-and-events.md)。

契约不是把公开函数逐一抄进文档，也不是要求每个子系统共享同一套内部对象。它描述的是一次能力交接：调用方必须提供哪些事实，所有者承诺返回什么，以及双方都不能偷偷改变哪些语义。只要这些交接保持稳定，Gateway 的分析流程、Patchouli 的记忆实现和 Alice 的执行循环就可以各自演进；一旦内部 workflow state 或引擎实体越过边界，局部重构便会重新变成全系统改造。

因此，公共模型倾向于使用 frozen、Pydantic 或依赖中立的 dataclass。不可变并不只是编码偏好，它要求上游先形成完整决定，再交给下游只读消费；依赖中立则阻止某个领域对象沿模型引用把存储、Runtime 或 Controller 一并泄漏出去。本文既记录字段和终态，也记录这些形态背后的所有权理由。

跨边界的身份坐标统一使用 `IdentityScope`：它同时冻结 actor 与 Workspace 归属，是一次请求、交互或后台任务的唯一身份来源。领域所有者在最终读写处校验 scope；共享的 queue、registry 和 Runtime 不因此按 Workspace 复制或分区。完整的资源归属模型见[Workspace 架构](../architecture/workspace.md)，本文只记录各契约需要携带和验证的部分。

## 1. 生命周期契约

标准子系统实现 `SubsystemProtocol`：

```python
name: str
async start() -> None
async stop() -> None
async health() -> dict[str, Any]
```

当前 Gateway、Patchouli、Alice 均满足此契约。`start()` 挂载 local/public route，`stop()` 撤销 route 并释放自己拥有的运行时资源；重复启停的具体幂等能力由宿主实现保证，调用方不应绕过 `HiveMemorySystem` 随意改变单个子系统状态。

统一生命周期的目的，是让路由是否可用与资源是否已准备好保持同一顺序。如果调用方分别启动 Runtime、注册 route 或释放存储资源，就可能出现“路由仍在但所有者已停止”或“依赖尚未就绪却已接受请求”的半启动状态。因此启停属于 System 的组合职责，领域对象不自行组织全局生命周期。

## 2. Gateway 契约

### 2.1 Process

```python
process(
    message: str,
    *,
    identity_scope: IdentityScope,
    ingress_mode: GatewayIngressMode,
    request_timeout_ms: int | None = None,
) -> GatewayProcessResult
```

`GatewayProcessResult` 是不可变判别联合：

- `GatewayCommandOutcome(kind="command")`：包含 `CommandExecutionResult`；
- `GatewayDecisionOutcome(kind="decision")`：包含 `GatewayDecision`。

二者互斥。命令终态不能同时携带普通分析结果；普通决策不能携带命令执行结果。

这种互斥使命令成为真正的短路终态。系统指令已经完成、被拒绝或要求确认时，继续执行检索、Agent run 和记忆提交既浪费资源，也可能把一条控制消息误当成普通对话沉淀。判别联合让调用方必须显式选择一条链路，不能依赖多个可空字段猜测 Gateway 的意图。

### 2.2 GatewayDecision

| 字段 | 语义 |
|:---|:---|
| `target_topic_id` | 已有 topic id 或 `NEW_TOPIC` |
| `new_topic_title` / `new_topic_summary` | 新话题的可选初始元数据 |
| `rewritten_query` | 下游检索使用的完整查询 |
| `search_keywords` | 稀疏检索关键词元组 |
| `memory_write_signal` | `WRITE`、`SKIP` 或 `UNKNOWN` |
| `retrieval_plan` | 检索模式、`top_k` 和 dense/sparse 权重 |
| `intent_type` | `RAG`、`WRITE`、`CHAT`、`COMPOSITE` 或 `UNKNOWN` |

`worth_saving` 是从 `memory_write_signal` 派生的只读值，不是第二份状态。

`GatewayDecision` 只保留下游可以稳定依赖的决策结果，而不携带 step、snapshot 或分析器内部对象。它是 Gateway 与执行链之间的交接单，不是远程操纵 Gateway workflow 的句柄。

### 2.3 模式不变量

- `ACTIVE_CHAT` 可以识别并执行系统指令；
- `PASSIVE_MEMORY` 必须返回普通决策，绝不能返回 command outcome；
- `request_timeout_ms` 只能收紧配置的默认总超时，不能扩大它；
- 局部可恢复失败可以降级，但最终结果仍必须满足完整终态不变量。

## 3. Patchouli 契约

Patchouli 的公开面分为 chat 协作、记忆、任务、Agent Profile、话题与就绪状态。调用方不能直接调用 Patchouli local route。

这些能力虽然服务于不同用例，却共享一个核心约束：Patchouli 对长期状态和身份可见性拥有最终解释权。公开契约允许外部请求“创建、检索或提交”，但不会把 MemoryLibrary、Familiar 或生成 Controller 交给调用方直接操作。

### 3.1 PrepareAgentRun

```python
prepare_agent_run(
    user_message: str,
    *,
    identity_scope: IdentityScope,
    interaction_id: str,
    gateway_decision: GatewayDecision,
    enable_memory_retrieval: bool = True,
) -> PreparedAgentRun
```

prepare 只做 Patchouli 自己的两件事：按 Gateway 的路由决定准备本轮 Topic（必要时新建，话题池已满时先按 LRU 结算一个已有话题），并按检索计划检索记忆。它不解析 Agent Profile、不接触附件、不编译记忆，也不为任何执行者组装运行上下文；这些属于任务进程在分配 CPU 时的工作（第 5 节）。

`PreparedAgentRun`（`patchouli.contracts.prepare`）是不可变 dataclass，既是 prepare 的结果，也是 finalize 与 cleanup 的输入句柄：

| 字段 | 内容 |
|:---|:---|
| `identity_scope`、`interaction_id`、`user_message`、`gateway_decision` | 由 prepare 入参冻结；`IdentityScope` 是唯一身份来源 |
| `topic_id`、`is_new_topic` | 本轮真实话题与是否由 prepare 新建 |
| `topic_context`、`pool_topics` | 话题上下文与话题池快照 |
| `retrieval_result` | 未编译的检索结果（`RetrievalResponse`） |
| `storage_available` | 记忆存储健康状态 |

它位于 Patchouli 的 `contracts` 子包，因为 workspace 的任务进程需要读取其中的话题与检索结果；L3 子系统之间只能导入对方的 `contracts`。

prepare 的意义在于：由 Patchouli 在交出控制权前确认真实话题与本轮可见的记忆，调用方无需理解 Patchouli 内部存储。检索结果以原始 `MemoryAtom` 交出，如何呈现给执行者由任务进程决定。

### 3.2 FinalizeAgentRun

```python
finalize_agent_run(
    prepared_run: PreparedAgentRun,
    payload: InteractionPayload,
) -> list[MemoryGenerationTask]
```

`payload` 是任务进程封口的本轮交互记录。进程在 Actor 正常完成、进入 finalize 之后组装它（`workspace/process/sealing.py`），材料全部来自进程自身：

| 字段 | 来源 |
|:---|:---|
| `user_message` | 任务请求的入口消息 |
| `rewritten_query`、`worth_saving` | Gateway 阶段的决定 |
| `assistant_final_text`、`turn_events`、`model_used`、`materialize_tasks` | Actor 的执行结果 |
| `mtp_traces` | 由 core 的 `ActionReducer` / `TraceReducer` 从 `turn_events` 归约 |
| `used_attachments` | 附件编译确认实际进入上下文的附件引用快照 |

附件租借由任务进程持有并在进程结束时释放，finalize 不负责。

Finalize：

1. 将 payload 原样提交到目标话题，不改写其内容；
2. 为 payload 中 WRITE/UPDATE 形成的 materialize task 启动主动记忆生成；
3. 记录预检索命中。

任务进程只对 `status == completed` 的执行结果封口并调用 finalize。Finalize 已成功后不能再 cleanup。

finalize 是执行事务与记忆事务的分界。交互记录由提交方封口：主动链路由任务进程封口，被动链路由 System 的 turn buffer 封口，两条链路的封口位置一致；Patchouli 的公开路由因此不必读懂任何执行者的运行结果，换一个 CPU 也不需要 Patchouli 随之改变。轨迹归约规则只有 core 中的一份，封口方调用它，不会形成第二套规则。Patchouli 负责判断已封口的交互如何进入长期知识（提交、感知、生成与 lifecycle）；只有 completed 的一轮才封口提交，取消和失败的半完成 run 不会默认进入长期知识。

### 3.3 CleanupPreparedAgentRun

```python
cleanup_prepared_agent_run(prepared_run: PreparedAgentRun) -> bool
```

Cleanup 只尝试删除 prepare 阶段新建但仍为空的话题，不负责附件租借。已有话题或已经产生内容的话题不应被删除。调用方把 cleanup 当作失败补偿，不把返回 `False` 视为新的业务错误。

它不是 rollback，也不承诺撤销整个 prepare 之后发生的一切。跨子系统没有一项可以原子回滚的数据库事务；cleanup 只补偿明确由 prepare 创建、且仍可安全判断为空的临时副作用。将它描述为回滚会诱使调用方删除已经存在或已被其他流程使用的长期状态。

### 3.4 其他公开能力

| 能力组 | 当前公开行为 |
|:---|:---|
| Interaction | 提交 `InteractionPayload` 到指定或新话题；进入 submission lane 时由 `InteractionSubmission.identity_scope` 携带作用域 |
| Memory | 携带 `IdentityScope` 的 create/list/get/update/delete、feedback、retrieve、retrieve_by_aliases；create/update 在提交边界生成完整版本记录（无历史不成功），无变化的编辑不创建版本 |
| Memory Task | list/get/cancel |
| Agent Profile | 携带 `IdentityScope` 的 create/list/get |
| Topic | 携带 `IdentityScope` 的 list active、topic data、manual settle、evict；Patchouli owner 拒绝越域 topic |
| Citation | 记录 MTP READ/RUN 等来源的记忆引用 |
| Readiness | 模型 warmup 与 ready 查询 |

Memory 与 Topic 的 Workspace 归属和 actor 可见性由 Patchouli 执行，调用方不能仅凭拿到 id 就假设目标可见；以上能力在携带访问上下文时按第 3.5 节执行行为授权，任务归属按权威 `IdentityScope` 投影判断，知道任务 ID 不构成权限。Topic ID 在领域上保持全局唯一；`IdentityScope` 用于确认访问归属，不构造另一套局部 ID 命名空间。

### 3.5 访问上下文与行为授权

公共 application 方法约定接收 `access: WorkspaceAccessContext | None` 参数：context 由 workspace 认证入口（统一认证网关，`workspace.authentication`）签发、由 workspace 访问基础设施逐次校验，Patchouli 经 `core.access.WorkspaceAccessVerifier` 消费这一检查。提供 access 时，application 在资源读取或副作用之前按方法绑定的 operation 调用共享行为检查，取得可信 `IdentityScope` 后才进入领域链；读取类 backing 路由（`memory.read`、`memory.retrieve`、`memory.retrieve_by_aliases`、`get_agent_profile`）例外：operation 授权由 workspace 能力层在调用前执行，Patchouli 一侧只校验 context 的签发、有效期与准入，不重复检查 operation。请求 DTO 中携带的 scope 只能作一致性校验，不得覆盖可信坐标。`WorkspaceAccessContext` 只公开已准入的 `IdentityScope`，不携带调用来源、行为白名单或单次 operation；同一有效 context 可先后执行不同的获准操作。

两类入口并存是显式契约而非疏漏：`read_memory`、`interaction.submit`、`memory_intent.submit` 等不在迁移兼容清单内，缺失 access 一律拒绝；管理 CRUD、检索、Profile、Topic 管理和附件上传等既有调用方在缺失 access 时按裸 scope 受信适配运行，清单（保留入口、已有调用方、A6 删除点）唯一维护在 `patchouli/application/access_consumption.py`，A6 完成生产消费者切换后删除兼容分支。Patchouli 提交与生成链沿用自身既有来源记录，公开 API 不接收 `CallerPrincipal` 或其他来源字段。阶段拒绝语义（接入认证、准入、行为授权、context 有效性）见[错误模型](./error-model.md)，完整访问模型见[Workspace 架构](../architecture/workspace.md)第 4 节。

## 4. CPU 端口与 Alice 实现

### 4.1 CPU 端口

```python
class CPUPort(Protocol):
    def execute(
        self,
        manifest: CPUInputManifest,
        *,
        generation_options: dict[str, Any] | None,
        stream: bool,
    ) -> AsyncGenerator[CPUOutput, None]: ...
```

CPU 端口（`workspace.contracts`）是任务进程调用执行者的唯一接口：端口由 workspace 定义，执行者实现，组合根注入 `TaskProcessService`。进程只依赖端口与本节的中立模型，因此执行者可以替换而不改动进程与入口；当前唯一的实现是 Alice 的 `AliceCPU`（4.4），测试中的 `ScriptedCPU` 同样能跑完整个任务进程。端口采用对象而不是总线路由，是因为外部 harness 的驱动多数不是子系统：按路由契约接入，每种驱动都要新增路由常量，或在总线之后再建一层分派。

`CPUInputManifest` 是任务进程在分配 CPU 时组装的输入清单，与具体执行者无关：`process_id`、`identity_scope`、用户消息、已解析的 Agent Profile、未编译的检索原子 `memories`、进程编译的记忆文本 `memory_context` 与附件文本 `attachment_context`、存储可用性，以及 `topic_id` 与 `topic_context`。

端口语义：

- **输出顺序**：`execute` 返回的异步生成器先产出交互事件（只在流式时），再产出唯一的终态结果 `CPUExecutionResult`；终态结果恰好出现一次，并且是最后一项，非流式时它是唯一一项。输出流在没有终态结果时结束属于协议错误，进程按失败处理。
- **交互事件**：带 `event` 与 `data` 的字典，进程原样转交给流式交付，不做解释。
- **结局**：执行者自报的结局经终态结果的 `status` 表达；执行者抛出异常，由进程按失败处理。
- **取消与关闭**：用户停止时，进程取消正在拉取下一项的任务；进程在拿到终态结果后、以及在关闭流程中关闭输出流。实现必须传播 `asyncio.CancelledError`，并在输出流关闭时释放自己创建的资源。
- **`generation_options`**：由各个执行者自行解释，进程原样传递。

### 4.2 执行结果

`CPUExecutionResult` 是执行者对一次执行的完整事实声明，而不是已经提交的长期记忆：

- `status`：`completed`、`cancelled` 或 `failed`（`CPUExecutionStatus`）；
- `final_text`：最终用户可见文本；
- `turn_events`：结构化运行事实（`TurnEvent`）；
- `model_used`：执行者实际使用的模型展示名，空字符串表示未解析；
- `materialize_tasks`：本次执行产生的不可变物化请求；写入意图的实时派发实现之前保留。

执行者专属的统计（例如 Alice 的 MTP 迭代次数）不进入执行结果，只出现在各自的观测事件中。任务进程据执行结果决定是否进入 finalize，并从中封口交互记录；任何一方都不能仅凭流中的部分文本推断执行已经完成。

### 4.3 Alice 的执行入口

```python
run_agent(
    input_manifest: CPUInputManifest,
    generation_options: dict[str, Any] | None = None,
    *,
    stream: bool = True,
) -> AsyncGenerator[dict[str, Any], None] | Coroutine[Any, Any, CPUExecutionResult]
```

`AgentRunService.run_agent` 是 Alice 唯一的执行入口，经全局路由 `alice.public.run_agent` 暴露。流式与非流式运行同一套执行骨架，`stream` 只决定是否产出交互事件：流式时返回事件的异步生成器，最后一项是 `done`（执行结果的字段加上运行元数据）；非流式时返回可 await 的 `CPUExecutionResult`。消费者关闭事件流时，Alice 取消并 join 自己创建的 runner。Alice 在内部把输入清单转换为提示词组装使用的 `AgentRunContext`；`AgentRunContext` 不出现在任何公开路由上。

### 4.4 Alice 的端口实现

`AliceCPU`（`alice/application/cpu.py`）经全局总线调用 `alice.public.run_agent`：流式时原样转交交互事件，把 `done` 转换为 `CPUExecutionResult`（运行元数据不进入结果）；非流式时产出路由返回的执行结果。端口输出流被关闭时，它一并关闭 Alice 的事件流。Alice 的运行时仍在公开路由之后，进程只持有端口对象。

### 4.5 Alice 不变量

- Alice 不修改输入清单所引用的长期记忆或话题；
- WRITE/UPDATE 只产生 PendingAtom 和 materialize task；
- 取消或失败结果不默认进入 Patchouli finalize；
- MTP 权限由 Agent Profile 的 `allowed_mtp_verbs` 与 `allowed_sys_tools` 控制；
- CALL 仅允许根 frame 发起，子 frame 不能继续递归 CALL。

## 5. 顶层主动链路契约

```text
Gateway command outcome
  -> System 返回命令结果
  -> 不调用 Patchouli prepare / CPU / Patchouli finalize

Gateway decision outcome
  -> 解析 Agent Profile（Patchouli 公开路由）
  -> Patchouli prepare（Topic 与检索）
  -> CPU 分配：附件租借、附件与记忆编译、组装输入清单
  -> Actor 执行：经 CPU 端口（当前为 Alice）
  -> completed: 任务进程封口交互记录（InteractionPayload，含实际使用的附件）
       -> Patchouli finalize
  -> cancelled/failed/exception: Patchouli cleanup (若已 prepare)
  -> 进程结束：释放附件租借
```

Agent Profile 属于 CPU 分配，但当前在 prepare 之前解析：prepare 可能新建 Topic 或按 LRU 结算已有话题，Profile 缺失的请求应在这些副作用发生前失败。

该顺序由 `TaskProcessService`（`workspace.process`）拥有。任何 transport adapter 都不能复制或调整此顺序。

顺序本身就是契约的一部分：Gateway 先收敛入口语义，Patchouli 再准备长期知识的本轮视图，CPU 只执行，最后由 Patchouli 提交。让 transport adapter 复制这条链路，会很快产生“HTTP 可以、其他入口不可以”或两条 finalize 规则不一致的问题。

## 6. 被动链路契约

Passive Ingress 由 System 拥有并调用 Gateway `PASSIVE_MEMORY`。它可以读取 Patchouli 记忆上下文并最终提交 interaction，但不调用 Alice、不执行命令、不运行 MTP，也不生成 assistant reply。

对外 `PassiveIngressOutcome` 只表达 accepted/buffered/duplicate/degraded 等业务结果；Gateway execution state、fallback 原因和 RuntimeEvent 不进入 API 响应。

同一 `PassiveConversationKey` 在单进程内按服务接收顺序串行处理，串行范围包含 Gateway/retrieval、accumulator 修改与 submission queue admission；不同会话仍可并发。admission 成功前 accumulator 不会清空，也不会被下一 user 覆盖。connector 负责按会话因果顺序投递，`sequence` 当前只用于关联和观测，不承诺对已经乱序到达的事件进行重排。该契约不扩展为跨进程排序或持久化 mailbox。

这条限制保护的是入口语义。Passive Memory 用于摄入已经发生的外部经历，并不等价于伪造一次用户与 Agent 的对话；如果它允许命令或 Alice 执行，外部内容便可能意外触发控制行为、工具调用和回复生成，也会让“谁发起了这次行动”失去可靠答案。

## 7. Interaction 与 Topic 时序契约

Active 与 Passive 的消息来源和入口流程不同，但二者最终都向 Topic 追加 Interaction，因此共享同一组时序职责：

`InteractionSubmission` 是进入 Patchouli submission lane 的稳定交接包：`identity_scope` 是唯一身份来源，`interaction_id` 负责幂等关联，`InteractionPayload` 只承载本轮内容和物化请求，不重复嵌入 scope。`TopicAssetBinding` 只有在该 Interaction 成功应用且用户明确使用 asset ref 时才成立；上传或 UI 选择不会单独产生 binding。

`InteractionPayload.used_attachments` 携带任务进程的 `AttachmentCompiler` 确认实际进入上下文的 bound ref 快照（由任务进程封口时写入）：submission handler 把这份快照以 `asset_refs` 形参一次性传给 `apply_interaction`，不回查原始选择、asset 列表或当前 UI 状态。retry 重放同一份快照，不能替换 ref。

1. **Interaction 内全序由生产者冻结。** `TurnEvent.sequence` 只在所属 interaction 内有效；payload 一旦进入 submission queue，retry、dedup、cleanup 和 handler 都不得改写既有事件顺序或生成新的语义身份。
2. **Topic append 顺序由 Patchouli 拥有。** 当前以成功 apply 的实际 append 顺序作为 topic-local 权威投影。若未来增加 `topic_position`，必须由 topic owner 在持久化提交时原子分配，调用方不能根据时间戳自行计算。
3. **Queue FIFO 只是执行约束。** ordering key 只串行化已经入队的 work，不代表源事件发生时间，也不表达 Agent 因果关系。idempotency journal 只防止重复副作用，不参与排序。
4. **Passive source sequence 不负责事后重排。** connector 应按会话因果顺序投递；源 `sequence` 当前用于关联和观测，不承诺缓存、等待或重排晚到消息。
5. **Active prepare 读取的是一个 topic snapshot。** finalize/apply 的先后位置只能证明提交顺序，不能证明某个 run 在生成 LLM input 时看见过中间提交。多个 run 可以基于同一 revision 并发执行。
6. **多 Agent 并发优先表达因果偏序。** 未来应保存 `base_topic_revision`、causal parent 或等价的 run/frame 关系，再由 prompt/display 层按需要确定性线性化；不得仅按完成时间伪造语义先后。

`occurred_at`、`received_at`、`enqueued_at` 与 `applied_at` 是不同阶段的观测时间，均不能替代 topic 内的权威顺序。当前契约要求的是 topic-local authoritative log 与可扩展的 causal relation，而不是全系统绝对时间线；本阶段不要求全局序列、向量时钟或完整事件溯源。

## 8. 契约矛盾检查

新增或修改公共能力时，应先回答以下问题：

1. 这个模型是否只包含一次交接需要的稳定事实，还是暴露了可变 workflow state、引擎对象或回调？
2. 接收方是否正在修改本应只读的决定，或重新推导一份与所有者可能分叉的状态？
3. command outcome 是否仍能立即短路？Passive Memory 是否可能通过新分支触发命令、Alice、MTP 或回复生成？
4. prepare、run、finalize 的顺序或资格是否被 transport、事件订阅者或兼容 fallback 悄悄改变？
5. cleanup 是否仍是对空话题的有限补偿，还是被当成可以撤销长期状态的事务回滚？
6. 执行结果、PendingAtom ACK 或流式片段是否被误认为 finalize 已成功？
7. 身份、可见性和权限检查是否仍由状态所有者执行，而不是由拿到 id 的调用方自行假设？
8. 是否把 enqueue/apply timestamp 或 queue FIFO 误当作业务发生顺序？
9. 是否把 topic append 顺序误当作 Agent 已观察到彼此结果的因果关系？
10. 是否声称 finalize ordering 已经解决 prepare/LLM input snapshot 的并发？
11. Patchouli 公开路由是否开始接收某个执行者专属的运行结果，或在 finalize 中改写提交方已封口的交互记录？

这些问题能帮助评审者从契约语义发现设计分叉，而不只是检查函数签名是否还能调用。

## 9. 兼容与变更

以下变化属于跨子系统破坏性变更，必须同步修改 route 常量、公共模型、调用方、契约测试和本文：

- 修改 route 字符串或 handler 参数；
- 修改判别联合的 `kind`；
- 新增必填公共字段或改变字段语义；
- 改变 prepare/finalize/cleanup 顺序；
- 允许 Passive Memory 返回命令；
- 改变 CPU 端口的输出语义、执行结果终态和 finalize 资格；
- 将 local route 或内部 workflow state 暴露为公共 API。

验证入口：`tests/unit/system/contracts/`、`tests/unit/workspace/process/`、`tests/unit/system/application/`、`tests/unit/gateway/test_phase3b_contracts.py`、`tests/unit/patchouli/test_phase3f_gateway_decision.py`、`tests/unit/alice/application/test_agent_run_service.py`、`tests/unit/alice/application/test_alice_cpu.py`、`tests/unit/workspace/process/test_cpu_port.py`、`tests/unit/patchouli/application/`、`tests/integration/workspace/test_application_access_boundary.py`。
