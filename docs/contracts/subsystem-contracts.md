---
title: Subsystem Contracts
status: current
owner: system
scope: subsystem-public-contracts
code_paths:
  - src/hivememory/core/contracts/
  - src/hivememory/gateway/contracts/
  - src/hivememory/patchouli/contracts/
  - src/hivememory/alice/interface/cpu.py
  - src/hivememory/workspace/contracts/
  - src/hivememory/core/protocol/
  - src/hivememory/system/services/import_bus/
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/README.md
  - docs/system/import-bus.md
last_reviewed: 2026-10-30
---

# 子系统公共契约

本文定义 System、Gateway、Patchouli、Alice 跨边界可观察的输入、输出和不变量，以及任务进程与执行者之间的 CPU 端口。路由字符串与事件名的完整清单见[routes-and-events.md](./routes-and-events.md)。

契约不是把公开函数逐一抄进文档，也不是要求每个子系统共享同一套内部对象。它描述的是一次能力交接：调用方必须提供哪些事实，所有者承诺返回什么，以及双方都不能偷偷改变哪些语义。只要这些交接保持稳定，Gateway 的分析流程、Patchouli 的记忆实现和 Alice 的执行循环就可以各自演进；一旦内部 workflow state 或引擎实体越过边界，局部重构便会重新变成全系统改造。

因此，公共模型倾向于使用 frozen、Pydantic 或依赖中立的 dataclass。不可变并不只是编码偏好，它要求上游先形成完整决定，再交给下游只读消费；依赖中立则阻止某个领域对象沿模型引用把存储、Runtime 或 Controller 一并泄漏出去。本文既记录字段和终态，也记录这些形态背后的所有权理由。

Gateway 与资源 owner 公开操作的身份输入使用 `IdentityScope`：它表达一次操作的发起者与目标 Workspace，由 workspace 的授权点在操作授权通过后组装。Patchouli 在公共边界拆为 `belong_to: WorkspaceIdentity` 与 `from_actor: ActorIdentity`，内部服务、引擎、存储、交互记录和后台任务独立携带所需身份，不保留或重建 scope。领域所有者在最终读写处先检查 Workspace 归属，再按资源 policy 检查发起者；资源来源不参与授权。共享的 queue、registry 和 Runtime 不因此按 Workspace 复制或分区。Alice/CPU 的清单仅携带观测标签，资源请求经绑定线程 context 的 workspace 操作入口授权（第 4 节）。WorkspaceAsset 旧端口的当前例外见[WorkspaceAsset](../workspace/assets.md#3-身份与归属)。

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
    session_turns: tuple[GatewaySessionTurn, ...] = (),
) -> GatewayProcessResult
```

`GatewaySessionTurn`（`core.protocol.gateway`）是交给 Gateway 的最近若干轮：`user_message`、`assistant_text` 与该轮实际落位的 `topic_id`（不知道时为空）。它不携带会话 ID 或授权身份；`topic_id` 不进入模型提示词，只在话题路由失败时使用：Gateway 续接最近一轮的话题（该话题仍在本次的候选话题中时），否则按新话题兜底。任务进程从会话轮次的话题标注取得落点，Import Bus 经 Patchouli 的交互状态查询取得上一轮应用成功后的真实落点（第 3.6 节）。

`GatewayProcessResult` 是不可变判别联合：

- `GatewayCommandOutcome(kind="command")`：包含命令解析结果 `CommandParseResult`；
- `GatewayDecisionOutcome(kind="decision")`：包含 `GatewayDecision`。

二者互斥。命令结果不能同时携带普通分析结果；普通决策不能携带命令解析结果。

Gateway 只解析命令，不执行命令：`CommandParseResult` 与 `CommandParseStatus` 位于 `core.protocol.gateway`，命令结果只携带解析产物（命令标识、名称、参数与解析状态），用户可见的命令终态由任务进程产生（第 5 节）。命令的实际运行不在 Gateway 中，因为后续指令会与用户请求同时出现，入口分析不能产生命令副作用；命令在任务进程中何时、由谁运行尚未决定，内置命令当前暂时不可用。

这种互斥使命令成为真正的短路终态。一条输入一旦被识别为系统命令，继续执行检索、Agent run 和记忆提交既浪费资源，也可能把一条控制消息误当成普通对话沉淀。判别联合让调用方必须显式选择一条链路，不能依赖多个可空字段猜测 Gateway 的意图。

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

- `ACTIVE_CHAT` 可以识别（解析）系统指令，不执行；
- `PASSIVE_MEMORY` 必须返回普通决策，绝不能返回 command outcome；它的调用方 Import Bus 只使用决定中的话题落位、新话题元数据、`rewritten_query` 与 `worth_saving`，忽略检索计划、不检索；
- `request_timeout_ms` 只能收紧配置的默认总超时，不能扩大它；
- 局部可恢复失败可以降级，但最终结果仍必须满足完整终态不变量。

## 3. Patchouli 契约

Patchouli 的公开面分为 chat 协作、记忆、任务、Agent Profile、话题与就绪状态。调用方不能直接调用 Patchouli local route。

这些能力虽然服务于不同用例，却共享一个核心约束：Patchouli 对长期状态和身份可见性拥有最终解释权。公开契约允许外部请求“创建、检索或提交”，但不会把 MemoryLibrary、Familiar 或生成 Controller 交给调用方直接操作。

### 3.1 PrepareAgentRun

```python
prepare_agent_run(
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
| `belong_to`、`interaction_id` | 冻结 prepare 的资源归属与交互关联；不保留 scope 或发起者 |
| `topic_id`、`is_new_topic` | 本轮真实话题与是否由 prepare 新建 |
| `memories` | 未编译的检索记忆（`tuple[MemoryAtom, ...]`，保持检索结果顺序）；检索被跳过时为空 |
| `storage_available` | 记忆存储健康状态 |

它位于 Patchouli 的 `contracts` 子包，因为 workspace 的任务进程需要读取其中的话题与检索结果；L3 子系统之间只能导入对方的 `contracts`。用户消息与 Gateway 决定由任务进程自己持有（组装输入清单与封口交互记录时使用），prepare 不接收用户消息，也不回传决定。

prepare 的意义在于：由 Patchouli 在交出控制权前确认真实话题与本轮可见的记忆，调用方无需理解 Patchouli 内部存储。检索结果以原始 `MemoryAtom` 交出，如何呈现给执行者由任务进程决定。

### 3.2 FinalizeAgentRun

```python
finalize_agent_run(
    prepared_run: PreparedAgentRun,
    payload: InteractionPayload,
    *,
    identity_scope: IdentityScope,
) -> None
```

`payload` 是任务进程封口的本轮交互记录。进程在 Actor 有输出时、进入 finalize 之前以 core 的共用封口函数 `seal_interaction`（`core/protocol/sealing.py`）组装它，材料全部来自进程自身：

| 字段 | 来源 |
|:---|:---|
| `outcome` | CPU 执行结局 completed/cancelled/failed，默认 completed |
| `user_message` | 任务请求的入口消息 |
| `rewritten_query`、`worth_saving` | Gateway 阶段的决定 |
| `assistant_final_text`、`turn_events`、`model_used` | Actor 的执行结果 |
| `mtp_traces` | 由 core 的 `ActionReducer` / `TraceReducer` 从 `turn_events` 归约 |
| `used_attachments` | 附件编译确认实际进入上下文的附件引用快照 |

附件租借由任务进程持有并在进程结束时释放，finalize 不负责。

Finalize：

1. 验证当次阶段 scope，确认其目标 Workspace 等于 `prepared_run.belong_to`；不一致时在交互接纳前抛 `WorkspaceMismatchError`；
2. 用 prepared 的归属与当次 scope 的发起者构造 `InteractionSubmission`，将 payload 原样提交到目标话题，不改写其内容；
3. 等待交互 applied，然后 best-effort 记录预检索命中，返回 `None`；WRITE/UPDATE 已在 actor 操作内独立派发。

任务进程对有输出的 completed/cancelled/failed 结果封口，同步写入会话，再创建 finalize 任务；提交时重新阶段授权，不能从 prepared 取回旧 scope。只有正文的 CPU 结果在封口处补成规范助手事件，感知侧仍只接收结构化事件。Finalize 已成功后不能再 cleanup；调用方取消等待时，Patchouli 已接管的 finalization continuation 继续完成，cleanup 也不能删除其话题。

finalize 是执行事务与记忆事务的分界。交互记录由提交方封口：主动链路由任务进程封口，Import Bus 由 System 的 Import Bus 编排器封口，两者调用 core 中同一个 `seal_interaction`，交给 Patchouli 的载荷同形（Import Bus 没有附件，`used_attachments` 为空；没有 Gateway 决定时 `rewritten_query` 与 `worth_saving` 为空）；Patchouli 的公开路由因此不必读懂任何执行者的运行结果，换一个 CPU 也不需要 Patchouli 随之改变。轨迹归约规则只有 core 中的一份，封口方调用它，不会形成第二套规则。Patchouli 负责判断已封口的交互如何进入长期知识（提交、感知、生成与 lifecycle）；有输出的三种结局都封口提交，无输出或 finalize 前断连不提交；显式写入始终按自己的派发结果独立生成。

### 3.3 CleanupPreparedAgentRun

```python
cleanup_prepared_agent_run(
    prepared_run: PreparedAgentRun,
    *,
    identity_scope: IdentityScope,
) -> bool
```

Cleanup 只尝试删除 prepare 阶段新建但仍为空的话题，不负责附件租借。任务进程调用前重新执行阶段授权；拒绝时记录警告并继续关闭。Patchouli 校验当次 scope 与 prepared 的资源归属，越域时返回 `False`，不触发删除；已有话题、已有内容或已被 finalization continuation/submission queue 接管的话题同样不删除。调用方把 cleanup 当作失败补偿，不把返回 `False` 视为新的业务错误。

它不是 rollback，也不承诺撤销整个 prepare 之后发生的一切。跨子系统没有一项可以原子回滚的数据库事务；cleanup 只补偿明确由 prepare 创建、且仍可安全判断为空的临时副作用。将它描述为回滚会诱使调用方删除已经存在或已被其他流程使用的长期状态。

### 3.4 其他公开能力

| 能力组 | 当前公开行为 |
|:---|:---|
| Interaction | 提交 `InteractionPayload` 到指定或新话题（新话题可附标题与摘要），查询交互进度与等待应用（第 3.6 节）；进入 submission lane 时由 `InteractionSubmission.belong_to/from_actor` 独立携带归属与发起者 |
| Memory | 携带 `IdentityScope` 的 create/list/get/update/delete、feedback、retrieve、retrieve_by_aliases；create/update 在提交边界生成完整版本记录（无历史不成功），无变化的编辑不创建版本 |
| Memory Task | 携带 `IdentityScope` 的 list/get/cancel；按 task 的 `belong_to` 检查归属，get/cancel 对越域与不存在统一抛 `ResourceNotFoundError` |
| Agent Profile | 携带 `IdentityScope` 的 create/list/get |
| Topic | 携带 `IdentityScope` 的 list active、topic data、manual settle、evict；Patchouli owner 拒绝越域 topic |
| Citation | 记录正式记忆引用；workspace 引用交付统一使用 `workspace.reference_read` 来源 |
| Readiness | 模型 warmup、预取与 ready 查询 |

Memory 与 Topic 的 Workspace 归属和 actor 可见性由 Patchouli 执行，调用方不能仅凭拿到 id 就假设目标可见；操作授权由调用方一侧的授权点完成（第 3.5 节）。任务快照的 `belong_to` 必须等于 scope 的目标 Workspace，知道任务 ID 不构成权限。Topic ID 在领域上保持全局唯一；scope 在公共边界拆分，内部以 `belong_to + topic_id` 检查归属，不构造另一套局部 ID 命名空间。

`MemoryGenerationTaskSpec`、只读快照 `MemoryGenerationTask` 与跨子系统请求 `PendingAtomMaterializeTask` 均独立保存必需的 `belong_to`、`from_actor`。WRITE/UPDATE 保留提交 Agent；手动、idle、LRU 和 shutdown SETTLE 统一使用 `system_actor_for_workspace(belong_to)`（owner 用户、保留的 system、无 Team），参与内容的 Agent 只记入贡献者，结算查重只看到本 Workspace 的 PUBLIC 记忆。`TopicMaterializeTask` 与 Topic lease 只保存资源归属，不保留最近访问者。

Profile、Topic 与 Active finalize 的公共失败定义在 `patchouli.contracts.errors`；会话与意图失败定义在 `workspace.contracts.errors`，共享访问、资源和存储类型位于 `core.errors`。调用者按类型处理，HTTP/MTP 在各自出口呈现；完整分类见[错误模型](./error-model.md)。Profile 不可见与缺失同样隐藏，类型或配置损坏显式失败，不回退内置 Profile。finalize 未能确认接纳时保留 `admission_unknown` 与原因链，不声称没有副作用。

### 3.5 访问 context 与操作授权

Patchouli 是授权点以下的资源 owner：公开方法与 `PatchouliService` 的阶段方法（prepare、finalize、cleanup）以 `IdentityScope` 接收操作身份，不接收访问 context，也不做操作授权。操作授权在调用方一侧的授权点完成，再把 `WorkspaceOperationAuthorizer` 返回的 `IdentityScope` 交给紧随的 Patchouli 路由：actor API 用 `authorize_actor_operation(access, operation)` 检查 actor 操作白名单（接受未撤销的执行线程 context 与请求级 context，目标取其驻留 Workspace）；root API 用 `authorize_root(access, target_workspace)`，只接受保留的 `system`；任务进程的阶段调用（Gateway 话题读取、prepare、finalize、cleanup、话题池读取）用 `authorize_stage(access, target_workspace)`，不检查 actor 操作，发起者仍是进程的 Agent。finalize 与 cleanup 各自重新授权，不复用 prepare 的 scope。公共 handler 拆分身份后，内部调用只传归属与需要的发起者（完整模型见[身份与授权](../workspace/identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)第 2 节）。Gateway 的处理路由同样只接收 `IdentityScope`。

Patchouli 一侧仍独立执行资源授权：资源归属与 `MemoryAccessPolicy` 可见性在最终读写处校验，越域目标按不存在处理，不因调用方已通过操作授权而放宽。

Alice 的 READ、SEARCH、RUN 记忆目标、WRITE、UPDATE、CALL 目标 Profile 与 context_refs 均经 actor API 逐次操作授权；READ 与 RUN 的引用交付由 actor API 记录 `workspace.reference_read` citation。Patchouli 的公开交互路由（提交、状态查询与等待）与主动记忆意图提交路由只接收 `IdentityScope`：交互路由唯一的生产调用方是 Import Bus，在导入授权（`authorize_import`，校验准入与目标、不检查 actor 操作，拒绝保留的 `system`）之后调用；意图提交唯一的生产调用方是 actor API。新的调用方须在自己的授权点先完成授权。

Patchouli 提交与生成链沿用自身既有来源记录，公开 API 不接收 `CallerPrincipal` 或其他来源字段。阶段拒绝语义见[错误模型](./error-model.md)第 4.4 节。

### 3.6 交互提交、进度与等待

主动链路的 finalize 在 Patchouli 内部构造 `InteractionSubmission` 并提交共享 lane；独立的交互提交经三条公开路由（`patchouli.contracts.interaction_submission`），当前调用方是 Import Bus：

```python
interaction.submit(
    *,
    identity_scope: IdentityScope,
    payload: InteractionPayload,
    requested_topic_id: str = "NEW_TOPIC",
    interaction_id: str | None = None,
    origin: InteractionOrigin = "workspace_port",
    new_topic_title: str | None = None,
    new_topic_summary: str | None = None,
) -> InteractionSubmitResult

interaction.status(*, identity_scope, interaction_id) -> InteractionStatus | None
interaction.wait(*, identity_scope, interaction_id, timeout_seconds) -> InteractionStatus | None
```

- **提交**：公共 handler 把 scope 拆为 `InteractionSubmission.belong_to` 与 `from_actor`，按 `interaction_id` 幂等接纳；省略 `interaction_id` 时每次生成新标识，需要重发语义的调用方必须给出稳定标识。`new_topic_title`/`new_topic_summary` 只在 `requested_topic_id` 为 `NEW_TOPIC` 时允许非空（已有话题的标题与摘要由话题自身持有，提交不能借此改写），感知建立新话题时使用，缺省时使用默认标题。`origin` 为 `active_chat`、`import_bus` 或 `workspace_port`，只是观测标签，不驱动任何分支。队列容量已满或已停止表示这次提交没有被接管，以 `InteractionAdmissionUnavailableError`（`patchouli.interaction.admission_unavailable`，reason `capacity_rejected` / `queue_stopped`）拒绝，调用方可用同一标识重发。
- **进度**：`InteractionStatus` 的 `state` 为 `pending`（尚未应用完成）、`succeeded`（已应用，`topic_id` 给出真实落点，`NEW_TOPIC` 已解析）、`failed`（到达终态但没有成功，不再重试）或 `unknown`（接纳过，但工作记录已超出队列的终态保留窗口）；未被接纳，或旁路索引与工作记录都已裁剪时返回 `None`。归属是硬边界：同名交互由另一个 Workspace 提交时以 `WorkspaceMismatchError` 拒绝，不把其他 Workspace 的接纳当作重复。
- **等待**：`interaction.wait` 先按状态查询校验归属；仍为 `pending` 时等待交互应用到终态或超时，返回此时的进度，超时仍为 `pending`，队列停止时返回停止那一刻的进度。
- **重发**：同一 `interaction_id` 在等待中或已成功时，相同载荷返回原收据，不同载荷报冲突；到达终态而未成功的交互允许以同一标识重新提交，载荷可以不同（例如落位的话题在应用前已被驱逐，提交方重新路由后再交），新提交以递增的尝试号作为工作项标识入队。为查重保留的旁路索引只存编码载荷的摘要，冲突比较不需要留住整份载荷。

Interaction codec 为 v6，编码新话题的标题与摘要；当前进程内队列不注册旧版本。查重与进度都只覆盖进程内的保留窗口，不跨重启。

## 4. CPU 端口与 Alice 实现

### 4.1 CPU 端口

```python
type CPUEventSink = Callable[[dict[str, Any]], Awaitable[None]]


class CPUPort(Protocol):
    async def execute(
        self,
        manifest: CPUInputManifest,
        *,
        thread: ExecutionThread,
        generation_options: dict[str, Any] | None,
        events: CPUEventSink,
    ) -> CPUExecutionResult: ...
```

CPU 端口（`workspace.contracts`）是任务进程调用执行者的唯一接口：端口由 workspace 定义，执行者实现，组合根注入 `TaskProcessService`。进程只依赖端口与本节的中立模型，因此执行者可以替换而不改动进程与入口；当前唯一的实现是 Alice 的 `AliceCPU`（4.4），测试中的 `ScriptedCPU` 同样能跑完整个任务进程。端口采用对象而不是总线路由，是因为外部 harness 的驱动多数不是子系统：按路由契约接入，每种驱动都要新增路由常量，或在总线之后再建一层分派。

`CPUInputManifest` 是任务进程在分配 CPU 时组装的输入清单，与具体执行者无关。提示词内容在跨越 CPU 边界前已定稿，以下三项都是必填字段：

| 字段 | 语义 |
|:---|:---|
| `language: str` | 按 Profile 语言、进程默认值解析出的提示词语言；CPU 的教学、摘要与执行期回填（Alice 的 MTP 结果、错误与 CALL 响应）都沿用该值，不再从 Profile 重新推导 |
| `instructions: str` | 环境通知、人设段，按此顺序连接；无内容时为空字符串 |
| `turn_context: str` | 记忆段、附件段，按此顺序连接；无内容时为空字符串 |

这两段文本是不透明的固定输入：CPU 决定放置位置，不解析或改写。分段使 Alice 可以在人设与本轮材料之间插入自己的历史摘要，也不要求其他 CPU 采用 Alice 的消息布局或历史管理方式。编译不读取 CPU 配置；存储不可用时总是附加中立离线通知，不取决于是否开启 MTP 教学。

清单还携带 `process_id`、进程级观测标签 `labels: ExecutionLabels`、原样用户消息 `user_message`、只作挂载/观测的 `topic_id`、全量会话轮次 `session_history: tuple[ConversationTurn, ...]` 和可选 `compaction_checkpoint: SessionCompactionCheckpoint | None`（默认 None）。历史渲染与压缩由 CPU 自管；进程不知道执行模型的窗口，不在提示词编译时按模型预算裁剪材料。未编译的检索原子只在进程内使用：编译为 `turn_context`，并直接从 prepare 结果投影 `memory_refs` 事件，不进入清单。清单的 `labels` 注册认证成功后创建一次，与进程事件沿用同一对象，供运行级观测与历史归属使用，与主线程的标签相同；标签只有 `agent_id` 与 `workspace_id` 两个只读字符串，不含 actor、owner 或访问 context，也不能改变授权或资源目标。

清单回答“处理什么”，执行线程回答“谁来处理”。`thread` 是本次执行的主线程 `ExecutionThread`（`workspace.contracts`）：线程描述 `descriptor` 与主线程 context `access`，是 execute 的独立参数，不进入冻结清单。线程描述 `ThreadDescriptor` 是冻结、CPU 可见的值：

| 字段 | 语义 |
|:---|:---|
| `thread_id: str` | 线程标识：主线程为 `main`，CALL 子线程按派生顺序为 `child-1`、`child-2`…… |
| `labels: ExecutionLabels` | 线程自己的观测标签：主线程为注册标签，子线程为被调用方与注册时的 workspace；由线程表生成，CPU 不从图纸推导 |
| `agent_profile: AgentProfile` | 本线程运行的图纸，供模型参数、人设与展示使用；主线程的图纸是进程在 prepare 之前经阶段授权与读取视图解析的 Profile |
| `allowed_operations: frozenset[WorkspaceOperation]` | 派生时以本线程 context 读取的 actor 操作快照，空白名单得到空集合 |

操作快照只描述执行者能做什么，供执行者生成提示词与执行策略（Alice 翻译为 MTP 动词，外部执行者的工具列表由其适配器翻译），不是授权依据：每个操作请求仍由 actor API 按访问登记即时授权。CPU 驱动以公共 `OperationEntry` 把执行线程绑定为线程句柄（通用实现 `ThreadOperationSubmitter`，见 4.6），操作适配器只提交操作请求；CPU 读不出 context 的内容，也不持有进程 context。Actor 阶段结束（`execute` 返回终态结果，或提前结束）时主线程与 CALL 派生的全部子线程即被结束，子线程也可能已随 CALL 返回结束；此后的提交以 `ThreadEndedError` 拒绝。

端口语义：

- **推送与返回**：执行过程中，CPU 经 `events` 逐项推送交互事件；结束时返回唯一的终态结果 `CPUExecutionResult`。
- **交互事件**：带 `event` 与 `data` 的字典，进程写入输出通道后原样转交给流式交付，不做解释。sink 由进程定义：订阅者暂停、通道写满时等待（背压），停止时在等待处抛出 `asyncio.CancelledError`。
- **结局**：执行者自报的结局经终态结果的 `status` 表达；执行者抛出普通异常，由进程按失败处理，不接受任何候选结果。
- **取消**：用户停止时，进程以原生 `asyncio.CancelledError` 取消正在执行的协程。CPU 内部各层原样传播取消并完成收尾，只在最外层把取消转换为返回值：返回 cancelled 结果，携带已产生的输出与已完成的检查点，`status` 如实描述 CPU 的结局。没有转换、直接传播取消的 CPU 按无输出的取消处理，`has_output` 排除初始用户输入事件。CPU 返回之后到达的停止不改写这个结局（停止晚于终态时仍是 completed）。
- **取消后的收尾**：停止被接受时全部线程即已结束，此后的提交都以线程已结束拒绝；CPU 收到取消后的收尾不得依赖 actor 操作，只能用已经产生的内容组装结果。
- **`generation_options`**：由各个执行者自行解释，进程原样传递。

### 4.2 执行结果

`CPUExecutionResult` 是执行者对一次执行的完整事实声明，而不是已经提交的长期记忆：

- `status`：`completed`、`cancelled` 或 `failed`（`CPUExecutionStatus`）；
- `final_text`：最终用户可见文本；
- `turn_events`：结构化运行事实（`TurnEvent`）；
- `model_used`：执行者实际使用的模型展示名，空字符串表示未解析；
- `failure_reason: CPUFailureReason | None`：可选的稳定失败原因，仅 failed 时有意义，外部 CPU 可不填，取值见[错误模型](./error-model.md#22-agent-run)；
- `compaction_checkpoint: SessionCompactionCheckpoint | None`：默认 None，本次完成的旧历史压缩检查点，正常结果与取消后返回的结果都可携带。它不令 `has_output` 为真。

执行结果与交互记录均不携带物化任务；WRITE/UPDATE 在操作内提交 `MemoryIntent`，由 Patchouli 的意图提交服务构造内部生成任务。

执行者专属的统计（例如 Alice 的 MTP 迭代次数）不进入执行结果，只出现在各自的观测事件中。任务进程据执行结果决定是否进入 finalize，并从中封口交互记录；任何一方都不能仅凭流中的部分文本推断执行已经完成。

### 4.2.1 会话压缩检查点

`SessionCompactionCheckpoint` 位于 `core.protocol.models`，是冻结的 CPU 中立值模型。会话、输入清单与执行结果均以可选字段 `compaction_checkpoint` 携带；会话至多持有一份，原始轮次完整保留。

| 字段 | 契约 |
|:---|:---|
| `producer` | 产生方标识；Alice 为 `alice` |
| `format_version` | 至少为 1 的格式版本；Alice 当前为 1 |
| `reduced_through` | 最后一个减重轮次的 `interaction_id`，默认 None |
| `summarized_through` | 最后一个摘要覆盖轮次的 `interaction_id`，默认 None |
| `summary` | 接力摘要正文，默认空字符串 |
| `references` | 不可变字符串 tuple，默认空；HTTP JSON 序列化为数组 |
| `created_at` | 生成时间，默认当前 UTC 时间 |

游标只覆盖该会话已有的完整轮次；摘要边界不晚于减重边界；同一产生方与格式版本内，已接受的两个游标分别单调前进，换用其他格式的 CPU 可以从头重建并替换旧值。会话服务同步校验 Workspace、认领与游标后接受，完整规则见 [会话资源](../workspace/sessions.md)。进程在 Actor 阶段结束、撤销线程后接受，不以本轮输出或是否追加新轮次为条件；检查点不进入 `InteractionPayload` 或记忆生成。未知产生方/版本由 CPU 自行选择忽略，外部 CPU 可以完全不消费、不产生检查点。

### 4.3 Alice 的执行入口

Alice 通过上述 `CPUPort` 对外提供执行能力，唯一的端口实现为 `AliceCPU`。`AliceSystem.start()` / `stop()` 切换 CPU 执行准入；未启动或已停止时，await `execute()` 抛出 `AliceUnavailableError`（异常上的 code 与 reason 只供诊断，见[错误模型](./error-model.md)）。任务进程按未分类的执行异常收尾，仍执行 prepare 清理与会话释放。

Alice 不注册执行总线路由，组合根构造 AliceSystem 时不传全局总线。启停只控制新执行的准入，运行中的进程由 workspace 取消和关闭；System 先关闭进程，再停止子系统。

### 4.4 Alice 的端口实现

`AliceCPU`（`alice/interface/cpu.py`）持有公共 `OperationEntry`，以 `ThreadOperationSubmitter` 把主线程绑定为线程句柄后直接调用内部运行服务。`AgentRunService` 接收清单、主线程句柄、生成参数与输出端口，始终返回 `CPUExecutionResult`，不区分传输形态：图纸与操作快照取自句柄交出的主线程描述，用于模型解析、根帧与 MTP 动词；注册标签取自清单，用于 `agent.run.*` 与历史中的当前 Agent；根帧布局与历史压缩由 Alice 的上下文层完成。

每次执行由 `AliceCPU.execute` 直接等待运行服务；接口层的 `SinkRunOutput`（`alice/interface/output.py`）把帧输出、CALL 起止与压缩进度投影为交互事件，补全 run 关联与 `stream_sequence` 后推送给进程的 sink，不缓存事件。运行服务是 Alice 作为 CPU 的最外层：内部各层原样传播取消，执行器递归收尾后，运行服务把根帧的部分文字、轮次事件与已完成的检查点组装为 cancelled 结果返回，不再传播取消；执行器尚未开始且没有完成的检查点时没有可交回的内容，原样传播取消。不存在 Alice 自己的 `done` 序列化；任务进程负责最终 HTTP 投影。

### 4.5 Alice 不变量

- Alice 不修改输入清单所引用的长期记忆或话题；
- Alice 不直接请求 Patchouli；检索、引用交付与 Profile 读取经 workspace 操作请求完成，Alice 不持有资源缓存或 resolver；
- WRITE/UPDATE 经操作请求登记 workspace 意图并实时派发，生成队列接纳后返回 ACK；Alice 不持有登记；
- 有输出的 completed/cancelled/failed 都封口并进入 Patchouli finalize；无输出或进程异常不提交；
- Agent Profile 不承载权限；frame 的 MTP 动词集合由所属线程描述的 `allowed_operations` 翻译而来（RUN 始终可用，`sys_*` 系统工具不设权限），提交到 workspace 的请求仍由 actor API 逐次授权；
- CALL 仅允许根 frame 发起；子 frame 是以被调用方身份运行的子线程，动词集合由被调用方的操作快照翻译并去掉 CALL，不能继续递归 CALL；被调用方 Profile 只提供人设、模型与语言，子 frame 的观测标签取子线程描述；
- 子线程随 CALL 返回结束：子帧以任何终态收尾、取消沿递归栈清理 CALL，以及派生之后子帧开始运行之前的失败，都经子线程句柄同步结束子线程；Alice 不结束主线程。

### 4.6 操作请求与线程 context

`OperationRequest[R]` 是 frozen dataclass 结果类型契约；具体请求只携带业务参数，不接收访问 context、`IdentityScope`、目标 Workspace 或 process_id。六项 actor 请求与 actor 操作一一对应，每种请求恰好检查一项操作；五项记忆请求是 `OperationRequest`，CALL 的派生请求只经入口的生命周期方法提交：

| 请求 | MTP 动词 | 参数 | 结果 `R` | actor 操作 |
|:---|:---|:---|:---|:---|
| `ResolveReferencesRequest` | READ（含 CALL 的 `context_refs`） | `aliases: tuple[str, ...]` | `list[ReferenceResolution]` | `memory.read` |
| `RetrieveRequest` | SEARCH | `semantic_query`、`keywords=()`、`top_k=None`（缺省时由 `retrieval.default_top_k` 决定）、`filters=None` | `list[MemoryAtom]` | `memory.search` |
| `ResolveRunTargetRequest` | RUN（记忆中的用户工具） | `alias: str` | `ReferenceResolution` | `memory.run` |
| `SubmitWriteIntentRequest` | WRITE | `focus: WriteFocus` | `PendingAtom` | `memory.write` |
| `SubmitUpdateIntentRequest` | UPDATE | `base_alias`、`instruction`、`content=None` | `PendingAtom` | `memory.update` |
| `DeriveCallThreadRequest`（不是 `OperationRequest`） | CALL（只有 Alice 使用） | `agent_alias=None` | 经 `derive_child_thread` 得到子线程的 `ExecutionThread`：线程描述（目标 `AgentProfile`、被调用方的操作快照与标签）与子线程 context，只交付 CPU 驱动 | `agent.call`（内置同样需要） |

RUN 的 `sys_*` 系统工具在 Alice 本地执行，不提交请求。

公共入口 `OperationEntry` 有三个方法：

- `execute[R](request: OperationRequest[R], *, access: WorkspaceAccessContext) -> R`：分派五项记忆请求；
- `derive_child_thread(request: DeriveCallThreadRequest, *, access) -> ExecutionThread`：以调用方的线程授权 `agent.call`、解析目标图纸并派生子线程；
- `end_thread(*, access) -> None`：同步结束子线程；已结束的线程是空操作，主线程以 `ScopeRequiredError` 拒绝。

线程句柄协议 `OperationSubmitter` 是 CPU 侧唯一的线程句柄：`thread` 属性交出线程描述；`__call__[R](request: OperationRequest[R]) -> R` 以本线程提交记忆请求；`derive_child(request: DeriveCallThreadRequest) -> OperationSubmitter` 以本线程提交 CALL 派生，返回子线程的句柄；`end() -> None` 同步结束本线程，可以在取消收尾中调用，不引入新的挂起点。`ThreadOperationSubmitter(entry, thread)` 是 `workspace.contracts` 提供的通用实现，任何 CPU 驱动都用它绑定执行线程，操作适配器、frame 与 CALL 编排因此不接触 context。每个 frame 持有所属线程的句柄。新增资源操作应扩展请求与入口，而不增加 workspace 总线路由或让适配器切换资源目标。

这一形态来自“绑定”与“调用面”的拆分：绑定（谁发起、属于哪个进程与线程）随线程变化，由线程 context 承载；调用面（能调用哪些能力）是稳定的，由请求类型与 actor API 各分类拥有。入口只按请求类型分派，接口不随能力增加而增长；操作适配器（Alice 的 MTP 适配器与外部 harness 的 MCP 入口）只把协议请求解析为请求对象，不持有能力层，也不持有身份。workspace 的调用方只有执行单元、HTTP 管理入口与 MCP 入口三类，因此不为它另建总线路由。

线程提交只接受存活线程的 context：`execute` 在分派前经 workspace 内部的线程端口 `ExecutionThreadTable`（`workspace/thread_ports.py`，不属于公共契约）向线程表做一次存活判定——线程未结束且进程仍在运行，再按请求类型把请求对象与线程 context 直接交给对应的 actor API 方法（`method(request, *, access)`）；入口不拆参数，既不替代逐次操作授权，也不替代资源 owner 的归属与 policy 校验；入口不持有 root API。actor API 的目标取线程的驻留 Workspace，意图登记的进程关联、线程标识与派发话题取自线程表的一次解析。未知请求类型抛 `TypeError`；进程 context、请求级 context、未登记的线程或其他对象抛 `ScopeRequiredError`；线程已结束（凭据已撤销、线程已结束或进程已收到停止）抛 `ThreadEndedError`（`core.errors`，由 `workspace.contracts` 重新导出）。拒绝都发生在任何副作用之前。

`WorkspaceOperationEntry` 另有请求级提交 `execute_request[R](request, *, access) -> R`，不属于 CPU 可见的 `OperationEntry`：不建进程的外部入口（MCP）以请求级 context 提交，门面以 `operation_entry` 交出同一个入口，只供 MCP 入口调用。它只接受 `RetrieveRequest` 与 `ResolveReferencesRequest`，与 `execute` 共用同一分派表，其余请求类型在分派前抛 `TypeError`；“请求级 context 只开放检索与读取”只写在这一处。它不做存活判定：请求级 context 只在一次调用内有效，撤销由 actor API 的授权检查。请求级 context 因此到不了 RUN、WRITE、UPDATE 与 CALL：经 `execute` 提交时被线程表拒绝，经 `execute_request` 提交时被类型表拒绝；WRITE、UPDATE 登记前还要解析线程，CALL 派生先判定调用方存活（[MCP 记忆访问](./mcp.md#4-操作入口与授权)）。

`derive_child_thread` 在一次提交内依次完成：确认调用方存活；actor API 的 `resolve_call_target` 授权 `agent.call` 并以调用方的可见性读取图纸（只交回图纸）；线程表在冷读之后重新判定调用方存活、确认调用方是主线程，以被调用方准入后签发子线程、读取被调用方的操作快照并登记（[执行线程](../workspace/execution-threads.md#6-call-派生与子线程)）。被调用方的身份因此只来自 workspace 自己的解析，CPU 不能交回一份图纸来自报子线程的身份。被调用方未准入或为保留的 `system` 时抛 `AdmissionDeniedError`，子线程试图派生抛 `ScopeRequiredError`，冷读期间调用方结束或进程收到停止抛 `ThreadEndedError`，均不登记子线程。

Actor 阶段结束时执行器同步结束本进程仍在运行的全部线程，关闭流程兜底；子线程也可以先随 CALL 返回结束。结束不取消在途调用任务，已开始的只读操作可以自然完成，后续提交拒绝。WRITE/UPDATE 在基础校验之后的同步段只解析一次线程，取得进程关联、线程标识与派发话题并登记为 MATERIALIZING；UPDATE 冷读跨越线程结束或停止时不登记。派发开始后持有 shield 保护的提交任务直到结束，再传播调用方取消；主、子线程共享 CPU 输入清单的话题。完整生命周期见 [执行线程](../workspace/execution-threads.md#7-actor-阶段结束与停止)。

## 5. 顶层主动链路契约

```text
Gateway command outcome（解析结果）
  -> 任务进程产生命令终态（解析成功：暂不可用；解析失败：拒绝）
  -> 不调用 Patchouli prepare / CPU / Patchouli finalize

Gateway decision outcome
  -> 解析 Agent Profile（阶段授权后经 workspace 的 Profile 读取视图）
  -> Patchouli prepare（Topic 与检索）
  -> CPU 分配：附件租借、附件与记忆编译、组装输入清单
  -> Actor 执行：从进程 context 派生主线程 context，等待 CPU 端口（当前为 Alice）返回
       -> CALL：经操作入口派生子线程（被调用方准入），进程登记
       -> CPU 返回终态结果后，同步撤销全部线程 context
  -> 有输出的 completed/cancelled/failed：封口交互记录（含结局和实际使用附件）
       -> 同步追加会话 -> 持有 finalize 任务 -> Patchouli finalize
  -> 无输出/exception/finalize 前交付方离开：不提交，转入关闭补偿
  -> 进程结束（在进程 task 中）：先同步撤销仍登记的线程 context（兜底）、释放附件租借
       -> prepare 成功但 finalize 未成功时请求 Patchouli cleanup
       -> 注册入口最后撤销进程 context、注销进程并释放会话认领
```

Agent Profile 属于 CPU 分配，但当前在 prepare 之前解析：prepare 可能新建 Topic 或按 LRU 结算已有话题，Profile 缺失的请求应在这些副作用发生前失败。

该顺序由 `TaskProcessService`（`workspace.process`）拥有。任何 transport adapter 都不能复制或调整此顺序。

顺序本身就是契约的一部分：Gateway 先收敛入口语义，Patchouli 再准备长期知识的本轮视图，CPU 只执行，最后由 Patchouli 提交。让 transport adapter 复制这条链路，会很快产生“HTTP 可以、其他入口不可以”或两条 finalize 规则不一致的问题。

## 6. Import Bus 契约

Import Bus 由 System 拥有：`ImportBusService` 做导入授权并委托编排器 `ImportBusIngestor`。它接收外部 harness 一轮已经结束的交互的完整记录，以 Gateway `PASSIVE_MEMORY` 模式只取话题落位与价值信号，以共用封口函数封口，经 Patchouli 的公开交互路由提交并等待应用完成（第 3.6 节）。它不调用 Alice、不执行命令、不运行 MTP、不生成回复，不检索记忆，也不返回记忆上下文。HTTP 入口与载荷字段见[公开路由与事件](./routes-and-events.md#62-import-bus-导入)，编排细节见 [Import Bus](../system/import-bus.md)。

对外结果只有 `accepted`（本次提交且已应用）、`duplicate`（此前已应用）与 `skipped`（用户消息之外没有输出，与任务进程一样不封口），并带推导出的 `interaction_id`；Gateway 决定、话题、价值信号与 RuntimeEvent 不进入响应。`accepted` 与 `duplicate` 都表示这一轮已经应用，确认只在应用完成之后给出：可重试与不可重试的失败经错误区分（[错误模型](./error-model.md#43-import-bus)）。

- **交互标识**：`import_` 加上对（principal、user、Agent、来源、外部会话、外部轮次）规范化序列取 SHA-256 的前 32 位十六进制。同一轮重发得到同一标识，队列据此幂等；纳入认证通过的 actor，使代表不同用户或 Agent 的连接器不会撞号。
- **串行与查重**：同一（user、Agent、来源、外部会话）在单进程内按到达顺序串行，串行范围包含进度查询、Gateway、提交与等待应用；不同会话可以并发。门内先查询进度：已应用（`succeeded`）或接纳过但结局已超出保留窗口（`unknown`）时返回 `duplicate`；仍在应用（`pending`）只继续等待，不再路由与提交——Gateway 的结果写入载荷，重新路由可能得到不同载荷而被队列判为冲突；未提交或应用失败的轮次才调用 Gateway 并（再次）提交。该契约不扩展为跨进程排序或持久化 mailbox。
- **落位**：`session_turns` 取请求中最近若干轮的最后 `gateway.context_preparation.session_turn_limit` 轮；最近一轮带外部标识时，Import Bus 推导它的交互标识并经状态查询取得应用成功后的真实落点，作为该轮的 `topic_id`。新话题以 Gateway 给出的标题与摘要提交。
- **事件**：`tool_kind` 只接受 `SEARCH`、`READ`，只标在 HiveMemory 自身 MCP 工具的调用上，生成视图对它们只留一行轨迹，不再渲染取回的记忆；其他外部工具调用不设置，不产生记忆操作轨迹，也不会被当作记忆操作。用户消息由服务端写成序号为 0 的事件。

这条限制保护的是入口语义。Import Bus 摄入的是已经在外部发生的经历，并不等价于伪造一次用户与 Agent 的对话；如果它允许命令或 Alice 执行，外部内容便可能意外触发控制行为、工具调用和回复生成，也会让“谁发起了这次行动”失去可靠答案。

## 7. Interaction 与 Topic 时序契约

主动链路与 Import Bus 的消息来源和入口流程不同，但二者最终都向 Topic 追加 Interaction，因此共享同一组时序职责：

`InteractionSubmission` 是进入 Patchouli submission lane 的稳定交接包：必需的 `belong_to` 与 `from_actor` 独立保存归属与发起者，`interaction_id` 负责幂等关联，`InteractionPayload` 只承载本轮内容，不嵌入操作 scope。Interaction codec v6 与 `schema_version="1.1"`（字符串）的 generation codec 分别 round-trip 两种身份；当前进程内队列不注册旧 codec，retry 恢复相同字段，不重跑身份解析、不重新组装 scope。同一 `interaction_id` 的幂等只由提交队列负责：等待中与已成功的提交以相同载荷重交时复用原收据、载荷不同时冲突，只有应用未成功的交互可以重新提交；应用把 block 写入话题是它最后一个可能失败的步骤，未成功的应用没有写入，重新提交因此不会重复写入。`TopicAssetBinding` 只有在该 Interaction 成功应用且用户明确使用 asset ref 时才成立；上传或 UI 选择不会单独产生 binding。

`InteractionPayload.used_attachments` 携带任务进程的 `AttachmentCompiler` 确认实际进入上下文的 bound ref 快照（由任务进程封口时写入）：submission handler 把这份快照以 `asset_refs` 形参一次性传给 `apply_interaction`，不回查原始选择、asset 列表或当前 UI 状态。retry 重放同一份快照，不能替换 ref。

1. **Interaction 内全序由生产者冻结。** `TurnEvent.sequence` 只在所属 interaction 内有效；payload 一旦进入 submission queue，retry、dedup、cleanup 和 handler 都不得改写既有事件顺序或生成新的语义身份。
2. **Topic append 顺序由 Patchouli 拥有。** 当前以成功 apply 的实际 append 顺序作为 topic-local 权威投影。若未来增加 `topic_position`，必须由 topic owner 在持久化提交时原子分配，调用方不能根据时间戳自行计算。
3. **Queue FIFO 只是执行约束。** ordering key 只串行化已经入队的 work，不代表源事件发生时间，也不表达 Agent 因果关系。话题折叠项与该话题的提交共用排序键，折叠执行期间同一话题的后续提交在队列中等待；折叠项不改写交互内容与交互之间的先后。队列的幂等接纳只防止重复副作用，不参与排序。
4. **外部轮次的先后由提交方保证。** Import Bus 只保证同一外部会话内按到达顺序接纳与应用；连接器按会话顺序提交、确认后才推进，Patchouli 不按外部时间缓存、等待或重排晚到的轮次。导入与主动链路落到同一话题时，由同一个按话题的排序键保证先后。
5. **Active prepare 读取的是一个 topic snapshot。** finalize/apply 的先后位置只能证明提交顺序，不能证明某个 run 在生成 LLM input 时看见过中间提交。多个 run 可以基于同一 revision 并发执行。
6. **多 Agent 并发优先表达因果偏序。** 未来应保存 `base_topic_revision`、causal parent 或等价的 run/frame 关系，再由 prompt/display 层按需要确定性线性化；不得仅按完成时间伪造语义先后。

`occurred_at`、`received_at`、`enqueued_at` 与 `applied_at` 是不同阶段的观测时间，均不能替代 topic 内的权威顺序。当前契约要求的是 topic-local authoritative log 与可扩展的 causal relation，而不是全系统绝对时间线；本阶段不要求全局序列、向量时钟或完整事件溯源。

## 8. 契约矛盾检查

新增或修改公共能力时，应先回答以下问题：

1. 这个模型是否只包含一次交接需要的稳定事实，还是暴露了可变 workflow state、引擎对象或回调？
2. 接收方是否正在修改本应只读的决定，或重新推导一份与所有者可能分叉的状态？
3. command outcome 是否仍能立即短路？Gateway 是否开始执行命令，或在公开结果中携带执行产物？`PASSIVE_MEMORY` 模式与 Import Bus 是否可能通过新分支触发命令、Alice、MTP 或回复生成？
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
- 允许 `PASSIVE_MEMORY` 模式返回命令；
- 改变 CPU 端口的输出语义、执行结果终态和 finalize 资格；
- 将 local route 或内部 workflow state 暴露为公共 API。

验证入口：`tests/unit/system/contracts/`、`tests/unit/workspace/process/`、`tests/unit/system/application/`、`tests/unit/gateway/test_contracts.py`、`tests/unit/patchouli/test_service_gateway_decision.py`、`tests/unit/alice/run/test_service.py`、`tests/unit/alice/interface/test_cpu.py`、`tests/unit/workspace/process/test_cpu_port.py`、`tests/unit/patchouli/application/`、`tests/unit/patchouli/control/test_interaction_submission.py`、`tests/unit/core/protocol/test_seal_interaction.py`、`tests/unit/system/services/import_bus/`、`tests/integration/system/test_import_bus_submission.py`、`tests/integration/workspace/test_application_access_boundary.py`。
