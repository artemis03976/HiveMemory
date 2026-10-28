---
title: Routes and Events
status: current
owner: system
scope: global-routes-and-events
code_paths:
  - src/hivememory/server/deps.py
  - src/hivememory/core/contracts/route.py
  - src/hivememory/patchouli/contracts/public_routes.py
  - src/hivememory/gateway/contracts/public_routes.py
  - src/hivememory/alice/interface/cpu.py
  - src/hivememory/alice/interface/output.py
  - src/hivememory/server/routers/chat.py
  - src/hivememory/server/routers/import_bus.py
  - src/hivememory/server/models/import_bus.py
  - src/hivememory/system/services/import_bus/events.py
  - src/hivememory/core/contracts/events.py
  - src/hivememory/core/contracts/runtime_events.py
  - src/hivememory/components/bus/
  - src/hivememory/components/events/bus.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/README.md
  - docs/system/import-bus.md
last_reviewed: 2026-10-21
---

# 公开路由与事件

本文是进程内跨子系统路由名、事件名和投递语义的规范入口；第 6 节同时固化共享身份交接、会话管理、Chat SSE 与导入入口的 HTTP 契约。

全局总线的目标是解除对象图耦合：Gateway 与 workspace 可以请求 Patchouli 的公共能力，却不需要持有它的 Runtime、Service 或存储对象。它不是为了把单体进程伪装成分布式系统，也不提供网络消息中间件的可靠性语义。明确这一点很重要，否则调用方容易把一次普通函数交接误写成无人负责的事件流，或反过来依赖总线并未承诺的持久化、重试和 exactly-once。

## 1. 总线语义

`GlobalSystemBus` 继承 `AsyncSystemBus`，只服务跨子系统公开能力。

选择 RPC 还是 Pub/Sub，取决于发布者是否必须得到一个确定结果。需要数据、确认或失败传播时使用 RPC；只需告知其他所有者“某件事已经发生”，且发布者不依赖订阅结果时使用 Pub/Sub。两者都能减少直接对象依赖，但不能互相替代。

### 1.1 RPC

- `register(route, handler)`：一个 route 对应一个 handler；重复注册会覆盖并记录 warning；
- `unregister(route)`：移除 handler，不存在时为 no-op；
- `request(route, *args, **kwargs)`：调用 handler 并等待结果；
- 未注册 route：抛出 `KeyError`；
- handler 应为 async；当前实现为兼容测试和适配器，也会直接返回非 awaitable 值。

RPC 不提供持久化、重试、跨进程投递或版本协商。

RPC 的等待关系也意味着责任关系：handler 的返回值或异常直接决定本次调用能否继续。将需要确定结果的 prepare、retrieve 或 run 改为 Pub/Sub，会让顶层链路无法知道工作何时完成，也无法可靠决定 finalize 或 cleanup。

### 1.2 Pub/Sub

- `subscribe(event, callback)`：同一事件可以有多个 async subscriber；
- `publish(event, ...)`：按订阅列表逐个等待 callback；
- 没有订阅者：静默 no-op；
- 单个 subscriber 异常：记录错误并继续，不传播给 publisher；
- `unsubscribe`：只移除指定 callback。

Pub/Sub 是通知语义，不能用于要求调用方获得确定返回值的工作流。

订阅者失败不传播给 publisher，是为了让通知的次要消费者彼此隔离；代价是事件不能承担必须成功的业务提交。若业务正确性依赖某个 subscriber 完成，就应将那一步建模为 RPC 或由明确所有者编排，而不是继续增加隐式订阅顺序。

## 2. 规范路由名

公开 route 的名称与签名由提供方 `contracts/public_routes.py` 中的 `PatchouliRoutes`、`GatewayPublicRoutes` 声明。调用方直接导入这些类型化 route；core 的 `Route` / `route` 只提供依赖中立的泛型原语。静态注册、调用与绑定表检查，以及运行时参数检查的分工见[运行时机制](../components/runtime-and-bus.md#1-globalsystembus)。

一个公开调用的契约由三部分共同构成：route 字符串确定能力身份，handler 参数名和调用形式确定交接方式，公共模型确定输入输出语义。只同步其中一项仍可能让注册方与调用方在运行时分叉，因此参数重命名、模型字段变化和 route 重命名都属于契约变更。

### 2.1 Gateway

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `gateway.public.process` | `GatewayService.process` | message、`IdentityScope`、ingress mode、`session_turns: tuple[GatewaySessionTurn, ...] = ()`、可选 `request_timeout_ms` | `GatewayProcessResult`（命令结果只携带解析结果） |

### 2.2 Patchouli Chat / Retrieval

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `patchouli.public.memory.retrieve` | `MemoryManagementService.retrieve` | `RetrievalRequest`（含 `identity_scope`） | `list[MemoryAtom]`（按领域排序） |
| `patchouli.public.memory.retrieve_by_aliases` | `retrieve_by_aliases` | aliases、`IdentityScope` | `list[MemoryAtom]`（只含实际可读的原子） |
| `patchouli.public.memory.read` | `read_memory` | memory id、`identity_scope`、refresh | `MemoryAtom \| None`（未知或不可见均为 `None`） |
| `patchouli.public.prepare_agent_run` | `PatchouliService.prepare_agent_run` | `IdentityScope`、`interaction_id`、`GatewayDecision`、是否检索 | `PreparedAgentRun`（Topic 准备结果与未编译检索结果） |
| `patchouli.public.finalize_agent_run` | `PatchouliService.finalize_agent_run` | `PreparedAgentRun`、任务进程封口的 `InteractionPayload`（含实际使用的附件引用）、必需的 `identity_scope` | `None`（交互 applied、命中记录完成）；目标归属不一致抛 `WorkspaceMismatchError` |
| `patchouli.public.cleanup_prepared_agent_run` | `cleanup_prepared_agent_run` | `PreparedAgentRun`、必需的 `identity_scope` | 是否清理空话题；越域 prepared 返回 `False` |
| `patchouli.public.record_memory_citation` | `record_memory_citation` | memory id、`IdentityScope`、source | 记录结果 |

交互与主动意图提交：

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `patchouli.public.interaction.submit` | `submit_interaction` | `identity_scope`、`InteractionPayload`、可选 requested topic / interaction id / origin、新话题的 `new_topic_title` / `new_topic_summary`（只在 `NEW_TOPIC` 时允许） | `InteractionSubmitResult`（接纳收据）；队列容量已满或已停止抛 `InteractionAdmissionUnavailableError` |
| `patchouli.public.interaction.status` | `interaction_status` | `identity_scope`、interaction id | `InteractionStatus \| None`（`pending` / `succeeded` 附真实落点 / `failed` / `unknown`；未接纳为 `None`）；同名交互属于其他 Workspace 抛 `WorkspaceMismatchError` |
| `patchouli.public.interaction.wait` | `wait_interaction` | `identity_scope`、interaction id、`timeout_seconds` | 应用到终态或超时时的 `InteractionStatus \| None`；归属校验同 status |
| `patchouli.public.memory_intent.submit` | `submit_memory_intent` | `identity_scope`、`MemoryIntent` | `MemoryIntentSubmissionResult`（accepted / rejected / unknown；仅 accepted 带任务坐标） |

上表路由只接收调用方授权点组装的 `IdentityScope`，本身不做操作授权。`memory_intent.submit` 的生产调用方是 workspace actor API 的 WRITE/UPDATE：`MemoryIntent.intent_id` 与 `pending_alias` 必填，沿用登记值；话题由任务进程提供。Patchouli 在内部构造物化任务，唯一调用 local `GENERATION_SUBMIT_ACTIVE`，每次一个意图，返回接纳、明确拒绝或未知结果。明确拒绝结算 FAILED；未知结果不提前结算，既有任务保留结算权。

三条交互路由的生产调用方是 Import Bus，在导入授权之后调用：提交前以 `interaction.status` 查重，提交后以 `interaction.wait` 等待应用；进度、重发与幂等规则见[子系统公共契约](./subsystem-contracts.md#36-交互提交进度与等待)第 3.6 节。chat finalize 不经这些路由：Patchouli 在 finalize 内部构造 `InteractionSubmission` 并提交同一条 lane，两条路径的 handler 都调用 `PerceptionFamiliar.apply_interaction()`。`InteractionPayload` 不携带物化任务；submission 的 `belong_to/from_actor` 分别是归属与发起者来源，调用方不能从 payload 或 topic id 反推作用域。`chat.run.completed` 和 SSE `done` 都不携带记忆任务 ID；任务通过 `memory.task.*` 事件及 list/get 查询观察。

Patchouli 公共 handler 验证 scope 后拆分身份，内部服务、引擎与存储只接收 `belong_to` 和需要的 `from_actor`。`PreparedAgentRun` 只保留 prepare 的归属；任务进程在调用 finalize 前重新执行阶段授权并传入当次 scope；cleanup 同样重新执行阶段授权，授权被拒时记录警告并完成关闭；资源 owner 拒绝删除越域 prepared 或已被提交链接管的话题。

### 2.3 Patchouli Memory

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `patchouli.public.memory.create` | `create_memory` | `identity_scope`、`MemoryCreation`（创建材料；类型不得为 `AGENT_PROFILE`） | 持久化后的 `MemoryAtom` |
| `patchouli.public.memory.list` | `list_memories` | `identity_scope`、query、filters、limit、exclude、refresh | `list[MemoryAtom]` |
| `patchouli.public.memory.get` | `get_memory` | memory id、`identity_scope`、refresh | `MemoryAtom | None` |
| `patchouli.public.memory.update` | `update_memory` | memory id、`identity_scope`、`MemoryEdit`（部分编辑，`None` 表示不修改） | `MemoryAtom | None` |
| `patchouli.public.memory.delete` | `delete_memory` | `identity_scope`、memory id | `bool` |
| `patchouli.public.memory.record_feedback` | `record_feedback` | memory id、`identity_scope`、positive、source | 记录结果 |

创建与编辑路由只接收材料（`patchouli.contracts.memory_management`），原子由 Patchouli 构造或修改：记忆 ID、时间戳、来源记录（取发起者）、默认访问策略与生命周期都在提交边界确定，字段非法以 `InvalidMemoryFieldError` 拒绝。

### 2.4 Patchouli Tasks / Profiles / Topics / Readiness

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `patchouli.public.memory_task.list` | `list_memory_tasks` | `identity_scope` | 本 Workspace 的 task 列表 |
| `patchouli.public.memory_task.get` | `get_memory_task` | task id、`identity_scope` | task；越域与不存在统一抛 `ResourceNotFoundError` |
| `patchouli.public.memory_task.cancel` | `cancel_memory_task` | task id、`identity_scope` | `bool`；越域与不存在统一抛 `ResourceNotFoundError` |
| `patchouli.public.agent_profile.create` | `create_agent_profile` | `identity_scope`、`AgentProfileCreation`（创建材料，alias 必填） | 持久化后的 `MemoryAtom` |
| `patchouli.public.agent_profile.list` | `list_agent_profiles` | `identity_scope`、limit | profile atom 列表 |
| `patchouli.public.get_agent_profile` | `get_agent_profile` | agent alias、必需的 `IdentityScope`（builtin 同样需要） | `ResolvedAgentProfile`（`AgentProfile` + 源原子 policy 依据与 source 关联，builtin 无源原子）；显式缺失/越权/无效时抛结构化 MTP error |
| `patchouli.public.topic.list_active` | `list_active_topics` | `identity_scope`、`include_empty` | `tuple[TopicSnapshot, ...]` |
| `patchouli.public.topic.get_data` | `get_topic_data` | `identity_scope`、topic id | `TopicData | None`（越域目标隐藏） |
| `patchouli.public.manual_settle_topic` | `settle_topic` | `identity_scope`、`topic_id` | `TopicSettleResult` |
| `patchouli.public.evict_topic` | `evict_topic` | `identity_scope`、topic id | `TopicEvictionResult` |
| `patchouli.public.models.warmup` | `warmup_models` | 无 | `None` |
| `patchouli.public.models.ready` | `is_models_ready` | 无 | `bool` |

### 2.5 CPU 端口与交互输出

Alice 经 [CPU 端口](./subsystem-contracts.md#4-cpu-端口与-alice-实现)交接，不注册全局执行路由。执行过程中，Alice 经进程给出的事件 sink 推送 `token`、`mtp_start`、`mtp_result`、`sub_agent_start`、`sub_agent_end`、`context_compaction` 交互事件，结束时以返回值交回 `CPUExecutionResult`。每个交互事件带 run-local `stream_sequence`；frame/CALL 事件还带 `agent_run_id/frame_id/action_id` 等关联字段，类型化结果不附加交互元数据。

交互流的 Agent 展示优先使用当前 Profile 的来源 alias，空值才回退注册标签；`agent.run.*` 与 `chat.run.*` 观测中的 Agent/Workspace 则使用同一次注册的 `ExecutionLabels`。Profile 展示名不会改变授权身份或事件标签。

Alice 不缓存交互事件，背压由进程的输出通道施加：通道写满时 sink 等待，Alice 随之暂停。停止以原生取消到达 Alice，Alice 收尾后返回带部分输出与已完成检查点的 cancelled 结果。客户端的 `done` 由任务进程从结果投影（第 6 节），不属于 Alice 的内部事件。Alice 的资源操作经 workspace 的公共 `OperationEntry` 与请求契约，操作入口本身不注册全局路由。

## 3. 全局业务事件

当前 `GlobalEvents` 包含 canonical 失效与既有 PendingAtom 结算通知：

| Event | Publisher | Subscriber | Payload |
|:---|:---|:---|:---|
| `patchouli.events.memory.changed` | PatchouliBridge | workspace CacheInvalidator | `payload: MemoryChangeEvent`（`belong_to`、`memory_id`、`operation: upsert/patch/delete`） |
| `patchouli.events.pending_atom.settled` | PatchouliBridge | workspace WriteIntentRegistry | `settlement`（含 `pending_alias`、`intent_id`） |
| `patchouli.events.pending_atom.failed` | PatchouliBridge | workspace WriteIntentRegistry | `pending_alias`、`intent_id` |
| `patchouli.events.pending_atom.cancelled` | PatchouliBridge | workspace WriteIntentRegistry | `pending_alias`、`intent_id` |

中期库的 upsert、patch_payload、delete 与 delete_by_key 在提交调用的 finally 中各内联发布一次变更通知，包括 primary/secondary 失败与删除未命中；只改 `meta.lifecycle` 动态状态的 patch_payload 不发布；通知不证明提交成功。store 只依赖注入发布端口，本地发布器与 bridge 顺序 await 转发，写入返回前失效订阅者已执行。订阅者先清原子及 alias、清源 Profile，再推进 Workspace 代次，之后不回填值；普通订阅异常由总线记录，不反向改变存储结果。首版不补齐未送达或重试。

结算的权威发布方是 Patchouli，事件名使用发布方的命名空间，由 bridge 从 Patchouli 本地总线以同名转发到全局总线、交给 workspace 登记。三个事件都携带 `pending_alias` 与 `intent_id`：alias 永不复用，intent_id 再区分过时事件，登记只在两者都匹配时迁移状态。主动生成任务规范中的两者总是同时出现，缺一即构造失败。登记不回填 canonical 缓存，正式物化写入触发上述失效。

## 4. RuntimeEvent 观测契约

RuntimeEvent 不通过 `GlobalSystemBus` 发布，而通过独立 `RuntimeEventSink`。信封字段分为：

独立旁路的原因是观测不能参与业务控制。RuntimeEvent 需要支持缓冲、回放、慢订阅者隔离和 best-effort 投递，这些语义与业务 RPC 和所有权通知不同；如果把它们混在同一事件流中，UI 断连或观测 sink 失败就可能反向阻塞一次正常 chat，业务消费者也可能误把丢失的观测事件当成未发生的业务事实。

- 标识与排序：`event_id`、进程内 `sequence`、UTC `timestamp`；
- 追踪：`trace_id`、`span_name`、`task_type`；
- 来源：`source`、`subsystem`、`component`、`severity`；
- 关联 id：process（`process_id`，任务进程唯一标识）、agent run、task、agent、frame、topic、atom；
- 可选观测标签：`workspace_id`；
- 描述：`status`、`reason`、`message`、`data`。

`RuntimeEvent.workspace_id` 只帮助日志、UI 和运维按资源归属筛选观测，不参与 route、订阅、sequence、授权、幂等或缓存分区。Workspace 的完整身份与资源所有权仍由[身份与授权](../workspace/identity-and-authorization.md)和领域所有者解释。

当前事件组：

| 事件组 | 类型前缀或成员 |
|:---|:---|
| Chat | `chat.run.*` |
| Command | `command.executed` |
| Import Bus | `import.turn.accepted`、`import.turn.duplicate`、`import.turn.skipped`、`import.turn.deferred`；提交生命周期复用 `work.*` |
| Gateway | `gateway.workflow.*`、`gateway.step.completed`、`gateway.analysis.capability.completed` |
| Agent | `agent.run.*`、`agent.context.compaction`、`agent.context.budget_exceeded` |
| Memory Task | `memory.task.*` |
| Maintenance | `maintenance.task.*` |
| System lifecycle | `system.starting/ready/start_failed/shutting_down/stopped/stop_failed` |
| Subsystem operation | `subsystem.operation.*` |
| Stream continuity | `event.stream.gap` |

`system.stopped` 与 `system.stop_failed` 的 `data` 包含 `steps`、`completed_steps`、`scheduler_stopped` 与 `import_bus_shutdown`；后者是 Import Bus 关闭时等待的结果（`{"waited_requests": 关闭开始时的在途导入数}`），停止在它之前失败时为 None。

`import.turn.*` 由 Import Bus 在一次导入的结局处发布，`task_type` 为 `background`（导入没有用户在 HiveMemory 一侧等待），`agent_id` 与 `workspace_id` 取导入授权的发起者与目标 Workspace，只作观测标签。`data` 只携带 `source`、`external_conversation_id`、`external_turn_id`、`interaction_id` 与事件数量 `event_count`，不携带用户消息、回复、思考、工具参数或工具结果的正文：`accepted` 在交互应用完成后发布，信封 `topic_id` 是实际落位的话题，`data` 另带 `outcome`、`is_new_topic` 与 `worth_saving`；`deferred` 是可重试或应用失败的推迟，`severity=warning`。关闭中拒绝（`shutting_down`）、认证与授权失败以及原样传播的契约错误与程序错误不发布 `import.turn.*`。

### 4.1 投递语义

- `RuntimeEventBus.emit()` 尽力而为，内部失败只记录 warning；
- 每次 emit 会覆盖 timestamp 并分配递增 sequence；
- 总线保留有界 ring buffer，支持 replay last 或从 `last_event_id` 后回放；
- 找不到客户端提供的 `last_event_id` 时，先生成 `event.stream.gap`；
- 每个订阅者队列有界，满时丢弃最旧事件；
- RuntimeEvent 不承诺跨进程连续、持久化或 exactly-once。

### 4.2 与 Agent 交互输出流的边界

Alice CPU 端口的交互输出与 `/runtime-events/stream` 是两条独立通道：

- 交互输出只属于一次 Agent run，承载 token、MTP、CALL 边界、压缩进度，执行结果以返回值交回；进程的输出通道写满时通过背压等待，交付方离开会以停止请求取消该 run；
- RuntimeEvent 是全局扁平观测流，承载 `agent.run.started/completed/cancelled/failed` 等生命周期摘要，允许缓冲、回放和慢订阅者丢弃旧事件；
- 两条流可以通过 `process_id/agent_run_id` 关联展示，但不得自动互相桥接；
- RuntimeEvent 的缺失或 transport 故障不能改变 Agent 结果，交互输出也不替代结构化 `TurnEvent` 与权威 run 状态。

### 4.3 上下文预算观测

`agent.context.compaction` 在压缩开始/结束记录与交互进度相同的统计（第 6.1 节），外层 `status` 表示阶段；结束可带 `reason`（摘要拒绝、超时或预算失败的稳定标识）。`agent.context.budget_exceeded` 在根或 CALL 子 frame 被轮内守卫拦截时发布：`status=blocked`、`severity=warning`、`reason=turn_context_budget_exceeded`，`data` 仅含 `estimated_tokens` 与 `input_token_limit`，关联字段包含 `frame_id`。两者均关联本次 process/run，不携带摘要、历史或异常正文，不决定检查点接受与业务终态。

### 4.4 失败与降级观测

| 事件/字段 | 公开值与含义 |
|:---|:---|
| `import.turn.deferred.reason` | `gateway_unavailable`（Gateway 整体抛出可恢复异常）、`queue_unavailable`（提交队列容量已满或已停止）、`apply_pending`（等待应用超时）或 `apply_failed`（应用失败）；`data.error_class` 为异常类型名，等待类原因为 None |
| `import.turn.duplicate.reason` / `import.turn.skipped.reason` | 分别为 `interaction_already_accepted` 与 `no_output` |
| Gateway workflow failed 的 `reason` | `gateway_timeout`、`dependency_unavailable` 或 `workflow_failed`；message 为固定失败说明 |
| Gateway step 的 `data.fallback_reason` | `request_deadline_exceeded`、`step_timeout` 或 `dependency_unavailable`；无 fallback 为 None |
| 查询分析 capability 的 `data.error` | 失败为 `query_understanding_failed`，成功为 None |
| Work Queue 接纳观测的 `reason` | `unknown_lane`、`runtime_stopped`、`unknown_payload_codec`、`capacity_rejected`、`duplicate_work_item`；未确认接纳为 `admission_unknown` |

`data.error_class` 只作 Python 类型诊断，不是稳定机器分类或重试依据；上述事件不包含异常正文。`WORK_REJECTED` 的未知原因只说明 enqueue 没有确认成功，不证明 work 没有入库；调用方仍以返回值、异常和 Store 记录为准。编码失败在接纳前，已接纳 payload 解码失败仍进入 dead letter，执行重试仅由 lane 的 classifier 决定。

API 请求错误共用 `ErrorResponse`，校验 422 的 detail 是字符串、字段说明在 issues；SSE 开始后使用独立的流内 code/message 投影，保持事件名和终态顺序。结构与迁移见[错误模型](./error-model.md#7-http-api-与客户端呈现)。

## 5. `SystemEvent` 的状态

`SystemEvent` / `SystemEventType` 是保留的冻结生命周期模型，当前有契约测试，但实际系统生命周期观测使用 `RuntimeEventType.SYSTEM_*`。在新的生产者出现前，不应把 `SystemEvent` 推断为正在运行的第二套事件流。

## 6. HTTP 入口的身份交接与管理读取边界

本节固化 HTTP 身份选择、会话管理、Chat SSE 与导入入口的契约；身份选择转换为进入上述 route 的 `IdentityScope`，因为它是所有总线调用共享的前置契约。完整的认证与授权模型见[身份与授权](../workspace/identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)第 2 节。

- 用户导向身份选择为 `user_id + workspace_id` 基础选择，Agent action（Chat、导入）由请求体附加具体 `agent_id`；基础选择只经统一请求头 `x-user-id`/`x-workspace-id` 承载，Chat/stop、导入请求体与 Topic、会话等管理路由都不接受 body 或 query 中的 user 与 Workspace 字段。
- `server/deps.py resolve_request_identity_claims` 是唯一解析入口，只产出身份声明（actor 声明与请求进入的 Workspace），不组装 `IdentityScope`；只含空白的身份请求头拒绝（400）；未知 Workspace 拒绝（404）；Agent action 缺失具体 `agent_id` 或显式使用保留的 `system` 返回 400；非 Agent action 使用保留 `SYSTEM_AGENT_ID = "system"`（"没有具体 Agent 作为操作来源主体"，不得成为 `MemoryAccessPolicy` target）。
- 声明交给统一认证网关取得访问 context：Chat 由任务进程的注册入口认证并创建进程；管理操作、`/chat/stop` 与导入取得绑定本次请求的请求级 context，请求结束时撤销。进入上述 route 的 `IdentityScope` 由授权点在操作授权后组装。
- Chat 请求体的 `session_id` 绑定会话资源，缺省自动创建；不进入身份构造，finalize 关联仍使用进程 ID；`ActorIdentity` 仅包含 `user_id`、`agent_id` 与可选 `team_id`。
- `/chat/stop` 不是 Agent action：注册入口经进程控制授权比对请求方与进程记录的驻留坐标，不可控与不存在统一按 `not_found` 返回，取消与事件发布使用进程注册时绑定的观测标签，不从当前选择重新构造身份。
- 导入以 (user, 具体 Agent) 声明经 server 的 principal 与 `http` adapter 认证；Import Bus 服务以导入授权组装 `IdentityScope`：只校验准入与目标，不检查 actor 操作白名单，拒绝保留的 `system`（[身份与授权](../workspace/identity-and-authorization.md#4-操作授权)）。认证前不组装 scope，没有例外入口。
- 管理读取（Memory/Agent Profile/Topic 管理）按 owner-management 语义执行：在 Workspace ownership hard boundary 通过后可读取该 Workspace 的 `PUBLIC/PRIVATE/TEAM` 全部 Memory，不执行 Agent 可见性过滤；Agent retrieval 仍按 `MemoryAccessPolicy` 过滤，`system` 不承担权限绕过语义。
- Memory task list 按快照的必需 `belong_to` 过滤；get/cancel 对越域与不存在统一抛 `ResourceNotFoundError`，HTTP 统一映射为 404，任务 ID 不授予观察或取消能力。
- 响应 DTO 中的 `user_id`（如 `MemoryResponse.user_id`）保留为对外 owner 展示兼容字段，来源是 `workspace_identity.owner_user_id`；前端不得把它反推为下一次 actor 选择。

### 6.1 会话管理与 Chat SSE

会话 HTTP 由 `server/routers/sessions.py` 经门面取得 `RootSessionService`，沿用请求认证与 root 授权。POST/GET `/api/v1/sessions` 创建（201）/列出摘要；GET/PATCH/DELETE `/api/v1/sessions/{session_id}` 读取全量历史/以 `{title}` 重命名/删除（204）。创建体同为 `{title}`，允许空标题，首轮写入时自动取用户消息前 80 字。标题最多 200 字。会话不属于 Agent，不增加 actor 操作。响应是 `server/models/session.py` 的投影，不含 Workspace 归属：摘要为 `session_id`、`title`、`created_at`、`updated_at`、`turn_count` 与 `active_process_id`（非空表示正被任务进程占用）；完整读取另含 `turns` 与可空的 `compaction_checkpoint`（结构见[子系统契约](./subsystem-contracts.md#421-会话压缩检查点)）。每轮只返回 `interaction_id`、`agent_id`（仅用于历史展示）、`topic_id`、`created_at` 与载荷中的 `outcome`、`user_message`、`assistant_final_text`、`turn_events`；轮次事件只含 `kind`、`sequence`、`content`、`action_id`、`tool_kind`、`target`、`tool_args`、`status`。列表摘要不含轮次或检查点。

Chat SSE 订阅任务进程的输出通道：进程在自己的 task 中运行，客户端断连时交付以 `client_disconnected` 退订并请求停止，与用户停止走同一条路径，发布 `chat.run.cancel_requested` 与带生效阶段的 `chat.run.cancelled`，本轮输出被丢弃；进入 finalize 后的断连不影响提交。响应不等待进程关闭，详见[任务进程](../workspace/task-process.md#111-流式交付)。

Chat 首个 `process_id` SSE 事件携带 `{process_id, session_id}`；非流式交付的结果也含 `session_id`，但不含 `topic_id`、停止原因与记忆引用。`memory_refs` 保留；不再发布 `topic_info`。有输出的 completed/cancelled/failed 的 done 包含 `topic_id`，对应会话本轮标注；没有 `pool_topics`。无输出的 done 不含 `final_text` 与 `topic_id`。前端通过独立 Topic 列表接口管理短期话题。取消、失败有输出时先 finalize 再交付终态，详见[任务进程](../workspace/task-process.md#8-封口部分输出与结算)。

failed done 保留 `reason=agent_run_failed`，CPU 有稳定原因时另带 `failure_reason`；`chat.run.failed.message` 使用该码，不下发异常正文。四类预算原因见[错误模型](./error-model.md#22-agent-run)。done 不包含 `turn_events` 或 `compaction_checkpoint`；检查点以会话读取结果为准，压缩结束事件不代表会话已接受。

Alice 在创建根 frame 前可推送 `context_compaction`，经 CPU 事件 sink 与任务进程原样交付；它与其他交互事件一样只受通道背压，不等待交付方确认：

| 字段 | 含义 |
|:---|:---|
| `status` | `started`、`completed`、`abandoned`（放弃摘要但视图仍可用）或 `failed` |
| `layer` | `reduction` 或 `summary` |
| `tokens_before` / `tokens_after` | 完整视图的估算 token；开始时相等 |
| `covered_turns` | 当前减重边界累计覆盖轮数 |
| `summarized_turns` | 结束时的累计摘要覆盖轮数 |
| `summary_status` | 结束时为 `not_needed`、`completed` 或 `abandoned` |
| `elapsed_ms` | 结束时的压缩耗时 |

事件沿用 run 的 `stream_sequence`，开始时 `layer=reduction`、覆盖轮数为 0；正常完成压缩产出开始和结束，取消或程序异常可只有开始。它不进入 `turn_events`，不构成本轮输出，也不单独触发封口。非流式交付丢弃交互进度事件，上述 RuntimeEvent 照常发布。

### 6.2 Import Bus 导入

`POST /api/v1/import/turns`（`server/routers/import_bus.py`）接收外部 harness 一轮已经结束的交互的完整记录，交给 System 的 Import Bus；不返回记忆上下文。一次请求只对应一轮，不跨请求累积。请求模型拒绝未知字段：

| 字段 | 说明 |
|:---|:---|
| `source` | 来源标识，1–64 个小写字母、数字或连字符，例如 `claude-code` |
| `external_conversation_id`、`external_turn_id` | harness 的会话标识与该轮标识（各 1–256 字符），同一轮重发时不变 |
| `agent_id` | 连接器代表的具体 Agent，只作为认证声明，不进入导入记录 |
| `outcome` | `completed`、`cancelled` 或 `failed` |
| `user_message` | 该轮的用户消息；服务端写成序号为 0 的 `user_message` 事件 |
| `events` | 用户消息之后的事件，`sequence` 从 1 起严格递增；`kind` 为 `assistant_message`、`thought`、`tool_call` 或 `tool_result`，另有 `content`，工具事件可带 `action_id`、`tool_name`、`tool_args`、`status`（出错为 `error`）。不接受 `role`、`tool_kind`、`target`：角色由服务端按种类填写，外部工具不设 MTP 动词 |
| `assistant_final_text`、`model_used` | 该轮最终的回复正文与实际使用的模型名，可为空 |
| `recent_turns` | 该轮之前最近的若干轮（`user_message`、`assistant_text`、可选的 `external_turn_id`），只供 Gateway 落位，不进入交互载荷 |

请求体上限 `import_bus.max_request_bytes`（默认 8 MiB）在框架读取并解析请求体之前判定：声明的 Content-Length 超出时不读取请求体，没有声明时边读边计数，超出即以 413 拒绝，过大且不是合法 JSON 的请求也是 413 而不是 422。

响应为 200，`status` 为 `accepted`（本次提交且已应用）、`duplicate`（此前已应用）或 `skipped`（用户消息之外没有输出，不提交），并带推导出的 `interaction_id`；不暴露 Gateway 决定、话题或记忆上下文。确认只在交互应用完成之后给出（等待时限 `import_bus.apply_wait_seconds`，默认 30 秒）。失败经错误信封区分是否可重试：

| 情况 | 状态码 | `error` / `reason` | 可重试 |
|:---|:---:|:---|:---:|
| Agent 缺失或为 `system`、身份请求头只含空白 | 400 | `request.agent_required` / `request.agent_reserved` / `request.identity_empty` | 否 |
| 未知 Workspace | 404 | `request.workspace_not_found` | 否 |
| 未获准入（接入或 Workspace 准入失败） | 403 | `workspace.admission_denied` | 否 |
| 请求体超过上限 | 413 | `request.body_too_large` | 否 |
| 请求校验失败 | 422 | `request.validation_failed` | 否 |
| 认证网关已关闭（服务关闭中） | 503 | `workspace.admission_denied` / `authentication_gateway_closed` | 是 |
| Gateway 暂不可用、提交队列拥塞、等待应用超时、导入服务关闭中 | 503 | `import_bus.unavailable` / `gateway_unavailable`、`queue_unavailable`、`apply_pending`、`shutting_down` | 是；`apply_pending` 重发时只继续等待，不重复提交 |
| 已提交的交互在应用时失败 | 500 | `import_bus.apply_failed` | 有限次；重发时重新路由并以同一 `interaction_id` 再次提交 |
| 其他未分类错误 | 500 | `server.internal_error` | 有限次 |

分类与原因码的归属见[错误模型](./error-model.md#43-import-bus)，编排见 [Import Bus](../system/import-bus.md)。

## 7. 设计矛盾检查

评审新的 route 或 event 时，应检查：

1. 调用方是否必须获得确定结果？如果是，为什么选择不会传播 subscriber 失败的 Pub/Sub？
2. 这是否只是观测信息？如果是，为什么要进入会影响业务顺序的 GlobalSystemBus，而不是 RuntimeEventSink？
3. route、handler 参数名和公共模型是否作为一个整体更新，还是出现了代码能注册、调用方却按旧语义传参的情况？
4. 新事件的发布者是否真正拥有被声明的事实，订阅者是否只维护投影而没有取得权威所有权？
5. 外部模块是否开始引用 local route，或为了绕过公共模型而直接持有另一个 Runtime？
6. 实现是否依赖未承诺的持久化、自动重试、跨进程连续或 exactly-once？
7. RuntimeEvent 丢失、重复或 sink 失败时，业务结果是否仍然完全由返回值、异常和权威状态决定？

## 7. 变更规则

- 新公开 route 在提供方 `contracts` 中以 `@route(name)` 声明名称与异步签名，输入输出引用公共契约；所有调用方直接导入该声明，绑定表用 `RouteBinding.of` 构造；
- route handler 的参数名也是调用契约的一部分；
- 新跨子系统通知应先判断需要 RPC 返回值还是 Pub/Sub；
- 新观测事件加入 `RuntimeEventType`，不得通过 `data` 中的临时字符串模拟事件类型；
- 删除或重命名 route/event 前必须更新所有注册方、调用方、测试和本文；
- local route 不记录在本规范中，也不得被外部模块直接引用。

主要验证入口：`scripts/typecheck_bus_routes.py`、`tests/unit/system/contracts/test_public_routes.py`、`tests/unit/components/events/test_bus.py`、`tests/unit/components/bus/test_async_bus.py`；导入入口与 `import.turn.*`：`tests/unit/server/routers/test_import_bus.py`、`tests/unit/system/services/import_bus/test_ingestor.py`。
