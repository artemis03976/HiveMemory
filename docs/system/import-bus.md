---
title: Import Bus
status: current
owner: system
scope: external-harness-complete-turn-import-and-claude-code-connector
code_paths:
  - src/hivememory/server/routers/import_bus.py
  - src/hivememory/server/models/import_bus.py
  - src/hivememory/system/application/import_bus_service.py
  - src/hivememory/system/services/import_bus/
  - src/hivememory/config/import_bus.py
  - src/hivememory/core/protocol/sealing.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/patchouli/contracts/public_routes.py
  - src/hivememory/patchouli/control/interaction_submission.py
  - integrations/claude-code/
  - integrations/.claude-plugin/marketplace.json
  - configs/workspace_actors.yaml
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/authentication.md
  - docs/workspace/task-process.md
  - docs/gateway/analysis.md
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/components/runtime-and-bus.md
  - docs/help/setup.md
related_plans:
  - docs/archive/plans/v0.7.1-import-bus-redesign.md
last_reviewed: 2026-10-27
---

# Import Bus

Import Bus 是外部 harness 对话回流 HiveMemory 的入口。harness 中的一轮交互结束后，连接器把这一轮的完整记录（用户消息、助手正文、思考、工具调用与结果）一次交给 Import Bus；Import Bus 经 Gateway 取得话题落位与价值信号，用与任务进程相同的封口函数组装交互记录，经 Patchouli 公开路由提交，并在感知应用完成后才确认。第一个接入的 harness 是 Claude Code，连接器以插件形式位于 `integrations/claude-code/`（第 8 节）。

Import Bus 只向记忆网络输入：它不检索记忆，也不向调用方返回记忆上下文，不运行 Alice、MTP、命令或回复生成，不经任务进程的注册入口，导入的轮次也不写入 ConversationSession（plugin 模式下会话归外部 harness 所有）。为话题落位读取候选话题属于 Gateway 的分析，不算读取记忆。Gateway 的调用模式仍名为 `PASSIVE_MEMORY`。

设计取舍的完整记录见[已归档的 Import Bus 重设计计划](../archive/plans/v0.7.1-import-bus-redesign.md)（历史材料），尚未实施的后续方向（流式监听、历史导入、第二个 harness）见 [Import Bus Idea](../ideas/import-bus.md)。

## 1. 为什么一次接收一整轮

- **不返回记忆上下文，提前路由没有收益**：Import Bus 不在用户消息到达时返回任何上下文，在轮次中途路由只剩时序成本——harness 的 hook 只在少数节点触发，响应很快时模型的首条消息可能先于路由完成到达。在轮次结束处一次接收完整记录，Gateway 可以照常路由，服务端也不需要跨请求累积事件、按空闲或下一条用户消息封口。
- **载荷与主动链路同形**：导入的交互记录与任务进程封口的交互记录由同一个函数产出，Patchouli 不区分来源即可感知、折叠与生成。
- **确认等于已应用**：连接器只在得到确认后前进游标。若在提交被队列接纳时就确认，应用失败（例如落位的话题在应用前已被驱逐或结算）发生在确认之后，连接器不会重发，这一轮就丢失了；因此 Import Bus 等待感知应用完成才确认，应用失败时交回可重发的错误。

## 2. 边界与组成

| 部分 | 位置 | 负责 | 不负责 |
|:---|:---|:---|:---|
| HTTP 入口 | `server/routers/import_bus.py`、`server/models/import_bus.py` | 请求体上限、请求校验、身份声明解析、经统一认证网关取得请求级 context、错误映射 | 授权、编排 |
| 应用服务 `ImportBusService` | `system/application/import_bus_service.py` | 以导入授权把 context 兑现为本次请求的 `IdentityScope` 与 principal 标识，委托编排器；关闭时转交 | 跨请求的业务状态 |
| 编排器 `ImportBusIngestor` | `system/services/import_bus/` | 按外部会话串行、推导 `interaction_id`、进度查询、Gateway 落位、封口、提交、等待应用、观测事件、关闭时等待在途请求 | 持有提交队列、注册维护任务、检索、记忆存储 |
| 封口函数 `seal_interaction` | `core/protocol/sealing.py` | 两条链路共用：组装 `InteractionPayload`，由 core 归约器得到 MTP 轨迹 | — |
| Patchouli | 公开路由 `patchouli.public.interaction.submit` / `.status` / `.wait` | 提交队列、查重、应用、新话题的建立 | 导入的身份与重试策略 |
| Claude Code 连接器 | `integrations/claude-code/` | 读取会话记录、切分轮次、游标与重发 | 服务端的查重与授权 |

编排器的状态只有串行门与在途计数，都只在进程内。它经 `GlobalSystemBus` 调用 Gateway 与 Patchouli 的公开路由，不持有任何子系统的 Runtime 或 Patchouli 的提交队列；组合根只注入全局总线、Gateway 请求时限与最近轮数（沿用 Gateway 配置）、等待应用的时限与观测 sink，导入授权使用与其他入口相同的操作授权者。

## 3. HTTP 入口

`POST /api/v1/import/turns`，请求体是 harness 一轮已结束交互的完整记录。请求模型拒绝未知字段：

| 字段 | 说明 |
|:---|:---|
| `source` | 来源标识，1–64 位小写字母、数字与连字符，例如 `claude-code` |
| `external_conversation_id` | harness 的会话标识 |
| `external_turn_id` | 该轮在 harness 中的标识，同一轮重发时不变 |
| `agent_id` | 连接器所代表的具体 Agent，只作为认证声明，不进入导入记录 |
| `outcome` | `completed`、`cancelled` 或 `failed`，与 `InteractionOutcome` 相同 |
| `user_message` | 该轮的用户消息 |
| `events` | 用户消息之后的事件：`kind`（`assistant_message`、`thought`、`tool_call`、`tool_result`）、`sequence`（从 1 起严格递增）、`content`，工具事件另带 `action_id`、`tool_name`、`tool_args`、`status`（出错的结果为 `error`） |
| `assistant_final_text` | 该轮最终的自然语言回复，可为空 |
| `model_used` | 实际使用的模型名，可为空 |
| `recent_turns` | 该轮之前的若干轮（`user_message`、`assistant_text`、可选 `external_turn_id`），只供 Gateway 落位，不进入交互记录 |

- 事件不携带 `role`、`tool_kind`、`target` 与 `render_as`。服务端把 `user_message` 写成序号为 0 的 `user_message` 事件，再按事件种类填写角色（工具结果为 `system`，其余为 `assistant`）。`tool_kind` 只表示 MTP 动词，外部工具不设置，因此导入的工具调用不会产生 MTP 轨迹，也不会被误当作记忆操作；生成视图把这类动作按思考、工具名、参数摘要与截断后的结果渲染（[生成](../patchouli/generation.md)）。
- 序号重复或回退会让动作与回复错位，请求以 422 拒绝。
- 请求体上限 `import_bus.max_request_bytes` 在框架读取并解析请求体之前判定（自定义路由类 `BodyLimitedRoute`）：声明的 `Content-Length` 超限时不读取请求体，没有声明长度时边读边计数，超限即停止。两种情况都返回 413（`request.body_too_large`），过大且不是合法 JSON 的请求因此也是 413 而不是 422。

成功响应为 200，带推导出的 `interaction_id` 与 `status`：

| `status` | 含义 |
|:---|:---|
| `accepted` | 本次提交，且已经应用 |
| `duplicate` | 此前已经提交，现已应用（含本次请求等到上一次提交应用完成） |
| `skipped` | 该轮没有用户消息之外的输出（既无事件也无最终正文），与任务进程一样不封口 |

响应不暴露 Gateway 决定、话题或记忆上下文。失败的状态码与可重试性见第 5 节。

## 4. 认证、授权与编排

### 4.1 认证与导入授权

导入是 Agent action：server 用 `resolve_request_identity_claims(require_agent=True)` 把请求头的用户选择与请求体的 `agent_id` 解析为 (user, 具体 Agent) 声明，缺失 `agent_id` 返回 400（`request.agent_required`），显式给出保留的 `system` 返回 400（`request.agent_reserved`）；`X-Workspace-ID` 只接受 `main_workspace`，其他值返回 404。随后以 server 自身的 principal 与 `http` adapter 经统一认证网关取得请求级 context（`request_access_for_claims`），请求结束时撤销；认证前不组装 `IdentityScope`，也不为连接器另行登记 principal。请求头中的用户身份不做证明，沿用本地单用户部署的信任假设。

workspace 授权点的 `authorize_import` 只校验 context 有效、已准入且目标等于驻留 Workspace，不检查 actor 操作白名单，拒绝保留的 `system` 作为发起者（`OperationDeniedError`，reason `agent_required`），返回本次请求的 `IdentityScope` 与认证通过的 principal 标识。不检查白名单的理由：导入提交的是已经在外部发生的交互，与任务进程封口后提交交互记录同类，不调用任何 actor 操作，因此白名单为空的 Agent 也能导入。`authorize_import` 的生产调用位置由架构测试锁定为 `system/application/import_bus_service.py`。

`IdentityScope` 只在这一次导入的调用链内经公共路由到达 Gateway 与 Patchouli，不写入记录、事件或后台任务；Patchouli 在提交边界把它拆为归属（目标 Workspace）与发起者（认证通过的 user 与 Agent）。访问模型见[身份与授权](../workspace/identity-and-authorization.md)，访问登记的匹配规则见[认证与访问登记](../workspace/authentication.md#2-两类访问登记)。随仓库发布的 `configs/workspace_actors.yaml` 为 `default` 用户在 `main_workspace` 登记了 `claude_code` 的精确记录，`allowed_operations` 为空：精确记录优先于用户级记录，这个 Agent 只能准入，不继承用户级的六项 actor 操作。

### 4.2 编排顺序

```text
关闭中？ -> 503 shutting_down
推导 interaction_id
持串行门（user, Agent, source, external_conversation_id）
  关闭中？ -> 503 shutting_down
  patchouli.public.interaction.status
    succeeded / unknown -> duplicate
    pending             -> 等待应用（第 4.5 节）-> duplicate
    failed / 未提交：
      未提交且没有输出   -> skipped
      Gateway PASSIVE_MEMORY（附最近轮次与上一轮的落位）
      seal_interaction -> InteractionPayload
      patchouli.public.interaction.submit（origin import_bus，新话题附标题与摘要）
      等待应用（第 4.5 节）-> accepted
```

### 4.3 交互标识与串行

`interaction_id` 是 `import_` 加上对（principal、user、Agent、`source`、`external_conversation_id`、`external_turn_id`）规范化 JSON 序列取 SHA-256 的前 32 位十六进制。同一轮重发得到同一标识，队列据此幂等；当前认证方式下 principal 恒为 server，纳入认证通过的 actor 使代表不同用户或 Agent 的连接器不会撞号；前缀把它与任务进程的 `process_id` 及其他来源的标识分开。

编排器按（user、Agent、`source`、`external_conversation_id`）持 `KeyedSerialGate`（[运行时与总线](../components/runtime-and-bus.md#5-keyedserialgate)），门内依次完成进度查询、Gateway、提交与等待应用。同一外部会话的并发请求（例如两个 hook 同时触发补发）因此按到达顺序处理，后到的请求看到前一个的结果；串行门持有到应用完成，同一外部会话的下一轮路由时，上一轮新建的话题已经建立，可以成为候选与续接的落点。不同外部会话互不阻塞。串行键只用于进程内排序，不参与授权。

### 4.4 进度查询与重发

查重先于路由，以 Patchouli 的交互状态查询 `patchouli.public.interaction.status` 为准，它返回 `InteractionStatus | None`：未被接纳时为空；接纳后为 `pending`、`succeeded`（附真实落点，`NEW_TOPIC` 已解析）、`failed` 或 `unknown`（接纳过但结局已超出队列的保留窗口）。

| 进度 | 处理 |
|:---|:---|
| `succeeded` 或 `unknown` | 直接返回 `duplicate`，不调用 Gateway |
| `pending` | 只继续等待应用，不再路由与提交 |
| `failed` | 重新路由，以同一 `interaction_id` 再次提交 |
| 未被接纳 | 没有输出时返回 `skipped`，否则路由并提交 |

已应用与仍在应用的轮次不再调用 Gateway：Gateway 的结果写入载荷，重新路由可能得到不同载荷而被队列判为冲突。应用失败的轮次是例外：提交队列允许终态未成功的交互以同一标识重新提交，载荷可以不同（重新路由后的新落点），新提交以递增的尝试号作为工作项标识；等待中与已成功的交互仍按原规则幂等（[运行时与总线](../components/runtime-and-bus.md#33-interaction-submission)）。

### 4.5 Gateway 落位、封口与提交

- **Gateway**：以 `PASSIVE_MEMORY` 模式调用 `gateway.public.process`，消息为该轮的用户消息，时限为 `gateway.workflow.default_request_timeout_ms`。`session_turns` 取请求中 `recent_turns` 的最后 `gateway.context_preparation.session_turn_limit` 轮；最近一轮带 `external_turn_id` 时，编排器推导它的 `interaction_id`、经状态查询取得应用成功后的真实落点，写入该轮的 `topic_id`。话题路由失败时，Gateway 续接上一轮仍在候选中的话题，否则按新话题兜底（[Gateway 分析](../gateway/analysis.md)）。Import Bus 只使用决定中的 `target_topic_id`、`new_topic_title`、`new_topic_summary`、`rewritten_query` 与 `worth_saving`，忽略检索计划，不调用检索。`worth_saving` 只是写入预判，不用于丢弃交互。
- **封口**：以用户消息、Gateway 决定、`assistant_final_text`、补上用户消息事件后的轮次事件、`outcome` 与 `model_used` 调用 `seal_interaction`；Import Bus 不携带附件，`used_attachments` 为空。
- **提交**：调用 `patchouli.public.interaction.submit`，`origin` 为 `import_bus`；目标为 `NEW_TOPIC` 时附上 Gateway 给出的 `new_topic_title` 与 `new_topic_summary`，感知以它们建立新话题。后续落位依赖候选话题的标题与摘要，缺少它们的“新建话题”会让持续导入的落位质量下降。提交按话题生成排序键，导入与主动链路落到同一话题时由同一个排序键保证先后。

### 4.6 等待应用与确认

提交被接纳后，编排器在串行门内调用 `patchouli.public.interaction.wait`，最多等待 `import_bus.apply_wait_seconds`：

- `succeeded`：发出 `import.turn.accepted`，返回 `accepted`；
- 仍为 `pending`（等待超时）：返回 503（`import_bus.unavailable`，reason `apply_pending`）；重发时查到仍在应用，只继续等待；
- `failed`：返回 500（`import_bus.apply_failed`）；重发时重新路由并再次提交。

等待的代价是每次导入多等一次应用。话题折叠不在应用内：触发折叠的一轮写入后即确认；同一话题的下一轮在提交队列中等待折叠结束，超过 `apply_wait_seconds` 时以 503 `apply_pending` 交回，重发时只继续等待（[感知与短期话题](../patchouli/perception.md#5-page-folding)第 5 节）。已接纳的交互在等待时查不到属于契约违约，以 `ImportBusContractError` 原样传播。

## 5. 失败分类与 HTTP 映射

错误按是否可重试区分，连接器据此决定游标（第 8.5 节）：

| 情况 | HTTP | 错误码 / reason | 连接器的处理 |
|:---|:---:|:---|:---|
| 请求校验失败（未知字段、序号不递增等） | 422 | `request.validation_failed` | 写入拒绝记录，前进 |
| 请求体超过上限 | 413 | `request.body_too_large` | 写入拒绝记录，前进 |
| 缺少 `agent_id` 或为 `system` | 400 | `request.agent_required` / `request.agent_reserved` | 保留游标，停止（配置问题） |
| `X-Workspace-ID` 不是 `main_workspace` | 404 | `request.workspace_not_found` | 保留游标，停止 |
| 未获准入 | 403 | `workspace.admission_denied`（如 `actor_not_admitted`） | 保留游标，停止 |
| 认证网关已关闭 | 503 | `workspace.admission_denied` / `authentication_gateway_closed` | 保留游标，稍后重发 |
| Gateway 整体抛出可恢复异常 | 503 | `import_bus.unavailable` / `gateway_unavailable` | 保留游标，稍后重发 |
| 提交队列容量已满或已停止 | 503 | `import_bus.unavailable` / `queue_unavailable` | 保留游标，稍后重发 |
| 已提交、等待应用超时 | 503 | `import_bus.unavailable` / `apply_pending` | 保留游标，重发时继续等待 |
| 编排器已开始关闭 | 503 | `import_bus.unavailable` / `shutting_down` | 保留游标，稍后重发 |
| 已提交、应用失败 | 500 | `import_bus.apply_failed` | 有限次重发，重发时重新路由并再次提交 |
| 契约违约、程序错误 | 500 | `server.internal_error` | 有限次重发 |

- Gateway 各决策步骤的失败由 Gateway 自身兜底，照常返回一个决定，导入按兜底结果继续。只有 Gateway 整体抛出的超时与明确的只读依赖故障（`TimeoutError`、`GatewayTimeoutError`、`StorageOfflineError`、`StorageReadError`、`ResourceUnavailableError`）才转为 `gateway_unavailable`：导入没有用户在等待，稍后重发可以得到正常的结果。未知异常、授权拒绝与取消原样传播。
- Patchouli 提交路由在队列容量已满或已停止时抛出 `InteractionAdmissionUnavailableError`，Import Bus 转为 `queue_unavailable`；队列没有接管这次提交，以同一 `interaction_id` 重发不会形成重复交互。
- `PASSIVE_MEMORY` 返回命令结果视为契约错误（`ImportBusContractError`），不转为可重试错误。
- 503 的 `reason` 只在白名单内公开（`gateway_unavailable`、`queue_unavailable`、`apply_pending`、`shutting_down`）；异常正文不进入响应。公共错误模型见[错误模型](../contracts/error-model.md)。

## 6. 关闭与观测

### 6.1 关闭

System 停止时（[组合根与生命周期](./composition.md#3-停止顺序)）：

```text
ActorAuthenticationGateway.close      新的导入请求以 503 拒绝
  -> GlobalMaintenanceScheduler.stop
  -> ImportBusService.shutdown        等待在途导入结束
  -> ……
  -> Patchouli.stop                   排空提交队列
```

编排器关闭时先标记关闭：认证网关关闭之前已通过认证、在关闭后才到达编排器的请求，以及在串行门外等待的请求，都以 503 `shutting_down` 拒绝（不发出 `import.turn.deferred`）；每个会话正在门内处理的请求继续完成，包括等待应用的时间，受 Gateway 请求时限与 `apply_wait_seconds` 约束。提交队列此时仍在运行，已提交的轮次因此能得到确认；Import Bus 不排空队列，队列由 Patchouli 停止时排空。`shutdown()` 返回 `{"waited_requests": n}`（开始关闭时的在途请求数），System 把它放进 `system.stopped` 与 `system.stop_failed` 的 `import_bus_shutdown` 字段。Import Bus 没有维护任务，不向调度器注册，也没有启动步骤。

### 6.2 运行时事件

编排器经 `ImportBusEventEmitter` 发出 best-effort 的 RuntimeEvent（subsystem `system`，component `import_bus`，`task_type` 为 `background`：导入没有用户在 HiveMemory 一侧等待）。事件只携带 `source`、`external_conversation_id`、`external_turn_id`、`interaction_id`、事件数量与结构化标签，`agent_id` 与 `workspace_id` 只是观测标签；不携带用户消息、回复、思考、工具参数或工具结果的正文。脱敏规则集中在发布器一处。

| 事件 | 时机 | 附加内容 |
|:---|:---|:---|
| `import.turn.accepted` | 本次提交应用完成后 | `topic_id`（实际落位）、`outcome`、`is_new_topic`、`worth_saving` |
| `import.turn.duplicate` | 已应用的轮次重发，或重发等到上一次提交应用完成 | reason `interaction_already_accepted` |
| `import.turn.skipped` | 没有输出的轮次 | reason `no_output` |
| `import.turn.deferred` | 可重试或应用失败的推迟（severity `warning`） | reason `gateway_unavailable`、`queue_unavailable`、`apply_pending` 或 `apply_failed`；`error_class` 只有异常类型名 |

事件发布失败不改变导入结果，事件也不能替代 HTTP 响应决定业务成功（[路由与事件](../contracts/routes-and-events.md)）。

## 7. 配置

`HiveMemoryConfig.import_bus`（`config/import_bus.py` 的 `ImportBusConfig`，拒绝未知键）：

| 键 | 默认值 | 作用 |
|:---|---:|:---|
| `max_request_bytes` | `8388608`（8 MiB） | 单次导入请求体的字节上限，超出返回 413 |
| `apply_wait_seconds` | `30` | 提交后等待感知应用完成的时限，超时返回 503 `apply_pending` |

Gateway 请求时限（`gateway.workflow.default_request_timeout_ms`，默认 8000）与最近轮数（`gateway.context_preparation.session_turn_limit`，默认 3）沿用 Gateway 配置。随仓库发布的 `configs/config.yaml` 不含 `import_bus` 段，按默认值运行。连接器的 HTTP 时限固定为 60 秒，覆盖默认的 Gateway 时限与等待时限之和；调大这两项时需要保持在连接器时限之内，否则连接器先超时并按可重试错误重发，重发时服务端查到仍在应用而继续等待。

## 8. Claude Code 连接器

连接器不属于 HiveMemory 包：它只用 Python 标准库、不导入 `hivememory`，由 Claude Code 所在环境的 `python3` 直接运行。安装与配置的操作步骤见[安装与启动](../help/setup.md#9-接入-claude-code)。

### 8.1 组成

- `integrations/.claude-plugin/marketplace.json`：本地 marketplace `hivememory-local`，只列出插件 `hivememory-connector`（来源 `./claude-code`），供本机安装，不发布。
- `integrations/claude-code/.claude-plugin/plugin.json` 与 `hooks/hooks.json`：插件清单与四个命令 hook。
- `hivememory_hook.py`：启动脚本，把插件根目录放进导入路径后交给 `connector` 的命令行入口（`hook <事件>`、`sync --transcript <路径>`、`sync --all`）。
- `connector/core/`：与 harness 无关的公共骨架——配置、状态与游标、提交与结果分类、同步循环、日志；`connector/claude_code/`：Claude Code 会话记录的读取器与 hook 处理。接入第二个 harness 时只需新增读取器。
- `scripts/transcript_stats.py`：维护工具，只统计会话记录的字段名、类型、计数与固定标记，不输出正文，用于在 Claude Code 升级后核对记录结构。

### 8.2 hooks

hook 同步运行但只做本地工作（登记会话记录、写结束记录），网络提交交给脱离 Claude Code 的后台同步进程（独立会话，不继承标准输入输出）。hook 入口无论成败都以 0 退出、不向标准输出写内容，连接器的任何失败只进日志，HiveMemory 不可用时不影响 Claude Code。

| hook | 行为 |
|:---|:---|
| `Stop` | 把游标之后的最后一轮记为 `completed`，唤醒这份会话记录的同步 |
| `StopFailure` | 把最后一轮记为 `failed`，唤醒同步 |
| `SessionStart` | 登记会话记录并撤销已结束标记（恢复的会话继续写入同一份记录），唤醒全部已登记会话记录的同步，补发尚未提交的已结束轮次 |
| `SessionEnd` | 标记会话已结束，唤醒同步；末尾没有结束记录的一轮按 `cancelled` 提交 |

`Stop` 必须同步运行：它在下一条消息写入之前结束，“最后一轮”才确定是刚结束的那一轮。结束通知在游标之后找不到轮次时（该轮已按超时或会话结束判定并提交）被忽略并写入日志。用户中断时 Claude Code 不触发 `Stop`，被中断的一轮由中断标记、下一轮或会话结束判定。

### 8.3 轮次的切分与映射

会话记录的格式不是 Claude Code 公开承诺的接口，解析只集中在读取器中，不认识的条目、块与字段一律跳过：

- 只读取 `user` 与 `assistant` 条目，跳过 `isSidechain`（子代理）、`isMeta`、`isCompactSummary` 条目与 `system` 条目（压缩边界、API 错误等）；只读完整的行。
- 一轮从一条 `turnOrigin` 为 `human`、内容不是工具结果的用户条目开始，到下一条这样的条目之前为止；轮次标识取该条目的 `uuid`，外部会话标识取它的 `sessionId`（缺失时取会话记录的文件名）。`[Request interrupted by user...]` 标记立即结束一轮（结局 `cancelled`）。
- 助手内容块 `text` → `assistant_message`，`thinking` → `thought`（没有正文的跳过：较新的模型默认只在本地记录签名），`tool_use` → `tool_call`（`action_id` 为调用标识，`tool_args` 为输入）；用户条目中的 `tool_result` → `tool_result`（`is_error` 为真时 `status` 为 `error`）。同一条 API 消息按内容块分条增量写入，重复的块按消息标识与块位置去重，工具调用与结果按调用标识去重。
- `assistant_final_text` 按顺序拼接该轮全部助手正文，与 Alice 的最终正文一致，工具调用之间的叙述也进入记忆生成；`model_used` 取最后一条助手条目的模型（`<synthetic>` 除外）。
- `recent_turns` 取该轮之前最近三轮已确认轮次的用户消息、最终正文与 `external_turn_id`；正文各截断到 2000 字符，避免一轮过大的内容拖着后续轮次一起超过请求体上限。

### 8.4 结束判定与游标

状态按会话记录分文件保存：游标（下一次读取的位置，之前的轮次都已确认）、最后确认的轮次标识、最近确认轮次的起点、各轮的结束记录与连续服务器错误次数。多个 hook 进程与同步进程同时运行，每次读改写持该会话记录的状态锁并以替换方式原子写入；同一会话记录的同步另持同步锁，提交期间不持状态锁，hook 不会被同步阻塞。

- **结束的判定**：一轮在其后出现下一轮、写入中断标记、已有结束记录或会话已结束时才算结束；仍在进行的末尾一轮不提交。没有 `SessionEnd` 的崩溃会话，末尾超过 30 分钟没有新条目时按 `cancelled` 结束。
- **末尾一轮是否写完**：结束通知可能先于最后几行写入。末尾一轮已判定结束、但最后一条助手条目的 `stop_reason` 仍是 `tool_use`（尚未正常收尾）时，同步轮询等待会话记录写完（最多约 10 秒）；读取之后会话记录又有新内容时重新读取，不以读取时的旧内容提交；会话已结束或会话记录静止超过等待时限时按读到的内容提交。
- **结局先于提交写入**：结束记录写入状态后才提交，重发沿用原结局；同步与 hook 同时为同一轮写结局时，先写入的生效，提交以状态中保存的为准。
- **首次见到的会话记录**：首个条目晚于连接器首次运行时间的从头读取；更早的（接入之前已有、后来被恢复的会话）游标放在当前末尾，不回填。
- **游标失效**：游标不在行首，或最后确认的一轮不再从记录的位置开始（会话记录被重写或缩短）时，从头扫描，在最后一个已确认轮次之后继续；找不到该轮时停止处理这份会话记录并写入拒绝记录，不重发整个会话。会话记录被删除时移除它的状态，不写拒绝记录。

### 8.5 提交结果的处理

按顺序提交，确认之后游标才前进：

| 结果 | 处理 |
|:---|:---|
| 2xx（`accepted`、`duplicate`、`skipped`） | 前进 |
| 网络错误、429、502、503、504 | 保留游标，停止处理这份会话记录，等下次唤醒 |
| 413、422（这一轮内容本身的问题） | 写入拒绝记录，前进 |
| 其他 4xx（身份声明、准入、地址或版本不对） | 保留游标，停止处理：这是连接器配置的问题，前进会让后续轮次全部丢失，修正配置后补发 |
| 500 等其他 5xx | 同一轮连续第三次时写入拒绝记录并前进，否则保留游标、停止 |

服务端按推导出的交互标识去重，在其查重保留窗口内，确认丢失后的重发不会形成重复交互。

### 8.6 配置与状态目录

| 环境变量 | 默认值 | 作用 |
|:---|:---|:---|
| `HIVEMEMORY_URL` | `http://localhost:8000` | HiveMemory 服务地址 |
| `HIVEMEMORY_USER_ID` | 不发送 | 作为 `X-User-ID` 请求头发送；缺省时由服务端取默认用户 |
| `HIVEMEMORY_AGENT_ID` | `claude_code` | 请求体中的 `agent_id` |
| `HIVEMEMORY_CONNECTOR_STATE_DIR` | 见下 | 状态目录 |

状态目录依次取 `HIVEMEMORY_CONNECTOR_STATE_DIR`、Claude Code 提供的插件数据目录 `CLAUDE_PLUGIN_DATA`，都没有时用 `~/.hivememory/claude-code-connector`；后台同步进程沿用 hook 解析出的状态目录。目录中保存 `transcripts/`（各会话记录的状态）、`connector.json`（首次运行时间）、`rejections.jsonl`（拒绝记录：会话与轮次标识、状态码、错误码与原因）与 `connector.log`（超过 1 MiB 时轮转一份）。状态、拒绝记录与日志只保存标识、位置与状态，不保存会话正文。

## 9. 设计不变量

- Import Bus 不检索记忆、不返回记忆上下文，不运行 Alice、MTP、命令或回复生成，不写入 ConversationSession；`PASSIVE_MEMORY` 只能得到 Gateway 决定。
- 确认只在感知应用成功之后：`accepted` 与 `duplicate` 都表示这一轮已经应用，应用失败或等待超时都不确认。
- 查重先于路由：已应用或仍在应用的轮次不再调用 Gateway；只有应用失败的轮次重新路由，并以同一 `interaction_id` 再次提交。
- 同一（user、Agent、`source`、外部会话）内按到达顺序处理，串行门持有到应用完成；不同外部会话不被一把全局锁串行化。
- `interaction_id` 只由认证结果与外部标识推导，同一轮重发不变。
- 导入授权只在 Import Bus 服务调用，只校验准入与目标，不检查白名单，拒绝保留的 `system`；`IdentityScope` 只在一次请求的调用链内使用。
- 交互记录由共用的 `seal_interaction` 产出，与任务进程的交互记录同形；`worth_saving` 不得删除原始交互。
- 编排器不持有提交队列、不注册维护任务；关闭时只等待在途请求，不排空队列。
- RuntimeEvent 不携带正文，发布失败不改变结果。
- 连接器只在得到确认，或判定这一轮不可重试并写入拒绝记录之后前进游标；hook 不做网络提交，不阻塞 Claude Code。

## 10. 已知限制

- 查重依赖 Patchouli 提交队列的进程内记录，终态记录有保留上限：确认送达之前服务重启，或超出保留窗口后重发，会形成重复的交互。
- 话题路由失败时续接上一轮的兜底，会把用户恰在故障期间换了话题的轮次归入旧话题。
- 超过请求体上限的一轮以 413 整轮跳过；提交队列的终态工作记录保存完整的编码载荷，工具输出很大时占用可观的内存。
- 工具结果可能包含密钥、令牌等凭据：连接器不筛选，原文随交互 Artifact 持久化，截断后的结果进入生成转写；提取提示词要求不把凭据写入记忆。
- 请求头中的用户身份不做证明，只适用于本地单用户部署。
- 连接器依赖 Claude Code 未公开承诺的会话记录格式；本地记录中的思考多数只有签名，连接器只能转交有正文的部分。30 分钟无新条目即按中断结束的规则，会把等待权限确认超过 30 分钟、恰逢其他会话唤醒补发的一轮按中断提交，之后写入的内容不再导入。连接器只在 Linux（WSL）上验证过，Windows 上的文件锁以持续重试取得。

这些项目在真实使用中的观察见 [Import Bus 真实使用观察 Todo](../todo/import-bus-real-use-observation.md)。

## 11. 代码与测试入口

代码：

- 入口：[`server/routers/import_bus.py`](../../src/hivememory/server/routers/import_bus.py)、[`server/models/import_bus.py`](../../src/hivememory/server/models/import_bus.py)、错误映射 [`server/error_handlers.py`](../../src/hivememory/server/error_handlers.py)、声明解析与请求级 context [`server/deps.py`](../../src/hivememory/server/deps.py)；
- 服务与编排：[`system/application/import_bus_service.py`](../../src/hivememory/system/application/import_bus_service.py)、[`system/services/import_bus/`](../../src/hivememory/system/services/import_bus/)（`ingestor.py`、`models.py`、`errors.py`、`events.py`）；装配见 [`system/assembler.py`](../../src/hivememory/system/assembler.py)，停止顺序见 [`system/system.py`](../../src/hivememory/system/system.py)；
- 导入授权：[`workspace/authorization.py`](../../src/hivememory/workspace/authorization.py) 的 `authorize_import`；
- 封口：[`core/protocol/sealing.py`](../../src/hivememory/core/protocol/sealing.py)；
- Patchouli 公开路由与提交队列：[`patchouli/contracts/public_routes.py`](../../src/hivememory/patchouli/contracts/public_routes.py)、[`patchouli/contracts/interaction_submission.py`](../../src/hivememory/patchouli/contracts/interaction_submission.py)、[`patchouli/control/interaction_submission.py`](../../src/hivememory/patchouli/control/interaction_submission.py)；
- 配置：[`config/import_bus.py`](../../src/hivememory/config/import_bus.py)；访问登记：[`configs/workspace_actors.yaml`](../../configs/workspace_actors.yaml)；
- 连接器：[`integrations/claude-code/`](../../integrations/claude-code/)、[`integrations/.claude-plugin/marketplace.json`](../../integrations/.claude-plugin/marketplace.json)。

测试：

- 编排（进度查询、跳过、Gateway 输入、等待应用与重发、失败分类、串行、事件脱敏、关闭）：`tests/unit/system/services/import_bus/test_ingestor.py`；输入模型与交互标识：`tests/unit/system/services/import_bus/test_models.py`；
- HTTP 入口（声明、准入、413 在解析前判定、503/500 映射）：`tests/unit/server/routers/test_import_bus.py`、`tests/unit/server/test_error_handlers.py`；
- 导入授权与调用位置：`tests/unit/workspace/test_access.py`、`tests/unit/architecture/test_access_boundaries.py`；
- 封口：`tests/unit/core/protocol/test_seal_interaction.py`；提交队列的状态、等待与失败后重交：`tests/unit/patchouli/control/test_interaction_submission.py`；
- 装配与关闭：`tests/unit/system/application/test_api_services.py`、`tests/unit/system/test_hivememory_system.py`、`tests/unit/system/test_lifecycle.py`；配置：`tests/unit/system/test_config_agent_runtime.py`；
- 真实队列与感知的协作：`tests/integration/system/test_import_bus_submission.py`；离线 E2E：`tests/e2e/system/test_import_turns.py`；
- 连接器：`tests/unit/connectors/test_claude_code_reader.py`、`tests/unit/connectors/test_claude_code_hooks.py`、`tests/unit/connectors/test_sync.py`。
