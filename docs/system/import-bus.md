---
title: Import Bus
status: current
owner: system
scope: external-harness-complete-turn-import-and-harness-connectors
code_paths:
  - src/hivememory/server/routers/import_bus.py
  - src/hivememory/server/models/import_bus.py
  - src/hivememory/system/application/import_bus_service.py
  - src/hivememory/system/services/import_bus/
  - src/hivememory/config/import_bus.py
  - src/hivememory/core/protocol/sealing.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/capability/root/interaction.py
  - src/hivememory/workspace/capability/root/topic.py
  - src/hivememory/workspace/capability/root/passive_routing.py
  - src/hivememory/patchouli/contracts/public_routes.py
  - src/hivememory/patchouli/control/interaction_submission.py
  - integrations/hivememory/
  - integrations/.claude-plugin/marketplace.json
  - configs/workspace_actors.yaml
  - configs/system_principals.yaml
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
  - docs/contracts/mcp.md
related_docs:
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/authentication.md
  - docs/workspace/task-process.md
  - docs/gateway/analysis.md
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/components/runtime-and-bus.md
  - docs/help/claude-code-plugin.md
  - docs/help/deepseek-harness-plugin.md
  - docs/help/codex-plugin.md
  - docs/help/pi-plugin.md
related_plans:
  - docs/archive/plans/v0.7.1-import-bus-redesign.md
  - docs/archive/plans/v0.7.1-memory-access-mcp.md
  - docs/archive/plans/v0.7.2-plugin-write-update.md
  - docs/archive/plans/dsh-plugin-integration.md
  - docs/archive/plans/codex-plugin-integration.md
  - docs/archive/plans/pi-plugin-integration.md
  - docs/archive/plans/claude-code-wake-turns.md
last_reviewed: 2026-11-12
---

# Import Bus

Import Bus 是外部 harness 对话回流 HiveMemory 的入口。harness 中的一轮交互结束后，连接器把这一轮的完整记录（用户消息、助手正文、思考、工具调用与结果）一次交给 Import Bus；Import Bus 用与任务进程相同的封口函数组装交互记录，经 root API 的提交交互一次完成查重、Gateway 话题路由、提交与等待应用，并在感知应用完成后才确认。第一个接入的 harness 是 Claude Code，连接器以插件形式位于 `integrations/hivememory/`（第 8 节）；第二个是 DeepSeek Harness（dsh），插件也在这个目录中（第 9 节）；Codex 与 Claude Code 合用这个插件目录（第 10 节），pi 的扩展同样在其中（第 11 节）。

外部 harness 经 MCP 写入与更新记忆时，写入发生在一轮进行之中，需要这一轮最终所在的话题。Import Bus 为此在写入时为这一轮确定话题并登记，这一轮结束回流时沿用已确定的话题（第 4.7 节）。

Import Bus 只向记忆网络输入：它不检索记忆，也不向调用方返回记忆上下文，不运行 Alice、MTP、命令或回复生成，不经任务进程的注册入口，导入的轮次也不写入 ConversationSession（plugin 模式下会话归外部 harness 所有）。为话题路由读取候选话题属于 Gateway 的分析，不算读取记忆。Gateway 的调用模式仍名为 `PASSIVE_MEMORY`。

设计取舍的完整记录见[已归档的 Import Bus 重设计计划](../archive/plans/v0.7.1-import-bus-redesign.md)（历史材料）；重设计之后的后续方向（流式监听、历史导入、第二个 harness）记录在已归档的 [Import Bus Idea](../archive/ideas/import-bus.md)（历史材料，2026-10-30 归档），去向见其归档说明。

## 1. 为什么一次接收一整轮

- **不返回记忆上下文，提前路由没有收益**：Import Bus 不在用户消息到达时返回任何上下文，在轮次中途路由只剩时序成本——harness 的 hook 只在少数节点触发，响应很快时模型的首条消息可能先于路由完成到达。在轮次结束处一次接收完整记录，Gateway 可以照常路由，服务端也不需要跨请求累积事件、按空闲或下一条用户消息封口。
- **载荷与主动链路同形**：导入的交互记录与任务进程封口的交互记录由同一个函数产出，Patchouli 不区分来源即可感知、折叠与生成。
- **确认等于已应用**：连接器只在得到确认后前进游标。若在提交被队列接纳时就确认，应用失败（例如提交指定的话题在应用前已被驱逐或结算）发生在确认之后，连接器不会重发，这一轮就丢失了；因此 Import Bus 等待感知应用完成才确认，应用失败时交回可重发的错误。

## 2. 边界与组成

| 部分 | 位置 | 负责 | 不负责 |
|:---|:---|:---|:---|
| HTTP 入口 | `server/routers/import_bus.py`、`server/models/import_bus.py` | 请求体上限、请求校验、身份声明解析、经统一认证网关认证得到请求的工作 actor、错误映射 | 授权、编排 |
| 应用服务 `ImportBusService` | `system/application/import_bus_service.py` | 导入（`import_turn`）与写入时确定话题（`place_turn`）：以系统自身的 principal 认证系统的 root 工作 actor，连同目标 Workspace、参与的 Agent 与 principal 标识委托编排器；关闭时转交 | 授权、跨请求的业务状态 |
| 编排器 `ImportBusIngestor` | `system/services/import_bus/` | 按外部会话串行、推导 `interaction_id`、跳过没有输出的轮次、封口、登记已确定的话题、把提交的结局译为导入结果、观测事件、关闭时等待在途请求；导入一轮与写入时确定话题各以系统的 root 工作 actor 调用一次 root API | 授权、查重、路由与等待（都在 root API 的方法内）、持有 scope 或提交队列、注册维护任务、检索、记忆存储、写入意图的登记与派发 |
| root API `RootInteractionService` | `workspace/capability/root/interaction.py` | 提交交互：入口 root 授权一次，方法内查这一轮的进度（仍在应用的经 `.wait` 等待）、以 PASSIVE_MEMORY 路由待路由的一轮（续接时查上一轮的进度），再调用 `patchouli.public.interaction.submit` 提交并等待应用（参与者显式传递）；与任务进程共用 | 串行、登记、结局到导入结果的映射 |
| root API `RootTopicService` | `workspace/capability/root/topic.py` | 为一轮确定话题：入口 root 授权一次，方法内查这一轮的进度、以 PASSIVE_MEMORY 路由，再经 `patchouli.public.topic.prepare` 确认或建立话题；与任务进程共用 | 编排、登记 |
| 封口函数 `seal_interaction` | `core/protocol/sealing.py` | 两条链路共用：组装 `InteractionPayload` | 解析事件（只在感知层进行） |
| Patchouli | 公开路由 `patchouli.public.interaction.submit` / `.status` / `.wait`、`patchouli.public.topic.prepare` | 提交队列、提交并等待应用、进度、应用、新话题的建立 | 导入的身份与重试策略 |
| Python 连接器（Claude Code 与 Codex） | `integrations/hivememory/` | 读取会话记录、切分轮次、游标与重发；两个 harness 合用一个插件目录与公共骨架 | 服务端的查重与授权 |
| DeepSeek Harness 插件 | `integrations/hivememory/node/dsh/` | 读取 dsh 会话的逻辑记录、切分轮次、水位与重发；注册记忆工具 | 服务端的查重与授权 |
| pi 扩展 | `integrations/hivememory/node/pi/` | 注册记忆工具、为每一轮写结束标记、按待提交的轮次回流与重发 | 服务端的查重与授权 |
| Node 公共部分 | `integrations/hivememory/node/core/`、`package.json` | dsh 插件与 pi 扩展共用的配置、MCP 调用与导入提交及结果分类、状态、日志与回声规则；包清单同时是 dsh bundle 与 pi 包 | harness 各自的轮次投影与触发 |

编排器的状态只有串行门、在途计数与已确定话题的登记，都只在进程内。它只经 root API 调用 Gateway 与 Patchouli 的公开路由，不持有全局总线、任何子系统的 Runtime 或 Patchouli 的提交队列；组合根只注入 root API 的 `RootInteractionService` 与 `RootTopicService`（与任务进程共用同一实例，Gateway 请求时限注入在它们上面）、最近轮数（沿用 Gateway 配置）、等待应用的时限与观测 publisher。应用服务另由组合根注入认证网关与系统自身的 principal 及其 adapter，与任务进程的注册入口相同，不注入操作授权者。

## 3. HTTP 入口

`POST /api/v1/import/turns`，请求体是 harness 一轮已结束交互的完整记录。请求以 `X-Principal-ID` 声明连接器的 principal、以 `Authorization: Bearer` 携带接入凭据（第 4.1 节），用户与 Workspace 选择只来自统一请求头。请求模型拒绝未知字段：

| 字段 | 说明 |
|:---|:---|
| `source` | 来源标识，1–64 位小写字母、数字与连字符，例如 `claude-code` |
| `external_conversation_id` | harness 的会话标识 |
| `external_turn_id` | 该轮在 harness 中的标识，同一轮重发时不变 |
| `agent_id` | 连接器所代表的具体 Agent，只作为认证声明，不进入导入记录 |
| `outcome` | `completed`、`cancelled` 或 `failed`，与 `InteractionOutcome` 相同 |
| `user_message` | 该轮的用户消息 |
| `events` | 用户消息之后的事件，与 `TurnEvent` 同形（[数据模型](../architecture/data-model.md#49-交互记录turnevent-与-interactionpayload)）：`kind`（`assistant_message`、`thought`、`subagent_message`、`tool_call`）、`sequence`（从 1 起严格递增）、`content`；`subagent_message` 是被委托的子 Agent 结束时交回的消息，只带 `content`；工具调用另带轮内唯一的 `action_id`、`tool_name` 或 `verb`、`targets`、`tool_args`，有结局时带成对的 `status` 与 `result` |
| `model_used` | 实际使用的模型名，可为空 |
| `recent_turns` | 该轮之前的若干轮（`user_message`、`assistant_text`、可选 `external_turn_id`），只供 Gateway 路由，不进入交互记录 |

- 服务端把 `user_message` 写成序号为 0 的 `user_message` 事件；回复正文不单独上传，由事件中的助手正文得到。
- 一次工具调用与它的结局是同一个 `tool_call` 事件：`status` 为 `success`，出错为 `error`，`result` 是结果正文；会话记录没有记下结果的调用不带结局（第 8.3 节）。
- `verb` 标记对 HiveMemory 能力层的记忆操作，与 Alice 的 MTP 动词同一词表，与 `tool_name` 互斥。外部 harness 经 HiveMemory 的 MCP 工具检索、读取、写入与更新，导入因此接受 `SEARCH`、`READ`、`WRITE` 与 `UPDATE`，其他取值以 422 拒绝；`READ` 与 `UPDATE` 必须带非空的 `targets`（所读的 alias、被修改的 alias），否则生成视图无法说明读了或改了哪些记忆。外部 harness 经 MCP 取回的记忆、显式写入与更新的内容会作为工具调用留在会话记录里并随本轮导入；生成视图对 `SEARCH`、`READ` 只保留一行摘要（`SEARCH: "查询"`、`READ: a, b`）、不渲染结果，`WRITE`、`UPDATE` 不渲染，与 Alice 的 MTP 写出同形的事件、渲染相同，因此不会把已有记忆或已显式写入的内容再提取一遍（回声）；其他外部工具只带 `tool_name`，生成视图把它们按工具名、参数摘要与截断后的结果渲染（[生成](../patchouli/generation.md)）。哪些调用归一化为能力调用由连接器按 [MCP 记忆访问](../contracts/mcp.md#6-连接器调用名轮次信息与回声标记)的工具名对应负责，交互记录与 Artifact 仍保存包括结果在内的原始事件。
- 事件不接受 `produced_aliases`：外部 harness 不产出正式 alias，写入与更新返回的 pending alias 只留在工具结果中。
- 单个事件的字段规则与一轮的不变量在请求校验阶段执行，与封口时相同：序号重复或回退、调用标识重复、字段出现在不允许的种类上或结局字段不成对时，整轮以 422 拒绝，不等到封口才失败。未升级的连接器仍上传旧的事件形态，它的轮次因此以 422 拒绝并被跳过（第 8.5 节），升级服务端时需要同时重新安装插件。反过来，带 `WRITE`、`UPDATE` 事件的轮次被不认识这两个动词的旧服务端以 422 拒绝，插件与服务端需要一起升级。
- 请求体上限 `import_bus.max_request_bytes` 在框架读取并解析请求体之前判定（自定义路由类 `BodyLimitedRoute`）：声明的 `Content-Length` 超限时不读取请求体，没有声明长度时边读边计数，超限即停止。两种情况都返回 413（`request.body_too_large`），过大且不是合法 JSON 的请求因此也是 413 而不是 422。

成功响应为 200，带推导出的 `interaction_id` 与 `status`：

| `status` | 含义 |
|:---|:---|
| `accepted` | 本次提交，且已经应用 |
| `duplicate` | 此前已经提交，现已应用（含本次请求等到上一次提交应用完成） |
| `skipped` | 该轮没有用户消息之外的事件，与任务进程一样不封口 |

响应不暴露 Gateway 决定、话题或记忆上下文。失败的状态码与可重试性见第 5 节。

## 4. 认证、授权与编排

### 4.1 认证与授权

导入是 Agent action：server 用 `resolve_request_identity_claims(require_agent=True)` 把请求头的用户选择与请求体的 `agent_id` 解析为 (user, 具体 Agent) 声明，缺失 `agent_id` 返回 400（`request.agent_required`），显式给出保留的 `system` 返回 400（`request.agent_reserved`）；`X-Workspace-ID` 只接受 `main_workspace`，其他值返回 404。principal 取自请求头 `X-Principal-ID`，缺省返回 400（`request.principal_required`）；凭据取自 `Authorization: Bearer`。随后以请求声明的 principal、`import` adapter 与凭据经统一认证网关认证（`authenticate_request_actor`），得到请求的工作 actor：它证明请求来自哪个调用来源、交互的参与者是哪个 Agent，只在本次调用内使用；凭据由网关第一阶段核对，缺失或不一致以 403 `invalid_credential` 拒绝。连接器以 Claude Code 自己的接入登记 `harness:claude-code` 认证，与插件中的 MCP server 共用（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)）；server 的 principal 只用于前端路由。认证前不组装 `IdentityScope`。请求头中的用户身份不做证明，沿用本地单用户部署的信任假设。

导入与确定话题是系统级操作，执行者是系统，交互中的 Agent 只是参与者（[身份与授权](../workspace/identity-and-authorization.md#4-操作授权)）。应用服务以系统自身的 principal 经 `local` adapter 认证同一用户在参与者驻留 Workspace 的 `system`，得到系统的 root 工作 actor（[认证与访问登记](../workspace/authentication.md#5-系统自身的-principal)），交给编排器；应用服务只认证、不授权。编排器以这个工作 actor 调用 root API，由 root API 在方法入口以 `authorize_root` 对目标授权。system 驻留在参与者的 Workspace，root 授权因此同时确认目标就是参与者驻留的 Workspace，否则 root 方法在入口即以 `target_workspace_not_resident` 拒绝，先于查进度、路由与提交，交互不会落到参与者所在之外的 Workspace。actor 操作白名单不参与：导入提交的是已经在外部发生的交互，与任务进程封口后提交交互记录同类，不调用任何 actor 操作，因此白名单为空的 Agent 也能导入；保留的 `system` 作为参与者已由声明解析以 `request.agent_reserved` 拒绝。写入时确定话题（第 4.7 节）同样由系统执行，它只为这一轮选定话题；写入与更新由 actor API 按请求的工作 actor 的白名单逐次授权，授权通过之后才确定话题。系统自身的 principal 未登记时以 403 `unknown_principal` 拒绝，该用户在这个 Workspace 没有 `system` 的访问记录时以 403 `actor_not_admitted` 拒绝。`authorize_root` 只在 root API 与任务进程中调用，系统自身的 principal 的使用位置也由架构测试锁定。

编排器不持有 `IdentityScope`：root API 在方法入口授权得到的 scope 只在那一次调用内经公共路由到达 Gateway 与 Patchouli，不写入记录、事件或后台任务；一轮导入与一次写入时确定话题各授权一次，查进度、路由、提交与等待都在同一次方法调用内沿用这个 scope。写入时确定话题的登记命中时直接返回、不经 root API：登记项只由此前经过 root 授权的请求写入，键由同一 principal 与参与者推导。提交经 root API 的提交交互进行，它把参与的 Agent 显式交给 `patchouli.public.interaction.submit`：Patchouli 以 scope 作归属（目标 Workspace），以参与者作交互的发起者。访问模型见[身份与授权](../workspace/identity-and-authorization.md)，访问登记的匹配规则见[认证与访问登记](../workspace/authentication.md#2-两类访问登记)。随仓库发布的 `configs/workspace_actors.yaml` 为 `default` 用户在 `main_workspace` 登记了 `claude_code` 的精确记录，`allowed_operations` 为 `memory.read`、`memory.search`、`memory.write` 与 `memory.update`，供 MCP 工具使用；导入与确定话题不读取白名单，不受它影响。精确记录优先于用户级记录，这个 Agent 不继承用户级的 `memory.run` 与 `agent.call`：MCP 入口不提供这两项操作。

### 4.2 编排顺序

```text
关闭中？ -> 503 shutting_down
推导 interaction_id
持串行门（user, Agent, source, external_conversation_id）
  关闭中？ -> 503 shutting_down
  没有输出              -> skipped（不查进度）
  话题来源：写入时已确定话题 -> 该话题（第 4.7 节）；否则一轮待路由的导入（第 4.5 节）
  seal_interaction -> InteractionPayload
  root API 的提交交互（origin import_bus，参与者，时限 apply_wait_seconds；目标不符在入口拒绝）
    查进度 succeeded / unknown -> 此前已接纳
           pending             -> 等待应用（第 4.6 节），等待的结果作为结局
           failed / 未提交      -> （待路由时）Gateway PASSIVE_MEMORY
                                  -> patchouli.public.interaction.submit：提交并等待应用
  接纳被拒 -> 503 queue_unavailable；Gateway 可恢复异常 -> 503 gateway_unavailable（登记都保留）
  调用返回 -> 取走已确定话题的登记
  结局 -> accepted / duplicate / apply_pending / apply_failed（第 4.6 节）
```

### 4.3 交互标识与串行

`interaction_id` 是 `import_` 加上对（principal、user、Agent、`source`、`external_conversation_id`、`external_turn_id`）规范化 JSON 序列取 SHA-256 的前 32 位十六进制。同一轮重发得到同一标识，队列据此幂等；principal 是连接器声明、经凭据核对的接入来源（Claude Code 为 `harness:claude-code`），user 与 Agent 取自显式的参与者（请求认证通过的 actor），代表不同用户或 Agent 的连接器因此不会撞号；前缀把它与任务进程的 `process_id` 及其他来源的标识分开。执行导入的 `system` 不进入推导，标识不随执行者变化。

编排器按参与者的（user、Agent）与（`source`、`external_conversation_id`）持 `KeyedSerialGate`（[运行时与总线](../components/runtime-and-bus.md#5-keyedserialgate)），门内完成封口与一次 root 提交，其中依次是进度查询、Gateway、提交与等待应用。同一外部会话的并发请求（例如两个 hook 同时触发补发）因此按到达顺序处理，后到的请求看到前一个的结果；串行门持有到应用完成，同一外部会话的下一轮路由时，上一轮新建的话题已经建立，可以成为候选话题与续接目标。不同外部会话互不阻塞。串行键只用于进程内排序，不参与授权。

### 4.4 进度查询与重发

查重先于路由，在 root API 的提交交互内部进行，以 Patchouli 的交互状态查询 `patchouli.public.interaction.status` 为准，它返回 `InteractionStatus | None`：未被接纳时为空；接纳后为 `pending`、`succeeded`（附实际所在的话题，`NEW_TOPIC` 已解析）、`failed` 或 `unknown`（接纳过但结局已超出队列的保留窗口）。

| 进度 | 处理 |
|:---|:---|
| `succeeded` 或 `unknown` | 返回此前已接纳的结局，导入返回 `duplicate`，不调用 Gateway |
| `pending` | 只继续等待应用，不再路由与提交（第 4.6 节） |
| `failed` | 重新路由，以同一 `interaction_id` 再次提交 |
| 未被接纳 | 写入时已确定话题则以该话题提交，否则路由并提交 |

没有输出的轮次在查进度之前就返回 `skipped`：它从不提交，同一轮重发时同样没有输出，不必查进度。

已应用与仍在应用的轮次不再调用 Gateway：Gateway 的结果写入载荷，重新路由可能得到不同载荷而被队列判为冲突。应用失败的轮次是例外：提交队列允许终态未成功的交互以同一标识重新提交，载荷可以不同（重新路由后的新目标话题），新提交以递增的尝试号作为工作项标识；等待中与已成功的交互仍按原规则幂等（[运行时与总线](../components/runtime-and-bus.md#33-interaction-submission)）。

### 4.5 Gateway 路由、封口与提交

- **Gateway**：编排器交给 root API 一轮待路由的导入（`PassiveTurnRouting`），由 root API 在方法内以 `PASSIVE_MEMORY` 模式调用 `gateway.public.process`，消息为该轮的用户消息，时限为 `gateway.workflow.default_request_timeout_ms`。`session_turns` 取请求中 `recent_turns` 的最后 `gateway.context_preparation.session_turn_limit` 轮；最近一轮带 `external_turn_id` 时，编排器推导它的 `interaction_id`，它在写入时已确定话题、尚未回流时直接取该话题，否则把这个交互标识交给 root API，由方法在路由前查它的进度，取得它应用成功后实际所在的话题，写入该轮的 `topic_id`。连接器的回流可能晚于下一轮的写入到达，此时上一轮只有写入时确定的话题。话题路由失败，或这一轮的用户消息为空（外部 harness 中被后台通知、子 Agent 交回等唤醒的续作）时，Gateway 不交给模型判断，续接上一轮仍在候选中的话题，否则新建话题（[Gateway 分析](../gateway/analysis.md#3-topic-router只做选择)）。Import Bus 只使用决定中的 `target_topic_id`、`new_topic_title` 与 `new_topic_summary`，三者都来自话题路由；Gateway 在这一模式下不运行查询分析，每个导入轮次至多只有话题路由一次模型调用，决定中的意图与检索计划是不检索的默认值，Import Bus 不调用检索。导入一侧不在入口作任何价值判断，交互记录不因入口判断被丢弃。
- **封口**：以补上序号 0 用户消息事件后的轮次事件、`outcome` 与 `model_used` 调用 `seal_interaction`；Import Bus 不携带附件，`used_attachments` 为空。Gateway 决定不进入交互记录。
- **提交**：root API 的提交交互调用 `patchouli.public.interaction.submit`，`origin` 为 `import_bus`，参与者为请求认证通过的 Agent；目标为 `NEW_TOPIC` 时附上 Gateway 给出的 `new_topic_title` 与 `new_topic_summary`，感知以它们建立新话题。后续路由依赖候选话题的标题与摘要，缺少它们的“新建话题”会让持续导入的路由质量下降。写入时已确定话题的轮次以该话题的真实 id 提交，不再携带标题与摘要：新话题在写入时就已经以它们建立。提交按话题生成排序键，导入与主动链路落到同一话题时由同一个排序键保证先后。

### 4.6 等待应用与确认

提交与等待在同一次 root 调用内：`patchouli.public.interaction.submit` 接纳交互后等待它应用，最多等待 `import_bus.apply_wait_seconds`；这一轮已被接纳、仍在应用时，root API 改以 `patchouli.public.interaction.wait` 等待同样的时限。编排器在串行门内等待这次调用返回，把结局译为导入结果：

- 本次提交且应用成功：发出 `import.turn.accepted`，返回 `accepted`；
- 此前已接纳且已应用（含这次等到上一次提交应用完成），或结局已超出保留窗口：发出 `import.turn.duplicate`，返回 `duplicate`；
- 仍为 `pending`（等待超时）：返回 503（`import_bus.unavailable`，reason `apply_pending`）；重发时查到仍在应用，只继续等待；
- `failed`：返回 500（`import_bus.apply_failed`）；重发时重新路由并再次提交。

等待的代价是每次导入多等一次应用。话题折叠不在应用内：触发折叠的一轮写入后即确认；同一话题的下一轮在提交队列中等待折叠结束，超过 `apply_wait_seconds` 时以 503 `apply_pending` 交回，重发时只继续等待（[感知与短期话题](../patchouli/perception.md#5-page-folding)第 5 节）。已接纳的交互在等待时查不到属于契约违约，root API 以 `RuntimeError` 原样传播。

### 4.7 写入时确定话题

外部 harness 经 MCP 写入与更新记忆（[MCP 记忆访问](../contracts/mcp.md)）时，写入发生在一轮进行之中，而导入要等这一轮结束才路由。写入意图需要一个派发话题：Patchouli 以它的状态摘要与最近几个 block 作为生成材料（[生成](../patchouli/generation.md)），写入的话题也应当就是这一轮最终所在的话题。MCP 入口因此给写入与更新一个确定话题的回调，actor API 授权通过（更新另需读到基础记忆）之后才调用它；回调以同一个工作 actor 调用 `ImportBusService.place_turn`，由系统为这一轮确定话题：

```text
关闭中？ -> 503 shutting_down
推导 interaction_id（与导入相同的推导，回流时得到同一个标识）
持串行门（与导入同一把）
  关闭中？ -> 503 shutting_down
  已确定话题         -> 返回该话题，不经 root API
  root API 的为一轮确定话题（目标不符在入口拒绝）
    查进度 succeeded -> 这一轮实际所在的话题
           pending   -> 不返回话题 -> 503 apply_pending，写入方稍后重试
    其余：Gateway PASSIVE_MEMORY（与导入相同的输入与续接规则）
          -> patchouli.public.topic.prepare：确认已有话题，或以标题与摘要建立新话题（必要时先按 LRU 结算）
  登记已确定的话题（话题标识与是否新建），返回该话题的真实 id
```

- **输入**：在途轮次模型 `InFlightTurn` 携带 `source`、`external_conversation_id`、`external_turn_id`、`user_message` 与 `recent_turns`；导入请求的 `ImportedTurn` 在它之上加入事件、结局与模型名，同名字段共用同一份校验。连接器以导入请求的同一规则构造它（[Claude Code 连接器](#82-hooks)）。
- **先授权、再确定话题**：没有 `memory.write` / `memory.update` 的 Agent 的写入与更新在调用回调之前被 actor API 拒绝，不路由、不建立话题，也不触发 LRU 结算。
- **已确定话题的登记**：编排器的进程内状态，以 `interaction_id` 为键保存确定的话题与是否新建。按轮次而不是按外部会话保存：连接器的回流可能晚于下一轮的写入到达，按会话只保留一轮会让上一轮丢失已确定的话题。同一轮的多次写入共用第一次确定的话题。
- **回流沿用**：导入一轮时查到它已确定的话题，以该话题提交，不调用 Gateway；root 提交调用返回之后（交互已被接纳）才取走登记。接纳被拒等调用异常时登记保留，重发仍然沿用；应用失败后（例如该话题在本轮进行中超过空闲时限被结算，或被 LRU 驱逐，应用以话题不存在失败），重发时已没有登记，按现行规则重新路由。写入的材料在派发时已经读取，不受影响。`import.turn.accepted` 的 `is_new_topic` 取已确定的话题是否新建。
- **迟到的写入**：后台运行的子代理可能在这一轮结束、回流之后才写入。root API 确定话题前先查这一轮的提交进度：已经应用时直接返回它实际所在的话题，不再路由；已经提交、等待应用超时仍在应用时不返回话题，Import Bus 以 503 `apply_pending` 交回，写入方重试时它已应用，避免重新路由让写入与这一轮落在不同的话题。
- **失败**：这次 root 调用整体抛出的可恢复异常（来自 Gateway 或话题准备的只读依赖）转为 `import_bus.unavailable`（`gateway_unavailable`），关闭中为 `shutting_down`；写入正等待这个结果，失败直接交给调用方，不发出 `import.turn.deferred`（推迟事件表示连接器保留这一轮稍后重发）。其他异常（例如活跃池已满且 LRU 候选都被占用的 `TopicBusyError`）原样传播。失败时不登记，下一次写入重新确定。
- **没有回流的轮次**：这一轮始终没有回流时，写入时建立的话题没有接收交互，不是 Gateway 的候选话题，由空闲结算按没有材料的话题结束生命周期，不另设清理；登记留到进程重启。
- 确定话题只读取候选话题与交互进度，不读取记忆，也不返回记忆上下文；Import Bus 只向网络输入的前提不变。没有写入的轮次只有回流，与没有这一步时完全相同。

## 5. 失败分类与 HTTP 映射

错误按是否可重试区分，连接器据此决定游标（第 8.5 节）：

| 情况 | HTTP | 错误码 / reason | 连接器的处理 |
|:---|:---:|:---|:---|
| 请求校验失败（未知字段、序号不递增等） | 422 | `request.validation_failed` | 写入拒绝记录，前进 |
| 请求体超过上限 | 413 | `request.body_too_large` | 写入拒绝记录，前进 |
| 缺少 `X-Principal-ID` | 400 | `request.principal_required` | 保留游标，停止（配置问题） |
| 缺少 `agent_id` 或为 `system` | 400 | `request.agent_required` / `request.agent_reserved` | 保留游标，停止（配置问题） |
| `X-Workspace-ID` 不是 `main_workspace` | 404 | `request.workspace_not_found` | 保留游标，停止 |
| 凭据未配置或不一致 | 403 | `workspace.admission_denied` / `invalid_credential` | 保留游标，停止；修正凭据后补发 |
| 未获准入 | 403 | `workspace.admission_denied`（如 `actor_not_admitted`） | 保留游标，停止 |
| 服务端接入登记缺少系统自身的 principal | 403 | `workspace.admission_denied` / `unknown_principal` | 保留游标，停止；与连接器自己的 principal 未登记同一原因，排查时两条登记都要核对 |
| 认证网关已关闭 | 503 | `workspace.admission_denied` / `authentication_gateway_closed` | 保留游标，稍后重发 |
| Gateway 整体抛出可恢复异常 | 503 | `import_bus.unavailable` / `gateway_unavailable` | 保留游标，稍后重发 |
| 提交队列容量已满或已停止 | 503 | `import_bus.unavailable` / `queue_unavailable` | 保留游标，稍后重发 |
| 已提交、等待应用超时 | 503 | `import_bus.unavailable` / `apply_pending` | 保留游标，重发时继续等待 |
| 编排器已开始关闭 | 503 | `import_bus.unavailable` / `shutting_down` | 保留游标，稍后重发 |
| 已提交、应用失败 | 500 | `import_bus.apply_failed` | 有限次重发，重发时重新路由并再次提交 |
| 契约违约、程序错误 | 500 | `server.internal_error` | 有限次重发 |

- Gateway 各决策步骤的失败由 Gateway 自身兜底，照常返回一个决定，导入按兜底结果继续。只有 Gateway 整体抛出的超时与明确的只读依赖故障（`TimeoutError`、`GatewayTimeoutError`、`StorageOfflineError`、`StorageReadError`、`ResourceUnavailableError`）才转为 `gateway_unavailable`：导入没有用户在等待，稍后重发可以得到正常的结果。查重、路由与提交在同一次 root 调用内，编排器对这次调用整体判断，其中抛出的这些异常都按这一原因交回。未知异常、授权拒绝与取消原样传播。
- Patchouli 提交路由在队列容量已满或已停止时抛出 `InteractionAdmissionUnavailableError`，Import Bus 转为 `queue_unavailable`；队列没有接管这次提交，以同一 `interaction_id` 重发不会形成重复交互。
- `PASSIVE_MEMORY` 返回命令结果视为契约错误，root API 以 `RuntimeError` 原样传播，不转为可重试错误。
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

编排器关闭时先标记关闭：认证网关关闭之前已通过认证、在关闭后才到达编排器的请求，以及在串行门外等待的请求，都以 503 `shutting_down` 拒绝（不发出 `import.turn.deferred`）；每个会话正在门内处理的请求继续完成，包括等待应用的时间，受 Gateway 请求时限与 `apply_wait_seconds` 约束。写入时确定话题与导入共用串行门与在途计数，关闭时同样被拒绝或等待。已确定话题的登记只在进程内，不持久化。提交队列此时仍在运行，已提交的轮次因此能得到确认；Import Bus 不排空队列，队列由 Patchouli 停止时排空。`shutdown()` 返回 `{"waited_requests": n}`（开始关闭时的在途请求数），System 把它放进 `system.stopped` 与 `system.stop_failed` 的 `import_bus_shutdown` 字段。Import Bus 没有维护任务，不向调度器注册，也没有启动步骤。

### 6.2 运行时事件

一轮导入在自己的 `import-` trace 中进行（`span_name` 为 `ImportBus.Turn`，`task_type` 为 `background`），结束时复原：Gateway 的被动路由、交互提交与新建话题的事件都继承它，与 `import.turn.*` 归为同一次运行（[运行时事件与可观测性](../components/observability.md#11-一次运行)第 1.1 节），这一轮的日志也带上它。

编排器经 `ImportBusEventEmitter` 在注入的 publisher 上发出 best-effort 的 RuntimeEvent（subsystem `system`，component `import_bus`，`task_type` 为 `background`：导入没有用户在 HiveMemory 一侧等待）。事件只携带 `source`、`external_conversation_id`、`external_turn_id`、`interaction_id`、事件数量与结构化标签，`agent_id` 与 `workspace_id` 只是观测标签；不携带用户消息、回复、思考、工具参数或工具结果的正文。脱敏规则集中在发布器一处。

| 事件 | 时机 | 附加内容 |
|:---|:---|:---|
| `import.turn.accepted` | 本次提交应用完成后 | `topic_id`（实际所在的话题）、`outcome`、`is_new_topic`（写入时已确定话题时取它是否新建，否则取本次提交是否请求建立新话题） |
| `import.turn.duplicate` | 已应用的轮次重发，或重发等到上一次提交应用完成 | reason `interaction_already_accepted` |
| `import.turn.skipped` | 没有输出的轮次 | reason `no_output` |
| `import.turn.deferred` | 可重试或应用失败的推迟（severity `warning`） | reason `gateway_unavailable`、`queue_unavailable`、`apply_pending` 或 `apply_failed`；`error_class` 只有异常类型名 |

事件发布失败不改变导入结果，事件也不能替代 HTTP 响应决定业务成功（[路由与事件](../contracts/routes-and-events.md)）。

写入时确定话题本身不发出 `import.turn.*`，只在 debug 级别记录一行日志：来源、`interaction_id`、确定的话题与是否新建，不记录正文；写入落在哪个话题由这次写入的 `memory.task.*` 给出（`topic_id` 与 `pending_alias`）。它在 MCP 工具调用的 trace 中执行，引起的 Gateway 路由与新建话题的事件归入那一次工具调用（[MCP 记忆访问](../contracts/mcp.md#7-观测)）。

## 7. 配置

`HiveMemoryConfig.import_bus`（`config/import_bus.py` 的 `ImportBusConfig`，拒绝未知键）：

| 键 | 默认值 | 作用 |
|:---|---:|:---|
| `max_request_bytes` | `8388608`（8 MiB） | 单次导入请求体的字节上限，超出返回 413 |
| `apply_wait_seconds` | `30` | 提交后等待感知应用完成的时限，超时返回 503 `apply_pending` |

Gateway 请求时限（`gateway.workflow.default_request_timeout_ms`，默认 8000）与最近轮数（`gateway.context_preparation.session_turn_limit`，默认 3）沿用 Gateway 配置。随仓库发布的 `configs/config.yaml` 不含 `import_bus` 段，按默认值运行。连接器的 HTTP 时限固定为 60 秒，覆盖默认的 Gateway 时限与等待时限之和；调大这两项时需要保持在连接器时限之内，否则连接器先超时并按可重试错误重发，重发时服务端查到仍在应用而继续等待。

## 8. Claude Code 连接器

连接器不属于 HiveMemory 包：它只用 Python 标准库、不导入 `hivememory`，由 harness 所在环境的 Python 直接运行。插件目录 `integrations/hivememory/` 同时是 Claude Code 与 Codex 的插件（插件名都是 `hivememory-connector`），两者共用连接器的公共骨架与同一份使用手册 skill；本节是 Claude Code 一侧，Codex 一侧见第 10 节。目录中的 `package.json` 与 `node/` 是 pi 扩展（第 11 节），与 Python 连接器无关。安装与配置的操作步骤见[接入 Claude Code](../help/claude-code-plugin.md)。

### 8.1 组成

- `integrations/.claude-plugin/marketplace.json`：本地 marketplace `hivememory-local`，只列出插件 `hivememory-connector`（来源 `./hivememory`），供本机安装，不发布；Claude Code 与 Codex 都从它安装。
- `integrations/hivememory/.claude-plugin/plugin.json` 与 `hooks/hooks.json`：Claude Code 的插件清单与五个命令 hook；插件说明同时覆盖对话回流与记忆访问。hook 在 Git Bash 中执行，解释器取 `${HIVEMEMORY_PYTHON:-$(command -v python3 || command -v python)}`：插件页安装时把运行 HiveMemory 的 Python 写进 `HIVEMEMORY_PYTHON`，没有时用 PATH 上的 Python。Codex 按自己的清单读取另外的 hook 与 MCP 文件，不读这两个文件。
- `integrations/hivememory/.mcp.json`：插件声明的 MCP server `hivememory`（HTTP，`${HIVEMEMORY_URL:-http://127.0.0.1:8769}/mcp`），请求头带 principal、`Bearer ${HIVEMEMORY_CLAUDE_CODE_TOKEN}`、用户与 Agent 声明；插件启用后 Claude Code 自动连接，HiveMemory 不可用时重试后标为连接失败，工具不出现，工作照常进行（[MCP 记忆访问](../contracts/mcp.md)）。
- `hivememory_hook.py`：启动脚本，把插件根目录放进导入路径后交给 `connector` 的命令行入口。第一个参数是 harness（`claude-code` 或 `codex`），决定来源标识、principal、默认 Agent、凭据变量、读取器与 hook 处理；其后是 `hook <事件>`、`sync --transcript <路径>` 或 `sync --all`，后台同步的命令沿用同一个 harness 参数。
- `connector/core/`：与 harness 无关的公共骨架——配置、状态与游标、提交与结果分类、同步循环、日志、本插件记忆工具的回声标记规则，以及用户消息中宿主附加的块（`user_message.py`，第 8.3 节）；`connector/claude_code/`：Claude Code 会话记录的读取器与 hook 处理；`connector/codex/`：Codex 的（第 10 节）。
- `scripts/transcript_stats.py`：维护工具，只统计会话记录的字段名、类型、计数与固定标记，不输出正文，用于在 Claude Code 升级后核对记录结构；其中 `user_message_tags` 列出导入为用户消息的条目中出现过的行首标签名，并标出导入时是否被去掉（与读取器使用同一张表）。

### 8.2 hooks

hook 同步运行但只做本地工作（登记会话记录、写结束记录、读会话记录），网络提交交给脱离 Claude Code 的后台同步进程（独立会话，不继承标准输入输出）。hook 入口无论成败都以 0 退出；只有 `PreToolUse` 在读到当前一轮时向标准输出写一个 JSON 对象，其他事件与任何失败都不写标准输出。连接器的任何失败只进日志，HiveMemory 不可用时不影响 Claude Code。

| hook | 行为 |
|:---|:---|
| `Stop` | 把游标之后的最后一轮记为 `completed`，唤醒这份会话记录的同步 |
| `StopFailure` | 把最后一轮记为 `failed`，唤醒同步 |
| `SessionStart` | 登记会话记录并撤销已结束标记（恢复的会话继续写入同一份记录），唤醒全部已登记会话记录的同步，补发尚未提交的已结束轮次 |
| `SessionEnd` | 标记会话已结束，唤醒同步；末尾没有结束记录的一轮按 `cancelled` 提交 |
| `PreToolUse` | 只匹配本插件写入与更新两个工具的调用名：把当前一轮的信息补进工具参数 `turn`，不登记、不唤醒同步、不做网络请求 |

`Stop` 必须同步运行：它在下一条消息写入之前结束，“最后一轮”才确定是刚结束的那一轮。最后一轮也可以是唤醒开启的续作（第 8.3 节），续作结束时的 `Stop` 为它写结束记录。结束通知在游标之后找不到轮次时（该轮已按超时或会话结束判定并提交）被忽略并写入日志。用户中断时 Claude Code 不触发 `Stop`，被中断的一轮由中断标记、下一轮或会话结束判定。

`PreToolUse` 的当前一轮是会话记录中最后开启的一轮：由人发出的消息开启的，或唤醒开启的续作（第 8.3 节）。它从游标处的最近上文读起，与导入读取的范围相同（尚未登记的会话记录从头读起），以生成导入请求体的同一函数构造 `source`、外部会话与外部轮次标识、用户消息与之前最多三轮，回流时服务端因此推导出同一个交互标识。输出 `hookSpecificOutput.updatedInput`：原参数加上 `turn`，覆盖模型自己填写的值；不给权限决定，改写后的参数照常经过 Claude Code 的权限评估。子代理中的调用得到的会话记录路径是主会话的会话记录，写入落在主会话的当前一轮。读不到当前一轮或任何失败时不输出，工具调用照常进行，由 MCP 入口以 `request.turn_required` 拒绝。

### 8.3 轮次的切分与映射

会话记录的格式不是 Claude Code 公开承诺的接口，解析只集中在读取器中，不认识的条目、块与字段一律跳过：

- 只读取 `user` 与 `assistant` 条目，跳过 `isSidechain`（子代理）、`isMeta`、`isCompactSummary` 条目与 `system` 条目（压缩边界、API 错误等）；只读完整的行。下面的唤醒条目与子 Agent 交回的消息例外。
- 一轮从一条开启模型运行的用户条目开始，到下一条这样的条目之前为止，`turnOrigin` 标明是谁开启的：`human` 是由人发出、内容不是工具结果的消息；`task_notification`（后台命令完成、Monitor 事件）、`peer`（后台子 Agent 交回报告或其他会话发来的消息，条目标为 `isMeta`）与 `sdk`（宿主注入的消息：派生会话的第一条交接说明，以及之后的宿主事件，如 side session 的汇报、CI 监视事件）是在 agent 空闲时唤醒它的消息，唤醒之后的工作同样是一轮。上一轮在 agent 停下时已经提交，唤醒开启的续作若不自成一轮，就会落在游标之前、永远不会提交。agent 正在工作时插入的同类消息不带 `turnOrigin`（只有 `origin`），并入进行中的一轮。轮次标识取开启它的条目的 `uuid`，外部会话标识取它的 `sessionId`（缺失时取会话记录的文件名）。`[Request interrupted by user...]` 标记立即结束一轮（结局 `cancelled`）。
- 用户消息：由人发出的消息取正文；派生会话的第一条交接说明（`sdk` 条目，没有 `parentUuid`）作为用户消息，它就是这个会话要做的事。两者都先去掉宿主附加的块：Claude 桌面端等宿主会把关联会话的状态、画布的视图状态、最近的项目文件夹等以标签块附加在用户消息里，它们不是用户说的话，而话题路由读取用户消息全文，记忆生成也以它为材料。去掉哪些块由连接器的一张共用表决定（`connector/core/user_message.py`：`system-reminder`、`artifact-view-context`、`recent-project-folders`），Codex 的用户消息使用同一张表（第 10.3 节）。只去掉从行首开始的完整块；代码块与用户粘贴的内容（`<pasted_content>`，保留标签以区分粘贴与本人的话）中的同名标签原样保留，行内提到的标签与表外的标签也不动，那是用户自己的内容。表外的标签不猜测：`scripts/transcript_stats.py` 列出导入的用户消息中出现过的行首标签，宿主新附加的块以 `kept` 出现，确认是宿主附加的再加入表中。其他唤醒的用户消息为空，服务端为这样的一轮续接上一轮的话题（[Gateway 分析](../gateway/analysis.md#3-topic-router只做选择)第 3 节）。子 Agent 交回的报告或其他会话发来的消息（`origin.kind` 为 `peer`，不论是否开启新轮）在出现位置写为 `subagent_message`，正文取 `origin` 中的报告原文（`handbackReportText`，其次 `body`）；它不并入用户消息，因为话题路由读取用户消息全文，而报告可能很长。后台通知与宿主事件只是状态提示，不读取。
- 助手内容块 `text` → `assistant_message`，`thinking` → `thought`（没有正文的跳过：较新的模型默认只在本地记录签名），`tool_use` → `tool_call`（`action_id` 为调用标识，`tool_args` 为输入；本插件自身 MCP 工具的四种调用名归一化为 `SEARCH`、`READ`、`WRITE`、`UPDATE` 能力调用，不上传调用名，读取的 `aliases` 与更新的 `alias` 移出参数作为 `targets`，写入与更新的 `turn` 去掉；调用名由插件名与 server 名两个常量按 Claude Code 的规则拼出，其他工具写 `tool_name`）。`aliases` 不是非空的字符串列表、`alias` 不是非空字符串时（模型传错了参数，MCP server 已以工具错误拒绝），读取与更新按普通工具写出，否则服务端会因缺少作用对象拒绝整轮。用户条目中的 `tool_result` 按 `tool_use_id` 补全到同一轮的调用上，作为它的结局（`is_error` 为真时 `status` 为 `error`，否则为 `success`；`result` 为结果正文），找不到调用的结果跳过。同一条消息并行发出多个调用时，Claude Code 有时只记下最后一个的结果，其余调用因此没有结局，模型其实收到了它们的结果。同一条 API 消息按内容块分条增量写入，重复的块按消息标识与块位置去重，工具调用与结果按调用标识去重。
- 回复正文不随轮次上传，服务端从助手消息事件得到；`model_used` 取最后一条助手条目的模型（`<synthetic>` 除外）。
- `recent_turns` 取该轮之前最近三轮已确认轮次的用户消息、按顺序以空行拼接的全部助手正文与 `external_turn_id`；正文各截断到 2000 字符，避免一轮过大的内容拖着后续轮次一起超过请求体上限。`PreToolUse` 补上的轮次信息以同一函数构造这些字段。
- 子代理的条目记在单独的会话记录中，主会话记录里的子代理条目（`isSidechain`）也跳过：子代理中的写入事件不随主会话的一轮导入，主会话只导入委托子代理的那次工具调用。

### 8.4 结束判定与游标

状态按会话记录分文件保存：游标（下一次读取的位置，之前的轮次都已确认）、最后确认的轮次标识、最近确认轮次的起点、各轮的结束记录与连续服务器错误次数。多个 hook 进程与同步进程同时运行，每次读改写持该会话记录的状态锁并以替换方式原子写入；同一会话记录的同步另持同步锁，提交期间不持状态锁，hook 不会被同步阻塞。

- **结束的判定**：一轮在其后出现下一轮、写入中断标记、已有结束记录或会话已结束时才算结束；仍在进行的末尾一轮不提交。没有 `SessionEnd` 的崩溃会话，末尾超过 30 分钟没有新条目时按 `cancelled` 结束。结局依次取已保存的结束记录（hook 写下的，或上次提交前保存的）、会话记录本身写明的结局（Claude Code 的记录没有，Codex 的有，见第 10.4 节）、以上推断。
- **末尾一轮是否写完**：结束通知可能先于最后几行写入。末尾一轮已判定结束、但最后一条助手条目的 `stop_reason` 仍是 `tool_use`（尚未正常收尾）时，同步轮询等待会话记录写完（最多约 10 秒）；读取之后会话记录又有新内容时重新读取，不以读取时的旧内容提交；会话已结束或会话记录静止超过等待时限时按读到的内容提交。
- **结局先于提交写入**：结束记录写入状态后才提交，重发沿用原结局；同步与 hook 同时为同一轮写结局时，先写入的生效，提交以状态中保存的为准。
- **首次见到的会话记录**：首个条目晚于连接器首次运行时间的从头读取；更早的（接入之前已有、后来被恢复的会话）不回填已经结束的轮次，游标位置由读取器决定：Claude Code 放在当前末尾（`SessionStart` 在第一条消息之前触发，首次见到时没有进行中的轮次）；Codex 在最后一轮仍在进行时放在这一轮的开头，否则放在末尾（第 10.2 节）。
- **游标失效**：游标不在行首，或最后确认的一轮不再从记录的位置开始（会话记录被重写或缩短）时，从头扫描，在最后一个已确认轮次之后继续；找不到该轮时停止处理这份会话记录并写入拒绝记录，不重发整个会话。会话记录被删除时移除它的状态，不写拒绝记录。

### 8.5 提交结果的处理

按顺序提交，确认之后游标才前进：

| 结果 | 处理 |
|:---|:---|
| 2xx（`accepted`、`duplicate`、`skipped`） | 前进 |
| 网络错误、429、502、503、504 | 保留游标，停止处理这份会话记录，等下次唤醒 |
| 413、422（这一轮内容本身的问题） | 写入拒绝记录，前进 |
| 其他 4xx（凭据未配置或不一致、身份声明、准入、地址或版本不对） | 保留游标，停止处理：这是连接器配置的问题，前进会让后续轮次全部丢失，修正配置后补发 |
| 500 等其他 5xx | 同一轮连续第三次时写入拒绝记录并前进，否则保留游标、停止 |

服务端按推导出的交互标识去重，在其查重保留窗口内，确认丢失后的重发不会形成重复交互。

### 8.6 配置与状态目录

| 环境变量 | 默认值 | 作用 |
|:---|:---|:---|
| `HIVEMEMORY_URL` | `http://127.0.0.1:8769` | HiveMemory 服务地址 |
| `HIVEMEMORY_USER_ID` | 不发送 | 作为 `X-User-ID` 请求头发送；缺省时由服务端取默认用户 |
| `HIVEMEMORY_AGENT_ID` | `claude_code` | 请求体中的 `agent_id` |
| `HIVEMEMORY_CLAUDE_CODE_TOKEN` | 不发送 | 接入凭据，作为 `Authorization: Bearer` 发送（去掉首尾空白）；未设置时服务端以 `invalid_credential` 拒绝，连接器保留游标等待补发 |
| `HIVEMEMORY_CONNECTOR_STATE_DIR` | 见下 | 状态目录 |

每次提交另以常量 `harness:claude-code` 作为 `X-Principal-ID` 发送。服务端核对这份凭据的环境变量与连接器同名，两侧需要配置同一份凭据（[配置](./configuration.md)）；变量名带上 harness，多个 harness 共用一套环境变量时各读各的凭据。

这里是 Claude Code 的取值；Codex 的来源标识、principal、默认 Agent 与凭据变量见第 10.5 节，其余变量相同。

状态目录依次取 `HIVEMEMORY_CONNECTOR_STATE_DIR`、Claude Code 提供的插件数据目录 `CLAUDE_PLUGIN_DATA`，都没有时用 `~/.hivememory/claude-code-connector`；后台同步进程沿用 hook 解析出的状态目录。目录中保存 `transcripts/`（各会话记录的状态）、`connector.json`（首次运行时间）、`rejections.jsonl`（拒绝记录：会话与轮次标识、状态码、错误码与原因）与 `connector.log`（超过 1 MiB 时轮转一份）。状态、拒绝记录与日志只保存标识、位置与状态，不保存会话正文。

## 9. DeepSeek Harness 插件

插件是一个 dsh 原生插件（bundle `hivememory-connector`），代码在 `integrations/hivememory/node/dsh/`，只用 Node 标准库，不导入 HiveMemory 包，经 dsh 的插件上下文取得工具、会话、会话查询与系统提示词服务；支持的 dsh 版本由 `engines.dsh` 声明（`>=0.2.1-rc.5 <0.3.0-0`）。它同时提供对话回流与记忆工具，记忆工具见 [MCP 记忆访问](../contracts/mcp.md#6-连接器调用名轮次信息与回声标记)第 6 节。安装与配置的操作步骤见[接入 DeepSeek Harness](../help/deepseek-harness-plugin.md)。

它不沿用 Claude Code 连接器的做法，原因在 dsh 一侧：

- dsh 的工具前置事件不能改写调用参数，无法像 `PreToolUse` 那样为通用 MCP 工具调用补上轮次信息，记忆工具因此由插件自己注册，回流与工具放在同一个插件中；
- dsh 的会话文件以 Zstandard 压缩，格式随版本迁移，插件不读文件，经会话查询服务读取会话的逻辑事件流，存活的会话读实时记录。

确认后前进、提交结果的分类与重发去重与 Claude Code 连接器相同（第 8.5 节），以 JavaScript 另外实现，不共用 Python 骨架（owner 决定，2026-11-01）；与 pi 扩展共用 `node/core/` 中的 JavaScript 公共部分（owner 决定，2026-11-08）。

### 9.1 组成

| 位置（`integrations/hivememory/` 下） | 负责 |
|:---|:---|
| `package.json` | bundle 清单：`main` 指向 `node/dsh/index.js`，`engines.dsh` 声明支持的版本，`dsh.bundle.patch` 指向配置层；同一份清单的 `pi` 字段供 pi 使用（第 11.1 节） |
| `node/dsh/cordis.patch.yml` | 配置层：插入插件行 `hivememory`，以及只扫描插件包中 `skills/` 的 skill 提供方 `hivememory-skills`，使用手册 skill 随插件提供；提供方注册在宿主的全局层，不改动用户与 agent preset 自己的 skill 配置，路径从 profile 目录解析插件包得到 |
| `node/dsh/index.js` | 装配，监听 Agent 创建与会话事件，连接 HiveMemory 并注册记忆工具 |
| `node/dsh/turns.js` | 会话事件到导入轮次与在途轮次的投影、结局映射 |
| `node/dsh/sync.js` | 同步循环与按会话保存的水位 |
| `node/dsh/tools.js` | 记忆工具定义 |
| `node/core/` | 与 pi 扩展共用：按 harness 取值的配置、MCP 调用与导入提交及结果分类、状态、日志、记忆工具的回声规则与最近轮次的截断 |

### 9.2 触发与读取

- **触发**：一轮结束（会话事件 `turn/end`，记下实际结束的轮号）与 Agent 创建或恢复时，同步该会话；插件加载时，同步全部还有实际结束但未确认的轮次、或水位尚未确定的会话。同一会话同时只有一个同步，运行期间再次触发时，结束后再同步一次。监听器在 dsh 的执行路径上同步运行，只做本地的快速工作，任何失败只进日志，网络提交在后台进行。
- **先落盘再读取**：会话在本进程中存活时，读取前先让 dsh 把缓冲的事件写入持久记录：确认之后 dsh 崩溃，已确认的轮次不会从会话记录中消失、再被新的一轮以同一轮号取代。
- **不在本进程的会话**：读到的是持久记录，读取方会为最后一轮未结束的记录补上 `interrupted` 收尾，而该会话可能正在另一个 dsh 进程中运行这一轮，因此只提交实际看到结束的轮次。崩溃时进行中的一轮，由 dsh 在会话恢复时补上收尾，之后照常回流。

### 9.3 轮次的切分与映射

- 一轮从 `turn/start` 到同一轮号的 `turn/end`，没有 `turn/end` 的一轮仍在进行，不提交。外部会话标识取会话标识，外部轮次标识取轮号：轮号在会话内唯一且递增，恢复与 fork 都接着已有的最大轮号编号。fork 出的会话只读继承前缀之后的事件；继承前缀中未结束的一轮由 fork 补上的收尾结束，它的 `turn/start` 在前缀中，因此不形成轮次。
- **用户消息**：来源为 `user` 的消息（一轮的提示与运行中追加的消息）按顺序以空行拼接，图片与文件写占位；harness 注入的上下文（技能正文、目录说明、压缩摘要、后台任务通知等）不读取。一轮可能没有这类消息，例如由子 Agent 的结算通知唤醒，用户消息因此为空。
- **事件**：助手消息的 `text` 块写为 `assistant_message`，`reasoning` 块写为 `thought`（没有正文的跳过），`tool-call` 块不读取，调用以独立的 `tool/call` 事件为准，写为 `tool_call`；`tool/result` 按调用标识补全到同一轮的调用上（`isError` 为真时 `status` 为 `error`，否则为 `success`），找不到调用的结果跳过。没有形成助手消息的尝试不读取。`model_used` 取最后一条助手消息的模型。
- **回声标记**：本插件四个记忆工具的调用归一化为 `SEARCH`、`READ`、`WRITE`、`UPDATE`，规则与 Claude Code 连接器相同（第 8.3 节）。插件注册的工具名就是 MCP 工具的名字，dsh 的工具注册表中同名工具只能有一个，工具名因此就是识别依据。
- **结局**：`completed` 为 `completed`；被取消（`aborted`）、崩溃后收尾（`interrupted`）与 fork 收尾为 `cancelled`；出错、被拦截（`blocked`）、输出达到上限（`max-tokens`）与插件扩展的未知原因为 `failed`。
- **最近轮次**：该轮之前最近三个已结束轮次的用户消息、全部助手正文与外部轮次标识，正文按字符截断到 2000。写入时补上的轮次信息以同一函数从同一份会话记录构造，回流时服务端推导出同一个交互标识。
- **子 Agent**：子 Agent 是父 Agent 临时委托的执行线程，与 Alice 的 CALL 子帧一样，它的过程不进入记忆生成，要保留的内容由它自己写入记忆；父会话的一轮保留它交回的最终输出（owner 决定，2026-11-02）。
  - 会话头来源为 `subagent` 的会话（进程内 spawn 与 fork 出的子 Agent）不登记、不导入，也就不建立话题。子会话若各自导入，每个子 Agent 都会占一个新话题、挤掉驻留话题，它的轨迹也会被单独提取成一批粒度与父 Agent 不一致的记忆。
  - 子 Agent 结算后 dsh 发给父 Agent 的通知（来源 `subagent-settled`：结局一行、子 Agent 最后的输出，有时附结构化结果与诊断）按出现位置写为 `subagent_message` 事件，原文保留；父 Agent 忙时通知并入进行中的一轮，空闲时开启新的一轮。它不并入用户消息，因为话题路由读取用户消息全文，一次调研交回的报告可达十万字符；也不作为委托调用的结局，因为生成材料中单条工具结果按上限截断。
  - 子 Agent 运行中用 `send_message` 发来的消息（来源 `agent-message`）不读取：结算通知已经带有它最后的输出，两者都读会让同一份报告重复进入材料。
  - 子 Agent 中的写入与更新沿会话头的 `parentSession` 逐级找到顶层会话，以顶层会话最近的一轮构造轮次信息，与这一轮回流时的同名字段相同；这一轮已经结束也照用，按第 4.7 节的迟到写入处理。

### 9.4 水位与首次见到的会话

- 每个会话的状态是：来源（`new` 或 `existing`）、水位（轮号不大于它的轮次都已确认、被不可重试地拒绝，或在登记时就已存在而不回填）、实际结束的最大轮号，以及各轮连续得到服务器内部错误的次数。水位只在确认之后、或写入拒绝记录之后前进，提交结果的处理见第 8.5 节。
- 首次见到的会话按创建时间与插件首次运行时间比较：之后创建的从头导入；更早创建的不回填，水位留空，第一次读取时放在当时最后一个已经结束的轮次，之后开始的轮次照常导入。
- 会话被删除（会话查询返回不存在）时移除它的状态，不再补发。插件卸载时中止进行中的请求，水位不前进。

### 9.5 配置与状态目录

插件从环境变量读取配置：地址、用户与 Agent 的变量与 Claude Code 连接器同名，地址默认为 `http://127.0.0.1:8769`，默认 Agent 为 `dsh`、默认状态目录为 `~/.hivememory/dsh-connector`；凭据变量是 `HIVEMEMORY_DSH_TOKEN`，与服务端核对这份凭据的变量同名（表格见[接入 DeepSeek Harness](../help/deepseek-harness-plugin.md#3-手动安装与命令行)第 3 节，服务端一侧见[配置](./configuration.md)）。每次请求以常量 `harness:dsh` 作为 `X-Principal-ID` 发送，来源标识为 `dsh`。

状态目录中保存 `sessions/`（各会话的状态）、`connector.json`（首次运行时间）、`rejections.jsonl`（拒绝记录：会话标识、轮号、状态码、错误码与原因）与 `connector.log`（超过 1 MiB 时轮转一份），只保存标识、轮号与状态，不保存会话正文。dsh 的插件代码在一个线程中运行，状态以同步文件操作读改写、替换写入，同一进程内不需要文件锁。

## 10. Codex 插件

Codex 与 Claude Code 合用插件目录 `integrations/hivememory/` 与 Python 连接器：Codex 的 `PreToolUse` 可以改写 MCP 工具参数，接入沿用 Claude Code 的形态——插件声明 HiveMemory 的 `/mcp`，hook 为写入与更新补上可信的轮次信息，会话记录读取器把轮次回流 Import Bus（owner 决定，2026-11-08）。游标、确认后前进、提交结果的分类与重发去重都是公共骨架（第 8.4、8.5 节）。会话记录与 hook 的结构依据 Codex 0.167.0，在 Windows 原生核对；安装与配置的操作步骤见[接入 Codex](../help/codex-plugin.md)。

### 10.1 组成

| 位置 | 负责 |
|:---|:---|
| `.codex-plugin/plugin.json` | Codex 的插件清单：写明 `skills`、`mcpServers: ./codex.mcp.json` 与 `hooks: ./hooks/codex.json`。Codex 按根目录 `plugin.json`、`.codex-plugin/plugin.json`、`.claude-plugin/plugin.json` 的顺序取第一份清单；清单写明 hook 与 MCP 文件后，不再读取 Claude Code 默认的 `hooks/hooks.json` 与 `.mcp.json` |
| `codex.mcp.json` | MCP server `hivememory`：`type: http`，地址 `http://127.0.0.1:8769/mcp`（Codex 不展开这里的环境变量）；`bearer_token_env_var: HIVEMEMORY_CODEX_TOKEN`；`http_headers` 带 `X-Principal-ID: harness:codex` 与 `X-Agent-ID: codex`；`env_http_headers` 把 `HIVEMEMORY_USER_ID` 作为 `X-User-ID`（未设置时不发送）；`default_tools_approval_mode: approve`，四个记忆工具不逐次审批 |
| `hooks/codex.json` | 五个命令 hook，命令为 `cmd /c "${PLUGIN_ROOT}/hivememory_hook.cmd" codex hook <事件>`：Codex 在交给 shell 之前替换 `${PLUGIN_ROOT}`，再交给会话所用的 shell（PowerShell 或 cmd）；两者引用环境变量的写法不同，所以由 `hivememory_hook.cmd` 选择解释器：设置了 `HIVEMEMORY_PYTHON`（插件页安装时写入）时用它，否则用 PATH 上的 `python` |
| `connector/codex/` | 会话记录读取器与 hook 处理 |

Codex 安装插件时把整个目录复制到 `CODEX_HOME/plugins/cache/` 下，仓库中的插件更新后要重新安装；插件的 hook 要经用户在 Codex 中审阅信任后才运行，插件内容变化（例如重新安装更新后的插件）后需要重新信任。

### 10.2 hooks

hook 只做本地工作，网络提交交给后台同步进程，规则与第 8.2 节相同；输出是 ASCII 转义的 JSON（Windows 上 hook 的标准输出按本地代码页编码）。

| hook | 行为 |
|:---|:---|
| `SessionStart` | 登记会话记录并撤销已结束标记（续会话在同一份会话记录末尾追加），唤醒全部已登记会话记录的同步 |
| `Stop` | 把输入中 `turn_id` 那一轮记为 `completed`，唤醒这份会话记录的同步 |
| `Interrupt` | 把输入中 `turn_id` 那一轮记为 `cancelled`，唤醒同步 |
| `SessionEnd` | 标记会话已结束，唤醒同步 |
| `PreToolUse` | matcher `^mcp__hivememory__(memory_write\|memory_update)$`：把当前一轮的信息补进工具参数 `turn` |

- `Stop` 与 `Interrupt` 早于这一轮的结束记录写入会话记录，同步沿用等待写完的轮询（第 8.4 节）。失败的一轮不触发 `Stop`，由会话记录中带 `error` 的结束记录判定，在下一次唤醒（下一轮结束、会话结束或下一次会话开始）时提交。
- 子 Agent 中触发的 hook（输入带 `agent_id`）不登记、不同步：只导入顶层会话，子 Agent 交回的消息随父会话的那一轮导入，与 dsh 插件相同（第 9.3 节）。
- `codex exec` 先写下第一轮的开始记录、再触发 `SessionStart`：连接器首次运行时间记在第一个 hook 触发的时刻，安装插件后的第一个会话因此按接入之前已有的会话登记。读取器把游标放在仍在进行的这一轮的开头，这一轮结束后照常导入，`PreToolUse` 也读得到它；之前已经结束的轮次不回填。
- 续会话时 Codex 给出的会话记录路径带 Windows 扩展长度前缀 `\\?\`，会话开始时的不带；连接器去掉这个前缀，同一份会话记录只保存一份状态。
- `PreToolUse` 的当前一轮是输入中 `turn_id` 那一轮。它从游标处的最近上文读起，以生成导入请求体的同一函数构造轮次信息，回流时服务端推导出同一个交互标识。输出 `permissionDecision: allow` 与 `updatedInput`（原参数加上 `turn`，覆盖模型自己填写的值）：Codex 只在给出 `allow` 时采用改写。子 Agent 中的调用（输入带 `agent_id`，`session_id` 是根会话，`turn_id` 与 `transcript_path` 是子 Agent 自己的）从子 Agent 的会话记录取得根轮（`task_started` 的 `root_turn_id`），在根会话已登记的会话记录中读取，写入因此归到派生它的那一轮。读不到当前一轮或任何失败时不输出，由 MCP 入口以 `request.turn_required` 拒绝。

### 10.3 轮次的切分与映射

会话记录是 `CODEX_HOME/sessions/<年>/<月>/<日>/rollout-<时间>-<线程ID>.jsonl`，每行为 `{type, payload}`：

- 第一行 `session_meta` 的 `id` 是线程 ID，作为外部会话标识；它的时间是会话开始时间。
- 一轮从 `event_msg` 的 `task_started` 开始（`turn_id` 为外部轮次标识），到同一 `turn_id` 的 `task_complete` 或 `turn_aborted` 为止。之后出现下一轮而没有结束记录的一轮按被打断处理。
- 内容只读 `event_msg` 中的 `item_completed` 条目；同一内容另有 `response_item` 形式的模型输入输出，其中工具参数是 hook 改写前的，不读取。`turn_context` 的 `model` 是 `model_used`。
- `UserMessage` 的文本是用户消息，一轮中有多条时按顺序以空行拼接，图片写占位；每条先按 Claude Code 连接器的同一张表去掉宿主附加的块（第 8.3 节），去掉后为空的不计入。模型输入中注入的环境上下文不在其中；提问工具的回答 `<send_user_message_question_reply>` 不在表中，原样保留。`AgentMessage` → `assistant_message`；`Reasoning` → `thought`（有原文取原文，否则取摘要，都没有时跳过）；`McpToolCall`、`CommandExecution`、`DynamicToolCall`、`FileChange`、`WebSearch` → `tool_call`（结局取条目状态，MCP 结果带 `isError` 时为 `error`）；等待子 Agent 的 `CollabAgentToolCall` 中已完成的子 Agent 消息 → `subagent_message`，其他子 Agent 调用 → `tool_call`。其他类型的条目不读取。
- 回声标记：`McpToolCall` 的 `server` 为 `hivememory` 时，四个记忆工具按 Claude Code 连接器的同一规则写为能力调用（第 8.3 节）。识别只看 server 名与工具名：用户在自己的配置中覆盖这个 server（第 10.5 节）时，Codex 记下的调用不带插件 ID。
- `recent_turns` 的取法与截断与第 8.3 节相同。

### 10.4 结局

| 会话记录 | 结局 |
|:---|:---|
| `task_complete` 不带 `error` | `completed` |
| `task_complete` 带 `error` | `failed` |
| `turn_aborted` | `cancelled` |

hook 写下的结束记录先于会话记录写明的结局（第 8.4 节）；`Stop` 只在正常结束时触发、`Interrupt` 对应中断记录，两者与会话记录一致。

### 10.5 配置

| 取值 | Codex |
|:---|:---|
| 来源标识 `source` | `codex` |
| principal（`X-Principal-ID`） | `harness:codex` |
| 默认 Agent | `codex` |
| 凭据变量 | `HIVEMEMORY_CODEX_TOKEN`，与服务端核对这份凭据的变量同名 |
| 状态目录 | Codex 为 hook 设置的插件数据目录（`CODEX_HOME/plugins/data/<插件>-<marketplace>`，以 `CLAUDE_PLUGIN_DATA` 别名读取），没有时 `~/.hivememory/codex-connector` |

`HIVEMEMORY_URL`、`HIVEMEMORY_USER_ID`、`HIVEMEMORY_AGENT_ID` 与 `HIVEMEMORY_CONNECTOR_STATE_DIR` 与 Claude Code 相同（第 8.6 节），用于导入；MCP 的地址与请求头由 `codex.mcp.json` 决定。用户在自己的 `config.toml` 中声明同名的 `[mcp_servers.hivememory]` 时，它整条替换插件的 server 定义（不合并），因此改地址时要连同凭据变量、请求头与审批模式一并写出。

## 11. pi 扩展

pi 扩展与 Claude Code、Codex 合用插件目录 `integrations/hivememory/`，代码是只用 Node 标准库的 JavaScript（`node/pi/`），不共用 Python 连接器，与 dsh 插件共用 `node/core/`。接入方式与 dsh 插件相同：扩展自己注册四个记忆工具，在工具执行时补上轮次信息，回流由扩展按事件完成（owner 决定，2026-11-08）。pi 也能在 `tool_call` 中改写工具参数，但扩展没有加载、或被后加载的扩展覆盖时，服务端无法分辨模型自己填写的轮次信息；pi 的内建 MCP 默认把工具交给 codemode，经 codemode 的内层调用不进入会话记录，回声标记会失效。确认后才算完成、提交结果的分类与重发去重与第 8.5 节相同。事件与会话记录的结构依据 pi 1.5.0，在 Windows 原生核对；安装与配置的操作步骤见[接入 pi](../help/pi-plugin.md)。

### 11.1 组成

| 位置 | 负责 |
|:---|:---|
| `package.json` | pi 包清单：`pi.extensions` 指向 `node/pi/index.js`，`pi.skills` 指向 `skills/`；Claude Code 与 Codex 不读取它 |
| `node/pi/index.js` | 装配：事件处理、连接 HiveMemory 并注册记忆工具、一轮的开始与结束 |
| `node/pi/turns.js` | 结束标记、一轮的条目到导入轮次与写入时轮次信息的投影、结局 |
| `node/pi/sync.js` | 待提交的轮次、同步循环与会话文件的读取 |
| `node/pi/tools.js` | 记忆工具定义 |
| `node/core/` | 与 dsh 插件共用（第 9.1 节） |

pi 以本地路径安装这个目录（`pi install <目录>`），原地加载、不复制，仓库更新后重新加载 pi 即生效。

### 11.2 记忆工具

- 会话开始后在后台取服务说明与工具列表；每次开始运行（`before_agent_start`）时最多等 3 秒让注册完成，第一轮的模型请求因此就带着这些工具。HiveMemory 不可用时工具不出现，下一次开始运行时再试。服务说明作为系统提示词中的 `hivememory` 一节。
- 四个工具以服务端给出的名字、说明与参数 schema 注册，schema 中去掉 `turn`；缺少任何一个时都不注册。注册为 `exposure: "model-only"`：只交给模型，codemode 与其他工具不能调用。
- 写入与更新执行时以当前一轮构造 `turn`，覆盖模型填写的值：外部会话为会话标识，外部轮次为这一轮的轮次标识，用户消息为至今收到的原始输入，之前几轮与导入时相同（第 11.4 节）。不在一轮之中时工具返回错误。

### 11.3 一轮与结束标记

pi 的会话记录是一棵树：条目以 `id` 与 `parentId` 相连，`/tree` 回到较早的条目继续时，离开的分支仍在同一个文件中；`/fork` 与 `/clone` 把当前路径上的条目连同标识复制进一个新会话。pi 没有与一轮对应的持久标识，由扩展写下：

- HiveMemory 的一轮对应 pi 从开始运行到 `agent_settled` 的一段，运行中追加的消息（steer）与排队的后续消息（follow-up）并入同一轮。开始运行时生成轮次标识（UUID），记下当时的叶子条目；`input` 中来源为 `interactive` 与 `rpc` 的原始文本（技能与模板展开之前）按顺序收集，其他扩展发来的输入不收集。
- `agent_settled` 时以 `pi.appendEntry` 追加一条 `custom` 条目（`customType: "hivememory-turn"`，不进入模型上下文）作为结束标记，数据为轮次标识、会话标识、开始时的叶子条目、结局与原始输入。
- 这一轮的条目是从结束标记沿 `parentId` 回溯、到开始时的叶子条目之前的那一段，与其他分支无关。
- fork 与 clone 复制来的标记带着来源会话的标识，不导入；接入之前的轮次没有标记，不回填。不保存会话的运行（`--no-session`）不写标记、不导入。
- 进程崩溃或被强制结束时，进行中的一轮没有结束标记，不导入。

### 11.4 轮次的映射

- 用户消息：结束标记中的原始输入，以空行拼接；由其他扩展的消息开始的运行没有原始输入，用户消息为空。会话记录中的用户消息是技能与模板展开后的正文，不读取。
- 事件：助手消息的 `text` 块写为 `assistant_message`，有正文的 `thinking` 块写为 `thought`，`toolCall` 块写为 `tool_call`，结局按 `toolCallId` 取对应的工具结果消息（`isError` 为真时为 `error`），图片写占位；系统消息、其他扩展注入的消息、压缩与分支摘要、上下文编辑、模型切换等条目不读取。`model_used` 取最后一条助手消息的模型。
- 回声标记：本扩展四个记忆工具的调用按第 8.3 节的同一规则写为能力调用；工具由扩展注册，工具名就是识别依据。
- 最近轮次：当前分支上、这一轮开始之前最近三条本会话结束标记对应的轮次，截断与第 8.3 节相同。写入时补上的轮次信息以同一函数从同样的条目构造。

| `agent_settled` 与最后一条助手消息 | 结局 |
|:---|:---|
| `aborted` 为真，或结束原因为 `aborted` | `cancelled` |
| 结束原因为 `error` 或 `length`（pi 已经做过恢复），或没有助手消息 | `failed` |
| 其余 | `completed` |

### 11.5 回流与补发

- 一轮结束时先把轮次标识记入这个会话的待提交列表，再写结束标记，然后在后台提交；会话开始时补发全部还有待提交轮次的会话。按待提交列表而不是文件中的全部标记提交，状态目录丢失时也不会重提已经确认过的轮次；待提交而文件中找不到标记的轮次移出列表。
- 提交时从会话文件逐行读取条目，不经 pi 的会话管理（它会迁移旧格式的文件）；末尾不完整的行与无法解析的行跳过。会话文件被删除时移除它的状态。
- 会话关闭（`session_shutdown`）时最多等 5 秒让进行中的提交完成，超时的留到下次启动补发；`pi -p` 因此在这一轮得到确认之后才退出。

### 11.6 配置与状态目录

| 取值 | pi |
|:---|:---|
| 来源标识 `source` | `pi` |
| principal（`X-Principal-ID`） | `harness:pi` |
| 默认 Agent | `pi` |
| 凭据变量 | `HIVEMEMORY_PI_TOKEN`，与服务端核对这份凭据的变量同名 |
| 服务地址 | `HIVEMEMORY_URL`，未设置时为 `http://127.0.0.1:8769` |
| 状态目录 | `HIVEMEMORY_CONNECTOR_STATE_DIR`，未设置时为 `~/.hivememory/pi-connector` |

`HIVEMEMORY_USER_ID` 与 `HIVEMEMORY_AGENT_ID` 与其他连接器相同（第 8.6 节）。状态目录保存 `sessions/`（各会话的文件路径、待提交的轮次与连续服务器内部错误的次数，待提交清空时移除）、`rejections.jsonl` 与 `connector.log`（超过 1 MiB 时轮转一份），只保存标识、路径与状态，不保存会话正文。pi 的扩展代码在一个线程中运行，状态以同步文件操作读改写、替换写入，同一进程内不需要文件锁。

## 12. 设计不变量

- Import Bus 不检索记忆、不返回记忆上下文，不运行 Alice、MTP、命令或回复生成，不写入 ConversationSession；`PASSIVE_MEMORY` 只能得到 Gateway 决定。
- 确认只在感知应用成功之后：`accepted` 与 `duplicate` 都表示这一轮已经应用，应用失败或等待超时都不确认。
- 查重先于路由：已应用或仍在应用的轮次不再调用 Gateway；只有应用失败的轮次重新路由，并以同一 `interaction_id` 再次提交。没有输出的轮次直接跳过，不查进度。
- 写入时已确定话题的轮次回流时以该话题提交、不调用 Gateway，写入的话题就是这一轮所在的话题；登记在提交被接纳后才取走。这一轮已经应用后才到达的写入沿用它实际所在的话题，不重新路由。
- 写入时确定话题与导入对同一轮推导出同一个 `interaction_id`，在同一把串行门内按到达顺序处理；确定话题只读取候选话题与交互进度，不读取记忆。
- 同一（user、Agent、`source`、外部会话）内按到达顺序处理，串行门持有到应用完成；不同外部会话不被一把全局锁串行化。
- `interaction_id` 只由认证结果与外部标识推导，同一轮重发不变。
- 导入与确定话题由系统经 root API 以 root 授权执行，各调用一次 root 方法、授权一次，应用服务只认证、不授权；目标只能是参与者驻留的 Workspace，不检查参与者的白名单；交互的发起者是参与的 Agent；`IdentityScope` 只在一次 root API 调用内、工作 actor 只在一次请求的调用链内使用。
- 写入时确定话题发生在 actor API 授权写入之后：被拒的写入不路由、不建立话题，也不触发 LRU 结算。
- 交互记录由共用的 `seal_interaction` 产出，与任务进程的交互记录同形，只以事件表达交互内容；Gateway 决定不进入交互记录，也不得删除原始交互。
- 编排器不持有提交队列、不注册维护任务；关闭时只等待在途请求，不排空队列。
- RuntimeEvent 不携带正文，发布失败不改变结果。
- 连接器只在得到确认，或判定这一轮不可重试并写入拒绝记录之后前进游标；hook 不做网络提交，不阻塞 Claude Code 与 Codex。dsh 插件的水位遵循同一规则，事件监听只做本地工作，不阻塞 dsh。pi 扩展的待提交列表同样只在确认或写入拒绝记录之后移出一轮；结束标记写在会话记录中，网络提交在后台进行，会话关闭时的等待有上限。

## 13. 已知限制

- 查重依赖 Patchouli 提交队列的进程内记录，终态记录有保留上限：确认送达之前服务重启，或超出保留窗口后重发，会形成重复的交互。
- 话题路由失败时续接上一轮的兜底，会把用户恰在故障期间换了话题的轮次归入旧话题。
- 超过请求体上限的一轮以 413 整轮跳过；提交队列的终态工作记录保存完整的编码载荷，工具输出很大时占用可观的内存。
- 工具结果可能包含密钥、令牌等凭据：连接器不筛选，原文随交互 Artifact 持久化，截断后的结果进入生成转写；提取提示词要求不把凭据写入记忆。
- 请求头中的用户身份不做证明，只适用于本地单用户部署；接入凭据是本地共享密钥，插件页安装时生成并写进 harness 一侧的 Windows 用户环境变量，同一用户运行的程序都能读到；后端环境变量中的凭据修改经重启生效。
- 回声标记依赖 Claude Code 的工具调用命名规则：规则变化时标记静默失效，回声恢复，表现为重复记忆；`PreToolUse` 的 matcher 写的也是这些调用名，同样失效，写入与更新以缺少轮次信息被拒绝。
- 写入时已确定话题的轮次始终没有回流（连接器长期不可用，或 hook 未运行时模型编造了轮次信息）时，登记留到进程重启，写入时建立、没有接收交互的话题由空闲结算按没有材料的话题结束。
- 后台运行的子代理在会话开启下一轮之后（用户发出下一条消息，或 agent 被唤醒）才写入时，写入落在那时的当前一轮：`PreToolUse` 的输入只给出会话当时的当前一轮。dsh 的子 Agent 同样落在顶层会话那时最近的一轮。
- 第一次写入要等一次 Gateway 路由；同一外部会话的上一轮仍持有串行门等待应用时，还要等它完成。
- Claude Code 中一次 Monitor 调用可能多次唤醒 agent，每次唤醒都是一轮，续接在同一话题中，交互数随之增加。后台任务运行很久、上一轮的话题已经结算或被驱逐时，唤醒之后的续作新开一个没有标题的话题。派生会话的第一轮只凭交接说明路由，可能落进别的会话的话题。
- 宿主附加的块只按连接器的表去掉：宿主新附加、尚未加入表的块随用户消息导入，进入话题路由与生成材料；表更新之前已经导入的轮次不改写。
- Claude Code 的会话记录有时只记下同一条消息并行发出的多个调用中最后一个的结果，其余调用以没有结局的形式导入，生成材料中缺少这些结果；连接器无法补回会话记录没有保存的内容。
- 连接器依赖 Claude Code 未公开承诺的会话记录格式；本地记录中的思考多数只有签名，连接器只能转交有正文的部分。唤醒条目的识别依赖 `turnOrigin` 的取值与子 Agent 交回的 `origin.handbackReportText`，字段变化或旧版本不写 `turnOrigin` 时，唤醒之后的续作不再单独成轮、不会导入。30 分钟无新条目即按中断结束的规则，会把等待权限确认超过 30 分钟、恰逢其他会话唤醒补发的一轮按中断提交，之后写入的内容不再导入。连接器在 Linux（WSL）与 Windows 原生上实际运行过，Windows 上的文件锁以持续重试取得。

- Codex 插件依赖 Codex 未公开承诺的会话记录格式与插件、hook 机制，按 0.167.0 核对；升级后需要重新核对。插件的 hook 要在 Codex 中审阅信任后才运行，每次更新插件都要重新信任；未信任、被禁用或出错时写入缺少轮次信息而被拒绝，此时模型自己填写的轮次信息也不会被覆盖，服务端无法分辨它不是适配器补上的。
- Codex 插件的 MCP 地址写死为 `127.0.0.1:8769`；后端在别的地址时，用户要在自己的配置中整条重写这个 server。
- Codex 的 fork、回退与上下文压缩对会话记录与游标的影响未经实测：fork 出的会话记录若带有父会话的轮次，会以新会话重复导入；回退不撤回已经导入的轮次。代码执行模式中间接调用的记忆工具不做回声标记。

- dsh 插件依赖 dsh 开发者预览版的插件服务与会话事件，跨版本可能不兼容：`engines.dsh` 限定在 0.2.x，桌面端升级会同时升级 dsh。dsh 的 PTC 模式（默认关闭）下记忆工具在 `run_code` 内部调用，会话记录只有外层调用，回声标记不生效。
- dsh 中同名 skill 由离 Agent 更近的来源胜出：桌面端的各模式自己扫描用户与项目的 skill 目录，那里的同名 `hivememory` 会遮蔽插件提供的使用手册 skill，并且不随插件更新。
- dsh 一轮中运行时追加的用户消息并入用户消息，与助手事件的相对位置不保留。
- dsh 子 Agent 的过程（读了哪些文件、工具输出了什么）不进入任何材料：子 Agent 没有写入、最后的输出也没有提到的发现会丢失；子 Agent 只经 `send_message` 交回完整结果、最后只说“已完成”时，材料中只有这句话。一轮带多个子 Agent 时结算通知让材料明显变大（一次 5 个子 Agent 的调研多出约 10 万字符），会更早触发 Page Folding，被折叠的轮次只以接力摘要进入结算时的生成。
- dsh 插件每次同步与每次写入都读取整个会话记录，开销随会话长度增长；崩溃时进行中的一轮只在会话恢复后回流，不再恢复的会话不回流这一轮；多个 dsh 进程共用状态目录时没有跨进程互斥，同一轮可能提交两次，由服务端按交互标识去重。

- pi 扩展依赖 pi 1.x 的扩展接口与会话格式，按 1.5.0 核对；1.x 没有不破坏扩展的承诺，升级后需要重新核对。
- pi 进程崩溃或被强制结束时，进行中的一轮没有结束标记，不导入；一轮中追加的输入与助手事件的相对位置不保留。
- pi 的记忆工具不对 codemode 开放，由 codemode 编排的任务不能在脚本中访问记忆。社区子 Agent 扩展的会话被当作普通会话导入，各占一个新话题，子 Agent 中的写入以它自己会话的轮次确定话题。
- 从 `/tree` 回到一轮中间的条目继续时，那一轮的结束标记不在新分支上，新一轮的最近轮次不含它。
- pi 扩展每次同步都读取整个会话文件，开销随会话长度增长；同一会话在两个 pi 进程中同时运行时，pi 本身不保证写入顺序；多个 pi 进程共用状态目录时没有跨进程互斥，同一轮可能提交两次，由服务端按交互标识去重。
- 与 Honcho 等记忆扩展共存时，它们自己的检索工具调用作为普通工具调用导入，可能形成回声。

这些项目曾在 v0.7.1 期间随日常使用观察，记录见已归档的 [Import Bus 真实使用观察](../archive/todo/import-bus-real-use-observation.md)（2026-10-30 归档，之后日常使用中发现的问题另行登记）。

## 14. 代码与测试入口

代码：

- 入口：[`server/routers/import_bus.py`](../../src/hivememory/server/routers/import_bus.py)、[`server/models/import_bus.py`](../../src/hivememory/server/models/import_bus.py)、错误映射 [`server/error_handlers.py`](../../src/hivememory/server/error_handlers.py)、声明解析与请求的工作 actor [`server/deps.py`](../../src/hivememory/server/deps.py)；
- 服务与编排：[`system/application/import_bus_service.py`](../../src/hivememory/system/application/import_bus_service.py)、[`system/services/import_bus/`](../../src/hivememory/system/services/import_bus/)（`ingestor.py`、`models.py`、`errors.py`、`events.py`）；装配见 [`system/assembler.py`](../../src/hivememory/system/assembler.py)，停止顺序见 [`system/system.py`](../../src/hivememory/system/system.py)；
- 授权：[`workspace/authorization.py`](../../src/hivememory/workspace/authorization.py) 的 `authorize_root`；root API 的提交交互：[`workspace/capability/root/interaction.py`](../../src/hivememory/workspace/capability/root/interaction.py)，为一轮确定话题：[`workspace/capability/root/topic.py`](../../src/hivememory/workspace/capability/root/topic.py)，两者共用的被动路由：[`workspace/capability/root/passive_routing.py`](../../src/hivememory/workspace/capability/root/passive_routing.py)；
- 封口：[`core/protocol/sealing.py`](../../src/hivememory/core/protocol/sealing.py)；
- Patchouli 公开路由与提交队列：[`patchouli/contracts/public_routes.py`](../../src/hivememory/patchouli/contracts/public_routes.py)、[`patchouli/contracts/interaction_submission.py`](../../src/hivememory/patchouli/contracts/interaction_submission.py)、[`patchouli/application/interaction_submission_service.py`](../../src/hivememory/patchouli/application/interaction_submission_service.py)、[`patchouli/control/interaction_submission.py`](../../src/hivememory/patchouli/control/interaction_submission.py)；
- 配置：[`config/import_bus.py`](../../src/hivememory/config/import_bus.py)；访问登记：[`configs/workspace_actors.yaml`](../../configs/workspace_actors.yaml)；
- 连接器（Claude Code 与 Codex）：[`integrations/hivememory/`](../../integrations/hivememory/)、[`integrations/.claude-plugin/marketplace.json`](../../integrations/.claude-plugin/marketplace.json)；dsh 插件与 pi 扩展：[`integrations/hivememory/package.json`](../../integrations/hivememory/package.json)、[`integrations/hivememory/node/`](../../integrations/hivememory/node/)（`dsh/`、`pi/` 与共用的 `core/`）。

测试：

- 编排（跳过、路由输入与续接、结局到导入结果的映射与重发、失败分类、串行、事件脱敏、关闭，写入时确定话题、回流沿用与登记的保留、以写入时确定的话题续接与迟到的写入）：`tests/unit/system/services/import_bus/test_ingestor.py`；输入模型、动词与交互标识：`tests/unit/system/services/import_bus/test_models.py`；
- HTTP 入口（principal 与凭据、声明、准入、413 在解析前判定、503/500 映射）：`tests/unit/server/routers/test_import_bus.py`、`tests/unit/server/test_error_handlers.py`；
- root 授权与调用位置、系统自身的 principal 的使用位置：`tests/unit/workspace/test_access.py`、`tests/unit/architecture/test_access_boundaries.py`；提交交互（查重先于路由、仍在应用的只等待、失败后重新路由、续接上一轮的话题、路由失败不提交）：`tests/unit/workspace/capability/root/test_interaction.py`；为一轮确定话题（已应用返回实际所在的话题、应用中不返回话题）：`tests/unit/workspace/capability/root/test_topic.py`；root 方法只接受 `system`、目标不是驻留 Workspace 时先于路由拒绝：`tests/integration/workspace/test_root_api_authorization.py`；
- 封口：`tests/unit/core/protocol/test_seal_interaction.py`；提交队列的状态、等待与失败后重交：`tests/unit/patchouli/control/test_interaction_submission.py`；提交并等待应用的结局、进度与等待：`tests/unit/patchouli/application/test_submission_services.py`；
- 装配与关闭：`tests/unit/system/application/test_api_services.py`、`tests/unit/system/test_hivememory_system.py`、`tests/unit/system/test_lifecycle.py`；配置：`tests/unit/system/test_config_agent_runtime.py`；
- 真实队列与感知的协作、交互的发起者是参与者、系统自身的 principal 未登记与目标不是参与者驻留的 Workspace 时的拒绝：`tests/integration/system/test_import_bus_submission.py`；经 MCP 写入后回流落在写入派发的话题：`tests/integration/system/test_mcp_memory_access.py`；离线 E2E：`tests/e2e/system/test_import_turns.py`；
- 用户消息中宿主附加的块（表中从行首开始的完整块去掉，代码块、粘贴的内容、行内提及与表外的标签保留）：`tests/unit/connectors/test_user_message.py`；
- 连接器（含唤醒开启的一轮与它的用户消息、由人发出的消息去掉宿主附加的块、子 Agent 交回的事件、上一轮提交之后的续作单独提交、回声标记、`PreToolUse` 的轮次信息与之后的导入一致、插件声明与 hooks matcher 和常量一致、principal、缺省 Agent、凭据变量与 hook 的解释器变量和插件服务一致、按 harness 取凭据、会话记录写明的结局）：`tests/unit/connectors/test_claude_code_reader.py`、`tests/unit/connectors/test_claude_code_hooks.py`、`tests/unit/connectors/test_sync.py`；Codex（轮次切分、用户消息的拼接与去掉宿主附加的块、事件映射、回声标记与结局、根轮的取得、`PreToolUse` 的输出与子 Agent 归属、扩展长度前缀、清单与插件服务的登记一致、hook 经启动脚本选择解释器）：`tests/unit/connectors/test_codex_reader.py`、`tests/unit/connectors/test_codex_hooks.py`；
- dsh 插件（轮次投影、回声标记与结局、写入时的轮次信息与之后的导入一致、水位与补发）：`integrations/hivememory/node/test/dsh-*.test.js`；事件种类是导入接受的，principal、缺省 Agent 与凭据变量和插件服务登记的一致，包清单的入口与 bundle 配置存在、补丁指向的使用手册 skill 目录存在：`tests/unit/connectors/test_deepseek_harness_plugin.py`；
- pi 扩展（一轮的条目只取结束标记回溯的一段、fork 复制的标记不导入、原始输入、事件映射、回声标记与结局、写入时的轮次信息与之后的导入一致、只交给模型的工具、待提交与补发、关闭时有上限的等待，以及事件处理的整体流程）：`integrations/hivememory/node/test/`，与 dsh 插件的测试和公共部分的测试（提交分类、按 harness 取凭据）一起以 `npm test` 运行（CI 的 `node-connectors` 任务）；共用的记忆工具与 MCP server 的工具名、参数一致，principal、缺省 Agent 与凭据变量和插件服务登记的一致，包清单指向的扩展与 skill 存在：`tests/unit/connectors/test_pi_extension.py`。
