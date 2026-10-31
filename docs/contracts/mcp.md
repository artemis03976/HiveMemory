---
title: MCP 记忆访问
status: current
owner: system
scope: mcp-memory-access-for-external-harnesses
code_paths:
  - src/hivememory/mcp_server/
  - src/hivememory/server/app.py
  - src/hivememory/server/deps.py
  - src/hivememory/server/error_handlers.py
  - src/hivememory/workspace/capability/operations.py
  - src/hivememory/system/application/import_bus_service.py
  - src/hivememory/system/services/import_bus/
  - src/hivememory/engines/memory_compiler/
  - integrations/claude-code/.mcp.json
  - integrations/claude-code/hooks/hooks.json
  - integrations/claude-code/connector/claude_code/hooks.py
  - integrations/claude-code/connector/claude_code/reader.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/capability.md
  - docs/workspace/authentication.md
  - docs/workspace/read-view-and-write-intents.md
  - docs/system/import-bus.md
last_reviewed: 2026-10-31
---

# MCP 记忆访问

外部 harness（当前为 Claude Code）经 HiveMemory 的 MCP 端点检索、读取、写入与更新记忆。它与 [MTP](./mtp.md) 并列：MTP 是 Alice 在生成循环中使用的文本协议，MCP 是外部 harness 使用的工具协议。两者都是**操作适配器**：把协议调用解析为固定类型的操作请求，经 workspace 的单一操作入口进入 actor API，不直接调用能力层，也不自行判定身份或权限。记忆能力属于 actor API，不为哪一种协议独占；写入与更新经两种协议进入的是同一组 actor 操作、同一条生成链路。

plugin 模式下，外部 harness 的任务对 HiveMemory 不可见，没有任务进程，也就没有执行线程。每次工具调用因此只取得一份**请求级访问 context**，按 Workspace 访问登记逐次授权，调用结束即撤销。写入与更新需要一个派发话题：线程提交从线程取得 prepare 为本轮选定的话题，请求级提交没有线程，改由 Import Bus 在写入时为这一轮就地落位（第 4 节）。RUN 的执行发生在执行单元的环境中，CALL 只属于 Alice，端点不提供这两项。

## 1. 端点与传输

| 项目 | 规则 |
|:---|:---|
| 地址 | `POST /mcp`，由 server 应用以路由接入，登记在 SPA 兜底路由之前。以路由而不是挂载接入：挂载在 `/mcp` 的子应用只匹配 `/mcp/` 下的路径，插件声明的 `/mcp` 会被重定向 |
| 传输 | MCP Python SDK 的 streamable HTTP，无状态、JSON 响应：每个 HTTP 请求独立处理，不保存 MCP 会话，也没有服务端通知、资源或提示 |
| 订阅 | 不提供 `subscriptions/listen`，也不宣告工具、资源与提示的列表变更：工具在运行时不变，没有可推送的变更。该请求的响应是一条一直打开的流，只在会话管理器退出时结束；而 uvicorn 要等所有连接结束后才执行 server 的 lifespan 关闭，接受订阅会让任何在线的客户端（Claude Code 以 2026-07-28 协议连接时会发出该请求）都让服务停不下来 |
| 生命周期 | SDK 会话管理器由 server 的 lifespan 运行：在装配系统之前进入，在系统停止之后退出。挂进宿主后 SDK 自己的 lifespan 不会执行，没有运行会话管理器时每个请求都会失败。会话管理器每个进程只能运行一次 |
| Host 与 Origin | 保留 SDK 默认的检查，防止 DNS 重绑定：只接受带端口的本机地址（`localhost:*`、`127.0.0.1:*`、`[::1]:*`），其他主机名以 421 拒绝 |

## 2. 身份交接

身份只取自请求头。工具参数由模型生成，永远不作为身份来源；连接器补上的轮次信息只用于落位，也不是身份。入口不组装 `IdentityScope`。

| 请求头 | 规则 |
|:---|:---|
| `X-Principal-ID` | 必填，声明调用来源（Claude Code 为 `harness:claude-code`）；缺省或空串返回 `request.principal_required`，只含空白返回 `request.identity_empty` |
| `Authorization` | `Bearer <凭据>`；缺省或不是 Bearer 方案时凭据为空，同样交给网关，由网关以 `invalid_credential` 拒绝 |
| `X-User-ID` | 用户声明，缺省回退到默认用户；用户身份不做证明，沿用本地单用户部署的信任假设，由接入登记的 `allowed_user_ids` 收紧 |
| `X-Workspace-ID` | 只接受 `main_workspace`，其他值返回 `request.workspace_not_found` |
| `X-Agent-ID` | 必填的具体 Agent；缺失返回 `request.agent_required`，保留的 `system` 返回 `request.agent_reserved` |

每次工具调用依次：

1. 写入与更新先检查轮次信息，缺少时以 `request.turn_required` 拒绝，不认证、不落位、不登记意图；
2. `resolve_principal_claim` 解析 principal 声明，`resolve_request_identity_claims(require_agent=True)` 解析用户、Workspace 与 Agent 声明。两者都在 `server/deps.py`，MCP 入口沿用 server 唯一的默认身份回退点，不复制解析规则；
3. `request_access_for_claims` 以声明的 principal、`mcp` adapter 与凭据经统一认证网关签发一份请求级 context。第一阶段核对接入登记、adapter、凭据与用户，以及能否声明 `system`，第二阶段做 Workspace 准入（[认证与访问登记](../workspace/authentication.md)）；
4. 写入与更新以同一份 context 经门面取得的 Import Bus 服务为这一轮落位，得到落点话题（第 4 节）；
5. 把工具调用归一为操作请求，以门面取得的操作入口的 `execute_request` 提交，写入与更新带上落点话题；
6. 调用结束（含失败与取消）时撤销 context。context 不离开本次调用。

## 3. 工具

| 工具 | 参数 | 归一为 | 授权 | 结果 |
|:---|:---|:---|:---|:---|
| `memory_search` | `query`：非空字符串 | `RetrieveRequest(semantic_query=query)` | `memory.search` | 检索上下文信封，每条含 alias、存档时间（记忆的更新时间）与置信度；没有结果时返回“没有检索到相关记忆。”，不算错误 |
| `memory_read` | `aliases`：非空的 alias 列表 | `ResolveReferencesRequest(aliases)` | `memory.read` | 按请求顺序逐项交付全文；未解析的 alias 以“未找到以下 alias：…”列在末尾；全部未解析时返回 `workspace.resource.not_found` 错误。交付正式原子时记录 `workspace.reference_read` 引用 |
| `memory_write` | `content`：非空字符串；`title`、`reason` 可选；`turn` | `SubmitWriteIntentRequest(WriteFocus(content, title, reason))` | `memory.write` | pending alias（`draft_` 开头），写明意图已登记并被生成队列接纳、正式记忆尚未保存，可以用 `memory_read` 读取 |
| `memory_update` | `alias`：要修正的正式记忆；`instruction`：非空字符串；`content` 可选；`turn` | `SubmitUpdateIntentRequest(base_alias=alias, instruction, content)` | `memory.update` | pending alias（`rev_` 开头），说明同上 |

- **参数**：检索不开放条数、过滤与关键词，使用默认条数。参数名 `query` 与生成视图读取 SEARCH 查询的参数名（`tool_args["query"]`）一致；`aliases`、`alias` 由连接器移为 READ、UPDATE 的作用对象，回声标记（第 6 节）因此得到带查询或 alias 的一行摘要。
- **轮次信息 `turn`**：写入与更新带一个可选参数 `turn`，类型即 Import Bus 的在途轮次模型（`source`、`external_conversation_id`、`external_turn_id`、`user_message`、`recent_turns`，校验规则与导入请求的同名字段相同）。它由连接器的 `PreToolUse` hook 在调用前填写（第 6 节），参数说明写明不要由模型自行填写。SDK 的工具参数模型会丢弃输入 schema 之外的键，因此它声明在 schema 中，而不是作为隐藏参数。操作请求只携带操作参数，不携带 `turn` 或话题。
- **编译**：检索结果与 Alice 的 MTP SEARCH 使用同一检索上下文信封与配置的检索策略，但关闭信封的头尾（编译选项 `include_header_footer=False`）：头尾是提示词中预检索上下文的引导，不适用于工具结果。读取以 `READ_RESULT` 目标逐项编译，哪些解析结果可以交付与 MTP READ 共用 `engines.memory_compiler.readable_source`；`memory_read` 用于取得检索结果中摘要之外的全文，因此不截断。编译文本中指引下一步操作的句子只写能力，不写任何协议的指令或工具名（[记忆编译](../patchouli/memory-compiler.md)）。
- **在途句柄**：写入与更新返回的 pending alias 经 `memory_read` 回读时，交付登记中的状态；生成链结算后，读取重定向到正式记忆（[读取视图与写入意图](../workspace/read-view-and-write-intents.md)）。
- **说明文字**：工具说明与 MCP 服务说明（instructions）写明何时检索（回答涉及项目历史、此前的决定或用户偏好的问题之前）、结果是带日期的历史证据（引用前先核实）、仓库中的代码与文档以及 harness 自己的指令与记忆文件优先于检索结果、检索得到的 alias 可以用 `memory_read` 读取全文；何时写入（用户要求记住的内容、新做出的决定、偏好与经验，对话本身会自动归档，不需要复述写入）与更新（已有记忆过时或有误，以检索得到的正式 alias 为对象，不能是 `draft_`、`rev_` 句柄）；以及写入与更新返回的是句柄。
- **标注**：检索与读取标注为只读（MCP 的 `readOnlyHint`），读取会记录引用，但不改变记忆内容；写入不是只读、也不破坏已有内容（`destructiveHint=false`）；更新改写一条已有记忆的当前版本，标注为破坏性。

## 4. 落位、操作入口与授权

MCP 入口与 Alice 的 MTP 适配器形态相同：都把协议调用归一为 `OperationRequest`，交给同一个操作入口，由入口分派到同一组 actor API 方法。写入与更新多一步落位。

- **落位**：入口以同一份请求级 context 调用门面取得的 `ImportBusService.place_turn(turn, access, target_workspace)`。它按导入授权（只校验准入与目标，不检查操作白名单）兑现 scope，在这一轮所属外部会话的串行门内返回落点话题：已有落点直接返回；这一轮已经回流并应用时返回它实际所在的话题；否则路由、准备话题并登记落点。规则见 [Import Bus](../system/import-bus.md#47-写入时落位) 第 4.7 节。目标 Workspace 取声明解析的结果，与导入入口相同。
- **请求级提交**：操作入口 `WorkspaceOperationEntry` 的 `execute_request(request, access, dispatch_topic_id=None)` 与线程提交的 `execute` 共用同一分派表。检索与读取不得给出派发话题，写入与更新必须给出；其余请求类型、或派发话题与请求类型不符时，在分派前以 `TypeError` 拒绝，没有副作用。它不做存活判定：请求级 context 只在本次调用内有效，撤销由授权检查。
- **actor API**：以 `authorize_actor_operation` 授权，接受执行线程与请求级 context，目标取驻留 Workspace，白名单照常逐次检查。写入与更新给出派发话题时以 `request_level=True` 授权，只接受请求级 context：线程 context 走这条路径会绕过线程的存活判定与线程的派发话题。给出派发话题时不解析线程，意图登记的进程与线程关联为空；派发、ACK、拒绝与结果未知的语义与线程路径相同，派发开始后调用方断开或取消不撤回写入（[读取视图与写入意图](../workspace/read-view-and-write-intents.md)）。请求级 context 到不了 RUN 与 CALL：经 `execute` 提交时被线程表拒绝，经 `execute_request` 提交时被类型表拒绝。
- **门面**：以 `operation_entry` 交出组合根装配的同一个操作入口，只供 MCP 入口调用 `execute_request`；以 `import_bus_service` 交出 Import Bus 服务。门面不暴露 actor API。
- **结构守护**：`execute_request` 只由 MCP 入口调用；MCP 入口不导入 actor API、root API、任务进程与 Import Bus 的服务和编排器，在 Import Bus 中只导入在途轮次的输入模型（[能力层](../workspace/capability.md)）。

没有写入许可的 Agent 调用写入时，落位仍先于 actor API 的授权发生。落位只是把这一轮的路由提前，这一轮回流时沿用同一落点，不会多一次路由。

## 5. 错误

MCP 入口复用 HTTP 出口的 `project_error` 投影，不另建映射；投影得到的 JSON（`error`、`detail`、可选的 `reason`，与 HTTP 错误响应体同构）写入 `isError` 的工具结果。

| 情况 | 来源 | 投影 |
|:---|:---|:---|
| 声明不合法 | `request.principal_required`、`request.identity_empty`、`request.agent_required`、`request.agent_reserved`、`request.workspace_not_found` | 原错误码 |
| 写入或更新缺少轮次信息 | `request.turn_required` | 原错误码（400） |
| 认证失败 | `AdmissionDeniedError`：`unknown_principal`、`adapter_mismatch`、`invalid_credential`、`actor_not_allowed_for_principal`、`system_actor_not_allowed`、`actor_not_owner`、`actor_not_admitted` | `workspace.admission_denied`，带 reason |
| 认证网关已关闭（系统停止中） | `AdmissionDeniedError`：`authentication_gateway_closed` | `workspace.admission_denied`，说明“认证服务已关闭” |
| 缺少操作许可 | `OperationDeniedError`：`operation_not_allowed` | `workspace.operation_denied`，带 reason |
| 读取的 alias 全部未解析；更新的基础 alias 不存在或不可见 | `ResourceNotFoundError` | `workspace.resource.not_found` |
| 更新的基础仍是在途句柄 | `PendingUpdateNotAllowedError` | `workspace.memory.intent.pending_update_not_allowed`（409） |
| 生成队列拒绝接纳 | `IntentDispatchRejectedError`，句柄已标记 FAILED | `workspace.memory.intent.dispatch_rejected`（503） |
| 派发结果未知 | `IntentDispatchUnknownError`，句柄保留 MATERIALIZING | `workspace.memory.intent.dispatch_unknown`（500） |
| 落位暂时无法完成 | `ImportBusUnavailableError`：Gateway 暂不可用（`gateway_unavailable`）、这一轮已提交仍在应用（`apply_pending`）、服务正在关闭（`shutting_down`） | `import_bus.unavailable`，带 reason，可稍后重试 |
| 存储不可用 | `StorageOfflineError`、`StorageReadError` | `storage.offline`、`storage.read_failed` |
| 其他异常 | —— | `server.internal_error`，异常正文不公开；在本次调用边界记录一次完整异常栈 |

参数不合法（空查询、空 alias 列表、`turn` 不符合在途轮次模型、类型不符）由 SDK 按工具的输入 schema 拒绝，返回 SDK 自己的 `isError` 结果，不经入口。`asyncio.CancelledError` 不是 `Exception`，照常传播，撤销照常发生。

## 6. 连接器：调用名、轮次信息与回声标记

Claude Code 为插件 MCP 工具生成的调用名为 `mcp__plugin_<插件名>_<server 名>__<工具名>`，`A-Z`、`a-z`、`0-9`、`_`、`-` 以外的字符替换为 `_`；会话记录中 `tool_use` 的 `name` 就是这个调用名。插件名为 `hivememory-connector`、server 名为 `hivememory`，四个工具因此分别为：

| 调用名 | `verb` | 作用对象 `targets` |
|:---|:---|:---|
| `mcp__plugin_hivememory-connector_hivememory__memory_search` | `SEARCH` | 无；查询留在参数 `query` |
| `mcp__plugin_hivememory-connector_hivememory__memory_read` | `READ` | 参数 `aliases`，移出参数 |
| `mcp__plugin_hivememory-connector_hivememory__memory_write` | `WRITE` | 无；参数 `turn` 去掉 |
| `mcp__plugin_hivememory-connector_hivememory__memory_update` | `UPDATE` | 参数 `alias`，移出参数；参数 `turn` 去掉 |

**轮次信息**：插件的 `hooks.json` 为写入与更新两个调用名注册 `PreToolUse` hook（matcher 是这两个调用名）。hook 从会话记录读出当前一轮，即最后一条由人发出的消息开启的一轮，以及之前最多三轮，按导入请求的同一规则构造轮次信息（同一函数生成导入请求体中的对应字段），从游标处的最近上文读起，与导入读取的范围相同；外部轮次标识与轮次结束时导入的相同，服务端因此推导出同一个交互标识。hook 输出 `hookSpecificOutput.updatedInput`：原参数加上 `turn`，模型自己填写的 `turn` 被覆盖。它不给权限决定，改写后的参数照常经过 Claude Code 的权限评估，用户确认写入时看到的就是带 `turn` 的参数。hook 只读会话记录与连接器状态，不登记会话记录、不唤醒同步、不做网络请求；读不到当前一轮或失败时不输出，退出码仍为 0，工具调用照常进行，由入口以 `request.turn_required` 拒绝。子代理中的调用得到的会话记录路径是主会话的会话记录，写入落在主会话的当前一轮。

**回声标记**：经 MCP 取回的记忆、写入与更新的内容都会作为普通工具调用留在会话记录里，本轮结束时随对话回流再次导入；生成视图若渲染它们，提取模型会把已有记忆或已显式写入的内容再提取一遍。连接器因此把本插件四个工具的调用归一化为 HiveMemory 能力调用：按上表写 `verb` 与作用对象，不上传调用名，也不上传 `turn`；其他工具写 `tool_name` 与完整参数。`aliases` 不是非空的字符串列表、`alias` 不是非空字符串时（模型传错了参数，MCP server 已以工具错误拒绝这次调用），读取与更新按普通工具写出，避免服务端因缺少作用对象拒绝整轮。Import Bus 接受这四个动词，READ 与 UPDATE 必须带作用对象（[Import Bus](../system/import-bus.md)）。生成视图对 `SEARCH`、`READ` 只渲染 `SEARCH: "查询"` 或 `READ: a, b` 一行摘要，不渲染结果，`WRITE`、`UPDATE` 不渲染，与 Alice 的 MTP 写出同形的事件、渲染相同（[Patchouli 生成](../patchouli/generation.md)）。交互记录与 Artifact 仍保存包括结果在内的原始事件。

Claude Code 改变命名规则时识别会静默失效，回声恢复，表现为重复记忆；`PreToolUse` 的 matcher 同样失效，写入与更新以缺少轮次信息被拒绝，模型在工具结果中看到错误。连接器单元测试把调用名固定为上表的格式，并核对插件清单、`.mcp.json`、`hooks.json` 的 matcher 与连接器常量一致。工具参数改名时，MCP server、识别表与 matcher 同时修改。

## 7. 观测

- 现有请求日志记录每个 `POST /mcp`，只记录路径与状态。
- MCP 入口为每次工具调用记录一行：工具名、声明的 principal 与 Agent、结果条数或未解析数（写入与更新为意图标识，失败时为错误码）、耗时；不记录查询、记忆正文、轮次信息与凭据。服务端错误以 error 级别连同异常栈记录，其余失败为 info。落位由 Import Bus 记录一行（见 [Import Bus](../system/import-bus.md) 第 6.2 节）。
- 不发布 RuntimeEvent。

## 8. 已知限制

- Host 检查只接受带端口的本机地址，远程部署与传输安全（TLS、OAuth）不在当前范围内。
- `turn` 声明在工具 schema 中：hook 未运行（被禁用或超时）时，模型若自行填写，写入会落到按编造内容路由的话题，编造的轮次不会回流，它的落点留到进程重启。
- 后台运行的子代理在用户发出下一条消息之后才写入时，写入落在那时的当前一轮，而不是启动它的那一轮：hook 输入只给出会话当时的当前一轮。
- 第一次写入要等一次 Gateway 路由，再等生成队列接纳；同一外部会话的上一轮仍在应用时，还要等它完成。
- 写入经过现行的查重，查重判出的 UPDATE 整体覆盖已有记忆，显式写入的记忆可能在之后的结算中被改写。

## 9. 代码与测试入口

- 入口包：`src/hivememory/mcp_server/server.py`（服务、工具、错误投影与日志）；接入：`src/hivememory/server/app.py`（路由与 lifespan）；声明解析：`src/hivememory/server/deps.py`；错误投影：`src/hivememory/server/error_handlers.py`。
- 操作入口：`src/hivememory/workspace/capability/operations.py`；落位：`src/hivememory/system/application/import_bus_service.py` 与 `src/hivememory/system/services/import_bus/ingestor.py`；编译：`src/hivememory/engines/memory_compiler/`。
- 插件声明：`integrations/claude-code/.mcp.json`、`integrations/claude-code/hooks/hooks.json`；轮次信息：`integrations/claude-code/connector/claude_code/hooks.py`；回声标记：`integrations/claude-code/connector/claude_code/reader.py`。
- 测试：`tests/unit/mcp_server/`（身份取自请求头、一次调用一份 context 并撤销、写入与更新先落位再带落点提交、缺少轮次信息与落位失败、错误投影、结果编译、不提供订阅）；`tests/integration/system/test_mcp_memory_access.py`（真实认证授权与操作入口下的检索、读取、拒绝，以及写入后读到在途句柄、结算后重定向、回流落在写入派发的话题）；`tests/integration/workspace/test_operation_requests.py`（请求级写入与更新、派发话题与请求类型不符时的拒绝）；`tests/unit/server/routers/test_app_lifespan.py`（`/mcp` 路由与会话管理器接线）；`tests/unit/architecture/test_access_boundaries.py`（调用面守护）；`tests/unit/connectors/`（`PreToolUse` 的轮次信息、回声标记与清单一致性）。
