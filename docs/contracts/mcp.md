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
  - integrations/hivememory/.mcp.json
  - integrations/hivememory/hooks/hooks.json
  - integrations/hivememory/connector/claude_code/hooks.py
  - integrations/hivememory/connector/claude_code/reader.py
  - integrations/hivememory/codex.mcp.json
  - integrations/hivememory/hooks/codex.json
  - integrations/hivememory/connector/codex/
  - integrations/hivememory/node/
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/capability.md
  - docs/workspace/authentication.md
  - docs/workspace/read-view-and-write-intents.md
  - docs/system/import-bus.md
last_reviewed: 2026-11-08
---

# MCP 记忆访问

外部 harness（当前为 Claude Code、DeepSeek Harness、Codex 与 pi）经 HiveMemory 的 MCP 端点检索、读取、写入与更新记忆。它与 [MTP](./mtp.md) 并列：MTP 是 Alice 在生成循环中使用的文本协议，MCP 是外部 harness 使用的工具协议。两者都是**操作适配器**：把协议调用解析为固定类型的操作请求，经 workspace 的单一操作入口进入 actor API，不直接调用能力层，也不自行判定身份或权限。记忆能力属于 actor API，不为哪一种协议独占；写入与更新经两种协议进入的是同一组 actor 操作、同一条生成链路。

plugin 模式下，外部 harness 的任务对 HiveMemory 不可见，没有任务进程，也就没有执行线程。每次工具调用因此只认证一次，得到一个**工作 actor**，它只是本次调用内的局部值，actor API 按 Workspace 访问登记逐次授权。写入与更新需要一个派发话题：线程提交向线程表取得进程这一轮的派发话题，请求级提交没有线程，入口给出一个回调，由 Import Bus 在写入时为这一轮确定话题（第 4 节）。RUN 的执行发生在执行单元的环境中，CALL 只属于 Alice，端点不提供这两项。

## 1. 端点与传输

| 项目 | 规则 |
|:---|:---|
| 地址 | `POST /mcp`，由 server 应用以路由接入，登记在 SPA 兜底路由之前。以路由而不是挂载接入：挂载在 `/mcp` 的子应用只匹配 `/mcp/` 下的路径，插件声明的 `/mcp` 会被重定向 |
| 传输 | MCP Python SDK 的 streamable HTTP，无状态、JSON 响应：每个 HTTP 请求独立处理，不保存 MCP 会话，也没有服务端通知、资源或提示 |
| 订阅 | 不提供 `subscriptions/listen`，也不宣告工具、资源与提示的列表变更：工具在运行时不变，没有可推送的变更。该请求的响应是一条一直打开的流，只在会话管理器退出时结束；而 uvicorn 要等所有连接结束后才执行 server 的 lifespan 关闭，接受订阅会让任何在线的客户端（Claude Code 以 2026-07-28 协议连接时会发出该请求）都让服务停不下来 |
| 生命周期 | SDK 会话管理器由 server 的 lifespan 运行：在装配系统之前进入，在系统停止之后退出。挂进宿主后 SDK 自己的 lifespan 不会执行，没有运行会话管理器时每个请求都会失败。会话管理器每个进程只能运行一次 |
| Host 与 Origin | 保留 SDK 默认的检查，防止 DNS 重绑定：只接受带端口的本机地址（`localhost:*`、`127.0.0.1:*`、`[::1]:*`），其他主机名以 421 拒绝 |

## 2. 身份交接

身份只取自请求头。工具参数由模型生成，永远不作为身份来源；连接器补上的轮次信息只用于确定话题，也不是身份。入口不组装 `IdentityScope`。

| 请求头 | 规则 |
|:---|:---|
| `X-Principal-ID` | 必填，声明调用来源（Claude Code 为 `harness:claude-code`，DeepSeek Harness 为 `harness:dsh`，Codex 为 `harness:codex`，pi 为 `harness:pi`）；缺省或空串返回 `request.principal_required`，只含空白返回 `request.identity_empty` |
| `Authorization` | `Bearer <凭据>`；缺省或不是 Bearer 方案时凭据为空，同样交给网关，由网关以 `invalid_credential` 拒绝 |
| `X-User-ID` | 用户声明，缺省回退到默认用户；用户身份不做证明，沿用本地单用户部署的信任假设，由接入登记的 `allowed_user_ids` 收紧 |
| `X-Workspace-ID` | 只接受 `main_workspace`，其他值返回 `request.workspace_not_found` |
| `X-Agent-ID` | 必填的具体 Agent；缺失返回 `request.agent_required`，保留的 `system` 返回 `request.agent_reserved` |

每次工具调用依次：

1. 写入与更新先检查轮次信息，缺少时以 `request.turn_required` 拒绝，不认证、不确定话题、不登记意图；
2. `resolve_principal_claim` 解析 principal 声明，`resolve_request_identity_claims(require_agent=True)` 解析用户、Workspace 与 Agent 声明。两者都在 `server/deps.py`，MCP 入口沿用 server 唯一的默认身份回退点，不复制解析规则；
3. `authenticate_request_actor` 以声明的 principal、`mcp` adapter 与凭据经统一认证网关认证一次，得到本次调用的工作 actor。第一阶段核对接入登记、adapter、凭据与用户，以及能否声明 `system`，第二阶段做 Workspace 准入（[认证与访问登记](../workspace/authentication.md)）；
4. 把工具调用归一为操作请求，以门面取得的操作入口的 `execute_request` 提交。写入与更新另给出确定话题的回调：它以同一个工作 actor 经门面取得的 Import Bus 服务为这一轮确定话题（第 4 节），actor API 授权通过（更新另需读到基础记忆）之后才调用它。工作 actor 不离开本次调用，调用返回即不再被引用。

## 3. 工具

| 工具 | 参数 | 归一为 | 授权 | 结果 |
|:---|:---|:---|:---|:---|
| `memory_search` | `query`：非空字符串 | `RetrieveRequest(semantic_query=query)` | `memory.search` | 检索上下文信封，每条含 alias、存档时间（记忆的更新时间）与置信度；没有结果时返回“没有检索到相关记忆。”，不算错误。为交付的记忆记录检索命中 |
| `memory_read` | `aliases`：非空的 alias 列表 | `ResolveReferencesRequest(aliases)` | `memory.read` | 按请求顺序逐项交付全文；未解析的 alias 以“未找到以下 alias：…”列在末尾；全部未解析时返回 `workspace.resource.not_found` 错误。交付正式原子时记录 `workspace.reference_read` 引用 |
| `memory_write` | `content`：非空字符串；`title`、`reason` 可选，`title` 由正式记忆沿用（[生成](../patchouli/generation.md#32-mode-b主动-write)第 3.2 节）；`turn` | `SubmitWriteIntentRequest(WriteFocus(content, title, reason))` | `memory.write` | pending alias（`draft_` 开头），写明意图已登记并被生成队列接纳、正式记忆尚未保存，可以用 `memory_read` 读取 |
| `memory_update` | `alias`：要修正的正式记忆；`instruction`：非空字符串；`content` 可选；`turn` | `SubmitUpdateIntentRequest(base_alias=alias, instruction, content)` | `memory.update` | pending alias（`rev_` 开头），说明同上 |

- **参数**：检索不开放条数、过滤与关键词，使用默认条数。参数名 `query` 与生成视图读取 SEARCH 查询的参数名（`tool_args["query"]`）一致；`aliases`、`alias` 由连接器移为 READ、UPDATE 的作用对象，回声标记（第 6 节）因此得到带查询或 alias 的一行摘要。
- **轮次信息 `turn`**：写入与更新带一个可选参数 `turn`，类型即 Import Bus 的在途轮次模型（`source`、`external_conversation_id`、`external_turn_id`、`user_message`、`recent_turns`，校验规则与导入请求的同名字段相同）。它由连接器的 `PreToolUse` hook 在调用前填写（第 6 节），参数说明写明不要由模型自行填写。SDK 的工具参数模型会丢弃输入 schema 之外的键，因此它声明在 schema 中，而不是作为隐藏参数。操作请求只携带操作参数，不携带 `turn` 或话题。
- **编译**：检索结果与 Alice 的 MTP SEARCH 使用同一检索上下文信封与配置的检索策略，但关闭信封的头尾（编译选项 `include_header_footer=False`）：头尾是提示词中预检索上下文的引导，不适用于工具结果。读取以 `READ_RESULT` 目标逐项编译，哪些解析结果可以交付与 MTP READ 共用 `engines.memory_compiler.readable_source`；`memory_read` 用于取得检索结果中摘要之外的全文，`READ_RESULT` 交付正式原子的全文。编译文本中指引下一步操作的句子只写能力，不写任何协议的指令或工具名（[记忆编译](../patchouli/memory-compiler.md)）。
- **在途句柄**：写入与更新返回的 pending alias 经 `memory_read` 回读时，交付登记中的状态；生成链结算后，读取重定向到正式记忆（[读取视图与写入意图](../workspace/read-view-and-write-intents.md)）。
- **说明文字**：工具说明与 MCP 服务说明（instructions）写明何时检索（回答涉及项目历史、此前的决定或用户偏好的问题之前）、结果是带日期的历史证据（引用前先核实）、仓库中的代码与文档以及 harness 自己的指令与记忆文件优先于检索结果、检索得到的 alias 可以用 `memory_read` 读取全文；何时写入（用户要求记住的内容、新做出的决定、偏好与经验，对话本身会自动归档，不需要复述写入）与更新（已有记忆过时或有误，以检索得到的正式 alias 为对象，不能是 `draft_`、`rev_` 句柄）；以及写入与更新返回的是句柄。
- **标注**：检索与读取标注为只读（MCP 的 `readOnlyHint`），检索会记录命中、读取会记录引用，但都不改变记忆内容；写入不是只读、也不破坏已有内容（`destructiveHint=false`）；更新改写一条已有记忆的当前版本，标注为破坏性。

## 4. 确定话题、操作入口与授权

MCP 入口与 Alice 的 MTP 适配器形态相同：都把协议调用归一为 `OperationRequest`，交给同一个操作入口，由入口分派到同一组 actor API 方法。写入与更新多给出一个确定话题的回调。

- **确定话题**：回调以同一个工作 actor 调用门面取得的 `ImportBusService.place_turn(turn, actor, target_workspace)`。确定话题是系统级操作：服务以系统自身的 principal 认证参与者驻留 Workspace 的 `system`，编排器以它调用一次 root API 的为一轮确定话题（查进度、路由与确认或建立话题都在这个方法内，入口 root 授权一次），不检查参与 Agent 的操作白名单，在这一轮所属外部会话的串行门内返回这一轮的话题：已确定话题时直接返回；这一轮已经回流并应用时返回它实际所在的话题；已提交仍在应用时以 `apply_pending` 拒绝；否则路由、确认或建立话题并登记已确定的话题。规则见 [Import Bus](../system/import-bus.md#47-写入时确定话题) 第 4.7 节。目标 Workspace 取工作 actor 驻留的 Workspace，与导入入口相同。
- **请求级提交**：操作入口 `WorkspaceOperationEntry` 的 `execute_request(request, actor, dispatch_topic=None)` 与线程提交的 `execute` 共用同一分派表。`dispatch_topic` 是返回派发话题的异步回调：写入与更新必须给出，检索与读取不得给出；其余请求类型、或回调与请求类型不符时，在分派前以 `TypeError` 拒绝，没有副作用。它不做存活判定：工作 actor 是入口调用链内的局部值，没有线程可判定。
- **actor API**：以 `authorize_agent` 授权，目标取工作 actor 驻留的 Workspace，白名单照常逐次检查。写入与更新接收线程句柄或派发话题的回调，二者恰有其一：线程提交给出句柄，向线程表取得派发话题；请求级提交给出回调。两种提交的顺序相同：授权，UPDATE 解析可读的正式基础原子，取得派发话题，再在无挂起点的同步段登记意图（线程提交在这里再解析一次线程），然后派发。CPU 一侧只持有不透明的线程句柄，取不到工作 actor，因此无法经请求级提交绕过线程的存活判定与派发话题（架构测试守护）。给出回调时不解析线程，意图登记的进程与线程关联为空；派发、ACK、拒绝与结果未知的语义与线程路径相同，派发开始后调用方断开或取消不撤回写入（[读取视图与写入意图](../workspace/read-view-and-write-intents.md)）。请求级提交到不了 RUN 与 CALL：`execute_request` 的类型表拒绝它们，`execute` 只接收线程句柄。
- **门面**：以 `operation_entry` 交出组合根装配的同一个操作入口，只供 MCP 入口调用 `execute_request`；以 `import_bus_service` 交出 Import Bus 服务。门面不暴露 actor API。
- **结构守护**：`execute_request` 只由 MCP 入口调用；MCP 入口不导入 actor API、root API、任务进程与 Import Bus 的服务和编排器，在 Import Bus 中只导入在途轮次的输入模型（[能力层](../workspace/capability.md)）。

没有写入许可的 Agent 调用写入时，actor API 在调用回调之前以 `operation_not_allowed` 拒绝，不路由、不建立话题，也不触发 LRU 结算；更新的基础记忆不存在或不可见时同样在确定话题之前拒绝。确定话题失败时错误交回这次写入，不登记意图。确定话题只是把这一轮的路由提前，这一轮回流时沿用同一话题，不会多一次路由。

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
| 确定话题暂时无法完成 | `ImportBusUnavailableError`：Gateway 暂不可用（`gateway_unavailable`）、这一轮已提交仍在应用（`apply_pending`）、服务正在关闭（`shutting_down`） | `import_bus.unavailable`，带 reason，可稍后重试 |
| 建立新话题时活跃池已满且 LRU 候选都被占用 | `TopicBusyError` | `patchouli.topic.busy`（409），可稍后重试 |
| 存储不可用 | `StorageOfflineError`、`StorageReadError` | `storage.offline`、`storage.read_failed` |
| 其他异常 | —— | `server.internal_error`，异常正文不公开；在本次调用边界记录一次完整异常栈 |

参数不合法（空查询、空 alias 列表、`turn` 不符合在途轮次模型、类型不符）由 SDK 按工具的输入 schema 拒绝，返回 SDK 自己的 `isError` 结果，不经入口。`asyncio.CancelledError` 不是 `Exception`，照常传播。

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

**DeepSeek Harness 插件**：dsh 的工具前置事件不能改写调用参数，上面的 `PreToolUse` 做法在 dsh 中不可行。插件因此不把这个端点配置为 dsh 的通用 MCP server，而是自己注册四个同名的 dsh 原生工具，执行时经本端点调用对应的 MCP 工具：

- **注册**：插件加载时请求 `initialize`（取服务说明）与 `tools/list`（无状态端点不需要先建立会话），把四个工具的说明与参数 schema 原样用于 dsh 工具，服务说明注册为 dsh 系统提示词的一节 `hivememory`；交给模型的 schema 去掉 `turn` 与只承载它的 `$defs`，模型看不到也无法填写。四个工具全部注册或全部不注册；连接失败时工具不出现，之后每次创建 Agent 时重试，直到成功一次。
- **调用**：请求头为 `X-Principal-ID: harness:dsh`、`Authorization: Bearer`、配置时的 `X-User-ID` 与 `X-Agent-ID`（默认 `dsh`）。写入与更新执行时，插件以调用方 Agent 的会话构造 `turn`：读取会话的实时记录，工具在一轮之中执行，最后一轮就是当前一轮；之前几轮与导入时取同样的已结束轮次，生成导入请求体对应字段的是同一个函数（[Import Bus](../system/import-bus.md#93-轮次的切分与映射)第 9.3 节）。模型给出的 `turn` 被覆盖；最后一轮已经结束时以工具错误拒绝，不发出请求。子 Agent 的会话不导入，子 Agent 中的调用沿会话头的 `parentSession` 找到顶层会话，以顶层会话最近的一轮构造 `turn`，这一轮已经结束也照用：已经回流时服务端返回它实际所在的话题（[Import Bus](../system/import-bus.md#93-轮次的切分与映射)第 9.3 节）。
- **结果**：工具结果的正文原样交给模型；`isError` 结果与网络错误作为 dsh 的工具错误交给模型。调用超时 60 秒，并随 dsh 的取消信号取消。
- **回声标记**：dsh 会话记录中的调用名就是工具名（`memory_search`、`memory_read`、`memory_write`、`memory_update`），回流时按上表归一化。插件测试与 `tests/unit/connectors/test_deepseek_harness_plugin.py` 核对插件的工具名、作用对象参数与 `turn` 参数与本 server 一致。

**Codex 插件**：Codex 与 Claude Code 合用插件目录，接入形态相同：插件声明本端点为 MCP server，`PreToolUse` hook 补上轮次信息（[Import Bus](../system/import-bus.md#10-codex-插件)第 10 节）。

- **声明**：`codex.mcp.json` 中的 server `hivememory`（`type: http`，地址 `http://127.0.0.1:8769/mcp`），凭据取自 `bearer_token_env_var: HIVEMEMORY_CODEX_TOKEN`，静态请求头 `X-Principal-ID: harness:codex` 与 `X-Agent-ID: codex`，`X-User-ID` 取自 `HIVEMEMORY_USER_ID`（未设置时不发送）；`default_tools_approval_mode: approve`，四个工具不逐次审批。
- **调用名**：模型看到的是 namespace `mcp__hivememory` 下的四个工具；hook 输入中的工具名为 `mcp__hivememory__<工具名>`，`hooks/codex.json` 的 matcher 为 `^mcp__hivememory__(memory_write|memory_update)$`；会话记录以结构化的 `server` 与 `tool` 记下调用，回声标记按 `server` 为 `hivememory` 识别，归一化与上表相同。
- **轮次信息**：hook 以输入中的 `turn_id` 为当前一轮，在会话记录中读出它与之前最多三轮，按导入请求的同一规则构造 `turn`，输出 `permissionDecision: allow` 与 `updatedInput`（Codex 只在给出 `allow` 时采用改写），模型自己填写的被覆盖。子 Agent 中的调用归到根会话中派生它的那一轮。读不到时不输出，由入口以 `request.turn_required` 拒绝。
- Codex 的 hook 要在 Codex 中审阅信任后才运行；`tests/unit/connectors/test_codex_reader.py` 核对 Codex 清单、MCP 声明、matcher 与连接器常量和接入登记一致。

**pi 扩展**：接入方式与 dsh 插件相同（owner 决定，2026-11-08）：扩展不把本端点配置为 pi 内建 MCP 的 server，而是自己注册四个同名的 pi 工具，执行时经本端点调用对应的 MCP 工具（[Import Bus](../system/import-bus.md#11-pi-扩展)第 11 节）。pi 的 `tool_call` 虽能改写参数，但扩展没有加载时服务端无法分辨模型自己填写的 `turn`；内建 MCP 默认把工具交给 codemode，内层调用不进入会话记录，回声标记会失效。

- **注册**：会话开始后请求 `initialize` 与 `tools/list`，把四个工具的说明与参数 schema 原样用于 pi 工具，服务说明作为系统提示词的一节 `hivememory`；交给模型的 schema 去掉 `turn`。工具注册为 `exposure: "model-only"`，codemode 与其他工具不能调用。四个工具全部注册或全部不注册；连接失败时工具不出现，之后每次开始运行时重试，直到成功一次。
- **调用**：请求头为 `X-Principal-ID: harness:pi`、`Authorization: Bearer`（凭据取自 `HIVEMEMORY_PI_TOKEN`）、配置时的 `X-User-ID` 与 `X-Agent-ID`（默认 `pi`）。写入与更新执行时，扩展以这一轮开始运行时生成的轮次标识构造 `turn`，之前几轮与导入时取同样的已结束轮次，生成导入请求体对应字段的是同一个函数；模型给出的 `turn` 被覆盖，不在一轮之中时以工具错误拒绝，不发出请求。
- **结果与回声标记**：工具结果的正文原样交给模型，`isError` 结果作为 pi 的工具错误；pi 会话记录中的调用名就是工具名，回流时按上表归一化。`tests/unit/connectors/test_pi_extension.py` 核对扩展的工具名、作用对象参数与 `turn` 参数与本 server 一致。

## 7. 观测

- 现有请求日志记录每个 `POST /mcp`，只记录路径与状态。
- MCP 入口为每次工具调用记录一行：工具名、声明的 principal 与 Agent、结果条数或未解析数（写入与更新为意图标识，失败时为错误码）、耗时；不记录查询、记忆正文、轮次信息与凭据。服务端错误以 error 级别连同异常栈记录，其余失败为 info。确定话题由 Import Bus 记录一行（见 [Import Bus](../system/import-bus.md) 第 6.2 节）。
- 每次工具调用在自己的 `mcp-` trace 中进行（`span_name` 为工具名，`task_type` 为 `foreground`），结束时复原。检索、读取、写入与更新经操作入口各发一个 `actor.operation`（[事件契约](./routes-and-events.md#4-runtimeevent-观测契约)第 4 节），请求级提交不带 `process_id`；写入与更新为这一轮确定话题引起的 Gateway 被动路由与新建话题的事件继承同一个 trace，与这次工具调用归为一次运行。工具调用本身不另发事件。

## 8. 已知限制

- Host 检查只接受带端口的本机地址，远程部署与传输安全（TLS、OAuth）不在当前范围内。
- dsh 的 PTC 模式（默认关闭）下，记忆工具在 `run_code` 内部调用，会话记录只有外层调用，回声标记不生效；写入与更新仍按调用方会话的当前一轮确定话题。
- `turn` 声明在工具 schema 中：hook 未运行（被禁用或超时）时，模型若自行填写，写入会落到按编造内容路由的话题，编造的轮次不会回流，它已确定话题的登记留到进程重启。
- Codex 的 hook 未信任、被禁用或出错时同样不会覆盖模型填写的 `turn`；每次更新 Codex 插件都要重新信任 hook。Codex 的代码执行模式中间接调用的记忆工具不做回声标记。
- pi 的记忆工具不对 codemode 开放，由 codemode 编排的任务不能在脚本中访问记忆。
- 后台运行的子代理在用户发出下一条消息之后才写入时，写入落在那时的当前一轮，而不是启动它的那一轮：hook 输入只给出会话当时的当前一轮。dsh 的子 Agent 同样落在顶层会话那时最近的一轮。
- 第一次写入要等一次 Gateway 路由，再等生成队列接纳；同一外部会话的上一轮仍在应用时，还要等它完成。
- 写入经过查重：与已有记忆相似时，内容先保存为一条新记忆，之后由合并任务并入已有记忆，`memory_write` 返回的句柄随之改指到保留的记忆；合并完成之前读取句柄得到新建的那条（[记忆生成](../patchouli/generation.md#41-合并任务)）。改指事件丢失时句柄指向已删除的记忆，读不到正式目标。

## 9. 代码与测试入口

- 入口包：`src/hivememory/mcp_server/server.py`（服务、工具、错误投影与日志）；接入：`src/hivememory/server/app.py`（路由与 lifespan）；声明解析：`src/hivememory/server/deps.py`；错误投影：`src/hivememory/server/error_handlers.py`。
- 操作入口：`src/hivememory/workspace/capability/operations.py`；确定话题：`src/hivememory/system/application/import_bus_service.py` 与 `src/hivememory/system/services/import_bus/ingestor.py`；编译：`src/hivememory/engines/memory_compiler/`。
- 插件声明：`integrations/hivememory/.mcp.json`、`integrations/hivememory/hooks/hooks.json`；轮次信息：`integrations/hivememory/connector/claude_code/hooks.py`；回声标记：`integrations/hivememory/connector/claude_code/reader.py`；Codex 的声明：`integrations/hivememory/codex.mcp.json`、`integrations/hivememory/hooks/codex.json`，轮次信息与回声标记：`integrations/hivememory/connector/codex/`；dsh 插件的工具注册与轮次信息：`integrations/hivememory/node/dsh/tools.js`，回声标记：`integrations/hivememory/node/dsh/turns.js`、`integrations/hivememory/node/core/memory-tools.js`；pi 扩展的工具注册：`integrations/hivememory/node/pi/tools.js`，轮次信息与回声标记：`integrations/hivememory/node/pi/turns.js`、`integrations/hivememory/node/core/memory-tools.js`。
- 测试：`tests/unit/mcp_server/`（身份取自请求头、一次调用认证一次、写入与更新以确定话题的回调提交、缺少轮次信息与确定话题失败、错误投影、结果编译、不提供订阅）；`tests/integration/system/test_mcp_memory_access.py`（真实认证授权与操作入口下的检索、读取、拒绝，每次工具调用认证一次、写入时确定话题由系统认证执行，被拒的写入不确定话题，以及写入后读到在途句柄、结算后重定向、回流落在写入派发的话题）；`tests/integration/workspace/test_operation_requests.py`（请求级写入与更新、回调与请求类型不符时的拒绝、更新在基础冷读之后与登记之前确定话题、被拒的请求不确定话题）；`tests/unit/server/routers/test_app_lifespan.py`（`/mcp` 路由与会话管理器接线）；`tests/unit/architecture/test_access_boundaries.py`（调用面守护）；`tests/unit/connectors/`（Claude Code 与 Codex 的 `PreToolUse` 轮次信息、回声标记与清单一致性，dsh 插件、pi 扩展与本 server 的工具名、参数一致）；`integrations/hivememory/node/test/`（dsh 插件与 pi 扩展的工具注册、轮次信息与回声标记，`npm test`）。
