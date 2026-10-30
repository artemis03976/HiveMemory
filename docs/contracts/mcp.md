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
  - src/hivememory/engines/memory_compiler/
  - integrations/claude-code/.mcp.json
  - integrations/claude-code/connector/claude_code/reader.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/capability.md
  - docs/workspace/authentication.md
  - docs/system/import-bus.md
last_reviewed: 2026-10-30
---

# MCP 记忆访问

外部 harness（当前为 Claude Code）经 HiveMemory 的 MCP 端点检索与读取记忆。它与 [MTP](./mtp.md) 并列：MTP 是 Alice 在生成循环中使用的文本协议，MCP 是外部 harness 使用的工具协议。两者都是**操作适配器**：把协议调用解析为固定类型的操作请求，经 workspace 的单一操作入口进入 actor API，不直接调用能力层，也不自行判定身份或权限。

plugin 模式下，外部 harness 的任务对 HiveMemory 不可见，没有任务进程，也就没有执行线程。每次工具调用因此只取得一份**请求级访问 context**，按 Workspace 访问登记逐次授权，调用结束即撤销。端点只提供检索与读取两个只读工具：写入意图的登记需要进程关联与派发话题，不建进程的调用取不到它们；RUN 的执行发生在执行单元的环境中，CALL 只属于 Alice。

## 1. 端点与传输

| 项目 | 规则 |
|:---|:---|
| 地址 | `POST /mcp`，由 server 应用以路由接入，登记在 SPA 兜底路由之前。以路由而不是挂载接入：挂载在 `/mcp` 的子应用只匹配 `/mcp/` 下的路径，插件声明的 `/mcp` 会被重定向 |
| 传输 | MCP Python SDK 的 streamable HTTP，无状态、JSON 响应：每个 HTTP 请求独立处理，不保存 MCP 会话，也没有服务端通知、资源或提示 |
| 订阅 | 不提供 `subscriptions/listen`，也不宣告工具、资源与提示的列表变更：工具在运行时不变，没有可推送的变更。该请求的响应是一条一直打开的流，只在会话管理器退出时结束；而 uvicorn 要等所有连接结束后才执行 server 的 lifespan 关闭，接受订阅会让任何在线的客户端（Claude Code 以 2026-07-28 协议连接时会发出该请求）都让服务停不下来 |
| 生命周期 | SDK 会话管理器由 server 的 lifespan 运行：在装配系统之前进入，在系统停止之后退出。挂进宿主后 SDK 自己的 lifespan 不会执行，没有运行会话管理器时每个请求都会失败。会话管理器每个进程只能运行一次 |
| Host 与 Origin | 保留 SDK 默认的检查，防止 DNS 重绑定：只接受带端口的本机地址（`localhost:*`、`127.0.0.1:*`、`[::1]:*`），其他主机名以 421 拒绝 |

## 2. 身份交接

身份只取自请求头。工具参数由模型生成，永远不作为身份来源；入口也不组装 `IdentityScope`。

| 请求头 | 规则 |
|:---|:---|
| `X-Principal-ID` | 必填，声明调用来源（Claude Code 为 `harness:claude-code`）；缺省或空串返回 `request.principal_required`，只含空白返回 `request.identity_empty` |
| `Authorization` | `Bearer <凭据>`；缺省或不是 Bearer 方案时凭据为空，同样交给网关，由网关以 `invalid_credential` 拒绝 |
| `X-User-ID` | 用户声明，缺省回退到默认用户；用户身份不做证明，沿用本地单用户部署的信任假设，由接入登记的 `allowed_user_ids` 收紧 |
| `X-Workspace-ID` | 只接受 `main_workspace`，其他值返回 `request.workspace_not_found` |
| `X-Agent-ID` | 必填的具体 Agent；缺失返回 `request.agent_required`，保留的 `system` 返回 `request.agent_reserved` |

每次工具调用依次：

1. `resolve_principal_claim` 解析 principal 声明，`resolve_request_identity_claims(require_agent=True)` 解析用户、Workspace 与 Agent 声明。两者都在 `server/deps.py`，MCP 入口沿用 server 唯一的默认身份回退点，不复制解析规则；
2. `request_access_for_claims` 以声明的 principal、`mcp` adapter 与凭据经统一认证网关签发一份请求级 context。第一阶段核对接入登记、adapter、凭据与用户，以及能否声明 `system`，第二阶段做 Workspace 准入（[认证与访问登记](../workspace/authentication.md)）；
3. 把工具调用归一为操作请求，以门面取得的操作入口的 `execute_request` 提交；
4. 调用结束（含失败与取消）时撤销 context。context 不离开本次调用。

## 3. 工具

| 工具 | 参数 | 归一为 | 授权 | 结果 |
|:---|:---|:---|:---|:---|
| `memory_search` | `query`：非空字符串 | `RetrieveRequest(semantic_query=query)` | `memory.search` | 检索上下文信封，每条含 alias、存档时间（记忆的更新时间）与置信度；没有结果时返回“没有检索到相关记忆。”，不算错误 |
| `memory_read` | `aliases`：非空的 alias 列表 | `ResolveReferencesRequest(aliases)` | `memory.read` | 按请求顺序逐项交付全文；未解析的 alias 以“未找到以下 alias：…”列在末尾；全部未解析时返回 `workspace.resource.not_found` 错误。交付正式原子时记录 `workspace.reference_read` 引用 |

- **参数**：不开放条数、过滤与关键词，检索使用默认条数。参数名 `query` 与生成视图读取 SEARCH 查询的参数名（`tool_args["query"]`）一致，`aliases` 由连接器移为 READ 的作用对象，回声标记（第 6 节）因此得到带查询或 alias 的一行摘要。
- **与 MTP 共用的编译**：检索结果与 MTP SEARCH 使用同一检索上下文信封与配置的检索策略，但关闭信封的头尾引导文字（编译选项 `include_header_footer=False`）：头尾面向 Alice，含预检索语境与 MTP 指令语法，会误导外部 agent。读取使用 MTP READ 的 `MTP_READ` 目标与同一份交付规则（`engines.memory_compiler.readable_source`），但不截断正文，因此不会出现指向 MTP READ 的截断提示。
- **说明文字**：工具说明与 MCP 服务说明（instructions）写明何时检索（回答涉及项目历史、此前的决定或用户偏好的问题之前）、结果是带日期的历史证据（引用前先核实）、仓库中的代码与文档以及 harness 自己的指令与记忆文件优先于检索结果，以及检索得到的 alias 可以用 `memory_read` 读取全文。
- **标注**：两个工具都标注为只读（MCP 的 `readOnlyHint`）。读取会记录引用，但不改变记忆内容。

## 4. 操作入口与授权

MCP 入口与 Alice 的 MTP 适配器形态相同：都把协议调用归一为 `OperationRequest`，交给同一个操作入口，由入口分派到同一组 actor API 方法。

- 操作入口 `WorkspaceOperationEntry` 的 `execute_request` 是请求级提交，只接受 `RetrieveRequest` 与 `ResolveReferencesRequest`，与线程提交的 `execute` 共用同一分派表；其余请求类型在分派前以 `TypeError` 拒绝，没有副作用。它不做存活判定：请求级 context 只在本次调用内有效，撤销由授权检查。
- actor API 以 `authorize_actor_operation` 授权：接受执行线程与请求级 context，目标取驻留 Workspace，白名单照常逐次检查。请求级 context 到不了 RUN、WRITE、UPDATE 与 CALL：经 `execute` 提交时被线程表拒绝，经 `execute_request` 提交时被类型表拒绝；WRITE 与 UPDATE 登记前要解析线程，CALL 派生先判定调用方线程存活。
- 门面以 `operation_entry` 交出组合根装配的同一个操作入口，只供 MCP 入口调用 `execute_request`；门面不暴露 actor API。
- 结构守护：`execute_request` 只由 MCP 入口调用；MCP 入口不导入 actor API、root API、任务进程与 Import Bus（[能力层](../workspace/capability.md)）。

## 5. 错误

MCP 入口复用 HTTP 出口的 `project_error` 投影，不另建映射；投影得到的 JSON（`error`、`detail`、可选的 `reason`，与 HTTP 错误响应体同构）写入 `isError` 的工具结果。

| 情况 | 来源 | 投影 |
|:---|:---|:---|
| 声明不合法 | `request.principal_required`、`request.identity_empty`、`request.agent_required`、`request.agent_reserved`、`request.workspace_not_found` | 原错误码 |
| 认证失败 | `AdmissionDeniedError`：`unknown_principal`、`adapter_mismatch`、`invalid_credential`、`actor_not_allowed_for_principal`、`system_actor_not_allowed`、`actor_not_owner`、`actor_not_admitted` | `workspace.admission_denied`，带 reason |
| 认证网关已关闭（系统停止中） | `AdmissionDeniedError`：`authentication_gateway_closed` | `workspace.admission_denied`，说明“认证服务已关闭” |
| 缺少操作许可 | `OperationDeniedError`：`operation_not_allowed` | `workspace.operation_denied`，带 reason |
| 读取的 alias 全部未解析 | `ResourceNotFoundError` | `workspace.resource.not_found` |
| 存储不可用 | `StorageOfflineError`、`StorageReadError` | `storage.offline`、`storage.read_failed` |
| 其他异常 | —— | `server.internal_error`，异常正文不公开；在本次调用边界记录一次完整异常栈 |

参数不合法（空查询、空 alias 列表、类型不符）由 SDK 按工具的输入 schema 拒绝，返回 SDK 自己的 `isError` 结果，不经入口。`asyncio.CancelledError` 不是 `Exception`，照常传播，撤销照常发生。

## 6. 工具调用名与回声标记

Claude Code 为插件 MCP 工具生成的调用名为 `mcp__plugin_<插件名>_<server 名>__<工具名>`，`A-Z`、`a-z`、`0-9`、`_`、`-` 以外的字符替换为 `_`；会话记录中 `tool_use` 的 `name` 就是这个调用名。插件名为 `hivememory-connector`、server 名为 `hivememory`，两个工具因此分别为：

| 调用名 | `verb` | 作用对象 `targets` |
|:---|:---|:---|
| `mcp__plugin_hivememory-connector_hivememory__memory_search` | `SEARCH` | 无；查询留在参数 `query` |
| `mcp__plugin_hivememory-connector_hivememory__memory_read` | `READ` | 参数 `aliases`，移出参数 |

经 MCP 取回的记忆会作为普通工具结果留在会话记录里，本轮结束时随对话回流再次导入；生成视图若渲染这些结果，提取模型可能把已有记忆再提取一遍。连接器因此把这两个工具的调用归一化为 HiveMemory 能力调用：按上表写 `verb` 与作用对象，不上传调用名；其他工具写 `tool_name` 与完整参数。`aliases` 不是非空的字符串列表时（模型传错了参数，MCP server 已以工具错误拒绝这次调用），读取按普通工具写出，避免服务端因 READ 缺少作用对象拒绝整轮。Import Bus 只接受 `SEARCH` 与 `READ`（[Import Bus](../system/import-bus.md)）。生成视图对带 `verb` 的调用只渲染 `SEARCH: "查询"` 或 `READ: a, b` 一行摘要，不渲染结果，与 Alice 的 MTP SEARCH、READ 写出同形的事件、渲染相同（[Patchouli 生成](../patchouli/generation.md)）。交互记录与 Artifact 仍保存包括结果在内的原始事件。

Claude Code 改变命名规则时标记会静默失效，回声恢复，表现为重复记忆，不丢失数据；连接器单元测试把调用名固定为上表的格式，并核对插件清单、`.mcp.json` 与连接器常量一致。

## 7. 观测

- 现有请求日志记录每个 `POST /mcp`，只记录路径与状态。
- MCP 入口为每次工具调用记录一行：工具名、声明的 principal 与 Agent、结果条数或未解析数（失败时为错误码）、耗时；不记录查询、记忆正文与凭据。服务端错误以 error 级别连同异常栈记录，其余失败为 info。
- 不发布 RuntimeEvent。

## 8. 已知限制

- 只提供检索与读取，不开放写入类工具。
- Host 检查只接受带端口的本机地址，远程部署与传输安全（TLS、OAuth）不在当前范围内。

## 9. 代码与测试入口

- 入口包：`src/hivememory/mcp_server/server.py`（服务、工具、错误投影与日志）；接入：`src/hivememory/server/app.py`（路由与 lifespan）；声明解析：`src/hivememory/server/deps.py`。
- 操作入口：`src/hivememory/workspace/capability/operations.py`；编译：`src/hivememory/engines/memory_compiler/`。
- 插件声明：`integrations/claude-code/.mcp.json`；回声标记：`integrations/claude-code/connector/claude_code/reader.py`。
- 测试：`tests/unit/mcp_server/`（身份取自请求头、一次调用一份 context 并撤销、错误投影、结果编译、不提供订阅）；`tests/integration/system/test_mcp_memory_access.py`（真实认证授权与操作入口下的检索、读取与拒绝）；`tests/unit/server/routers/test_app_lifespan.py`（`/mcp` 路由与会话管理器接线）；`tests/unit/architecture/test_access_boundaries.py`（调用面守护）；`tests/unit/connectors/`（回声标记）。
