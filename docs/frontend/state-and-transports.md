---
title: Frontend State Persistence and Transports
status: current
owner: frontend
scope: state-ownership-persistence-http-sse-and-websocket
code_paths:
  - frontend/src/stores/
  - frontend/src/services/
  - frontend/src/transports/
  - frontend/src/transports/connectionStatus.ts
  - frontend/src/utils/debugMode.ts
  - frontend/src/constants/settings.ts
  - frontend/vite.config.ts
  - src/hivememory/server/routers/
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/README.md
last_reviewed: 2026-11-11
---

# 前端状态、持久化与传输

浏览器可以记住“用户怎样使用界面”，却不应自行决定“系统实际上发生了什么”。这是前端状态设计的首要边界：主题、面板和筛选属于可恢复偏好；Topic、Memory、Config、日志和 task 属于后端事实或短期运行投影。把两者一起塞进 localStorage 会让刷新看似恢复，实则复活已经失效的业务状态。

## 1. 状态所有权原则

```text
UI preference
  -> browser localStorage may own a durable copy

Draft / optimistic projection
  -> browser owns only until submit or reconciliation

ConversationSession / Topic / Memory / Agent Profile / Config / Task
  -> backend owns truth; frontend refetches and projects

Chat tokens / logs / RuntimeEvents
  -> runtime stream projection; no durable browser truth
```

`chat-store` 的 persist 版本为 4，`partialize` 只保存 `currentSessionId` 与 `currentAgentId`。启动后从服务端读取会话历史；旧版本状态不会复活消息或旧 Topic 指针。用户以 `#` 指定的话题 `designatedTopic` 也只在运行期保存，发出这一轮后清除，刷新后不会把一个可能已经结算的话题带进下一轮。读取 404 时清除选择，其他失败保留错误并阻止发送，直到成功加载或新建会话。会话历史的权威副本仅在后端进程内，浏览器不保存消息正文。

压缩进度是 SSE 的运行期投影，检查点是会话 GET 的快照，两者都不写入 localStorage。压缩结束后从服务端刷新检查点，不根据 token 或进度合成本地摘要；历史视图代次隔离晚到的读取，选择会话、清空和新一轮都会使旧读取失效。原始历史和边界标记的展示见 [Chat 视图](./chat-view.md#4-会话恢复与-topic-标注)。

浏览器的 localStorage 也应被视为一个需要版本治理的本地契约，而不是“随手缓存”。当前持久化 store 已采用 `name`、`version`、`migrate` 或 `partialize` 的组合来清理旧结构、缩小写入面并恢复可解释的小状态；新增 store 或改变字段时也必须遵循同一规则。优先保存主题、tab、折叠、筛选和短小的运行偏好，谨慎保存可重新解释的草稿，避免把过期的 Agent ID、Topic 指针、消息块或后端实体整棵复活。

## 2. 本地持久化矩阵

| Store | 持久化 | 明确不持久化 |
|:---|:---|:---|
| `chat-ui-store` | 主导航、主题（深色、浅色或跟随系统）、Chat 两栏的收起状态、右栏标签（引用记忆、记忆任务、模型配置）、Settings 分类、记忆预检索开关、运行页的列表筛选 | toast、页面业务数据、Chat 抽屉的开合、运行页的选中运行与日志 trace 筛选 |
| `chat-runtime-config-store` | 会话的模型选择 `modelSelection`（`model_id` 与 temperature/top_p/max_tokens）、是否覆盖参数；持久化版本 3，旧版本的 `generationOptions` 在加载时迁移 | 某次请求终态 |
| `chat-store` | 当前 Session ID 与 Agent ID | messages、会话列表/历史、输入草稿、指定话题 `designatedTopic`、引用记忆、连接、process ID、run status、压缩进度/检查点、本轮失败码 |
| `memory-view-store` | 记忆图书馆的标签（长期记忆或短期话题池）、search mode、类型、排序、grid/list | search query、selected tags、Memory 数据 |
| `memory-task-store` | 是否显示 terminal tasks | task、选中 task、请求状态 |
| `workspace-store` | 日志筛选（级别、logger 前缀、文本）与 ui 偏好（暂停、缓冲上限） | logs、RuntimeEvents、连接、主窗口状态 |
| `topic-store` | 无 | Topic 池、记忆图书馆中选中的话题 `selectedTopicId`、loading、error |

`chat-ui-store` 的持久化版本为 7（`stores/ui/chatUiMigration.ts`）。主题的缺省值是跟随系统，只作用于没有保存过主题的新安装，已经保存为深色或浅色的值原样保留。加载旧版本状态时依次迁移：已移除的 Workspace Vision 终端标签改为引用记忆（v5）；原来的通用设置、内部引擎与各子系统分类并入“常用”，提供商凭证、模型注册表、接入与权限保持不变（v6）；左栏不再分标签，删除左栏标签，原来停在左栏模型配置的改为右栏的模型配置（v7）。两栏的收起状态继续持久化，只在窗口放得下该栏时生效。桌面客户端的窗口使用不落盘的会话分区，这些偏好只在一次运行内有效（[桌面客户端](./desktop-client.md#10-已知限制)）。

Settings 草稿使用组件 state/useDraft，是否有未保存的更改另记在不持久化的 `stores/ui/unsavedSettings.ts` 中，供离开设置页前提醒；OmniInput 草稿使用 chat store 的运行期字段，供无输出停止时回填，两者都不跨刷新恢复。Topic `settle`/`delete` 使用乐观移除并在失败时回滚；这里的 `settle` 是结束 Topic 生命周期并交给记忆生成，不是 Memory Library 的 `archive`。Memory 删除同样先移除，后端删除失败时放回并提示（[管理页面](./management-views.md)）。

## 3. 当前传输

| 能力 | 传输 | 当前入口 | 终态/恢复语义 |
|:---|:---|:---|:---|
| 统一身份上下文 | HTTP 请求头 | `x-user-id` / `x-workspace-id`（`services/identity.ts`） | 所有身份相关请求统一携带；后端只从请求头读取身份选择 |
| Chat | fetch-based SSE | `POST /api/v1/chat` | `done` / `error`；无断线续传 |
| Stop | HTTP | `POST /api/v1/chat/stop` | best-effort 请求，等待流终态确认；后端拒绝时恢复运行状态 |
| Sessions / Topics / Memories / Agents / Config / Registries / Tasks | HTTP JSON | 相对 `/api/v1/...` | 请求级成功失败；页面按能力 refetch 或乐观更新 |
| RuntimeEvent | EventSource SSE | `/api/v1/runtime-events/stream` | 新连接以 `/runtime-events/status` 返回的 `buffer_size` 作为 `replay_last`，取回后端缓冲内的全部事件；支持 `last_event_id`/gap 的后端语义：`Last-Event-ID` 请求头优先于 URL 中的回放点，浏览器自动重连沿用首次连接的 URL 时按最新收到的事件续传；浏览器只保留当前运行期 |
| Logs | WebSocket | `/api/v1/ws/logs` | ping/pong、有限重连，无历史恢复 |

Chat 使用 fetch 而不是原生 EventSource，是因为需要以 POST 发送请求体。RuntimeEvent 是 GET 流，使用 EventSource。日志需要双向 ping/pong 和持续推送，使用 WebSocket。

身份选择传输约定：Session、Memory、Memory task、Agent、Topic、访问记录、Chat 与 stop 请求统一携带 `x-user-id`/`x-workspace-id` 基础选择；后端不接受 Topic 等管理路由的 `?user_id=` query；Chat/stop 请求体不携带身份字段。管理请求（Session/Memory/Memory task/Agent/Topic/访问记录）默认依赖统一请求头，不需要额外参数。

### 3.1 HTTP 错误解析

`services/apiError.ts` 的 `readApiError()` / `apiErrorMessage()` 供 HTTP services 与 SSE 握手共用。后端 `error/detail/reason/issues` 以安全 detail 展示、code/reason 分支；兼容旧 detail 字符串、message 字符串、detail.message 与 422 校验列表。旧校验列表只读取字段位置和受控类别，不展示 input/ctx/validator 原文。未知对象、HTML 和空体使用带 HTTP status 的默认提示，正文读取的 AbortError 继续抛出。Session 等服务保留原有状态异常。后端信封与迁移说明见[错误模型](../contracts/error-model.md#7-http-api-与客户端呈现)。

`readApiError()` 返回的 `ApiError` 是一种 `Error`，带 HTTP status、code 与 reason，表示请求已经到达后端、被后端拒绝；网络失败与流中断不是它。界面据此区分“后端拒绝”与“连不上后端”：

- Chat 在开流之前被拒绝（准入、会话占用、管理凭据等）时，回复气泡显示后端给出的原因；只有连不上后端或流中断才显示“系统错误，请检查后端服务器”；
- Stop 被拒绝时恢复为 streaming，可以再次停止；
- Agent 列表读取被拒绝时，Agent 管理页显示后端给出的原因；
- RuntimeEvent 的状态接口被拒绝时报连接错误并显示原因，不当作未启用。浏览器在事件流收到非 200 响应后不再重连，也不交出状态码，未启用的 503 与凭据失效的 401 看起来一样，前端此时再请求一次状态接口来区分：被拒绝显示原因，未启用显示未启用，状态接口说已启用则按连接失败处理、由用户重试。

### 3.2 连接状况与局部错误

连不上后端与管理凭据失效会让每个页面同时读取失败。如果各页面各自写出原因，同一件事会在侧栏、提示与页面中重复多次。因此连接类的原因由页面顶部的一条状态条统一说明（[应用壳与视觉系统](./application-shell.md)），状态条的依据是 `transports/connectionStatus.ts` 维护的连接状况：

- 所有 HTTP 请求与 Chat 的 fetch SSE 都经 `backendFetch` 发出。请求在网络层失败（fetch 抛出 `TypeError`）时置位“连不上后端”，之后任一请求得到后端的响应即清除；请求被取消不算连接问题。
- `readApiError()` 遇到管理凭据缺失或不一致的错误码 `request.management_credential_invalid` 时置位“凭据失效”。只有刷新页面才能取得新的凭据，所以它不随后续响应清除。
- RuntimeEvent 的 `EventSource` 与日志 WebSocket 不是 fetch，不报告连接状况；事件流开启前的状态接口请求经 `backendFetch`。

页面的局部读取失败在捕获错误时经 `localErrorText()` 处理：状态条正在说明连不上后端或凭据失效时，局部只写“数据暂不可用”，其他失败照常显示各自的原因。它覆盖会话列表与历史、Agent 列表、记忆库的读取与语义检索、短期话题池、记忆任务、运行时事件流，以及提供商与模型注册表的加载提示。停止、重命名等操作失败仍给出原因，因为用户需要知道这一次操作为什么没有成功。

## 4. Origin 与代理

本地开发时，Vite 在 `127.0.0.1:5173` 运行，并将 `/api` 代理到 `http://localhost:8769`，包括 WebSocket upgrade。Docker/整合部署由 FastAPI 在 `8000` 同源提供前端和 API。

`/api` 下的管理接口要求管理凭据（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)第 1 节），前端源码不读取也不保存它，fetch、`EventSource` 与 WebSocket 客户端都不处理凭据：

- Vite 代理每次转发 HTTP 与 WebSocket 请求时，从 `../configs/management.token` 读出凭据并附上 `Authorization` 请求头；文件尚未生成时不附加，后端以 401 拒绝；
- 后端提供页面时，页面响应下发 `HttpOnly` 的 `hivememory_management` cookie，同源请求、SSE 与 WebSocket 自动携带它。
- 桌面客户端中后端不下发 cookie，由主进程把同名 cookie 放进窗口的内存会话，页面的携带方式不变（[桌面客户端](./desktop-client.md#4-管理凭据)）。

凭据缺失或不一致的 401 由第 3.1 节的错误解析换成“管理凭据缺失或已失效，请刷新页面后重试”，同时置位连接状况中的凭据失效，页面顶部出现带“刷新页面”按钮的状态条（第 3.2 节）：后端的说明文字面向所有调用方，而页面由后端提供时，后端换了凭据之后刷新页面即可取得新的 cookie。日志 WebSocket 的握手被拒时浏览器拿不到原因，与后端未启动一样按断线有限重连。

HTTP、SSE（chat 与 RuntimeEvent）与日志 WebSocket 的 URL 都由 `transports/backendUrl.ts` 生成，三者始终连接同一个后端：构建或开发服务器设置 `VITE_BACKEND_ORIGIN` 时直连这个 origin；不设置时（默认）HTTP 与 SSE 保持相对路径，WebSocket 取页面 origin，由同源部署的后端或 Vite 代理承接。因此：

- 同源部署与 Vite proxy 是当前完整支持路径；
- 设置 `VITE_BACKEND_ORIGIN` 只改变请求的目标：直连的请求不经 Vite 代理，也不在后端页面的同源 cookie 范围内，不带管理凭据，会被管理凭据检查拒绝；后端 CORS 也只允许固定的本地开发 origin（见下文）。真正的前后端分域部署目前仍需要反向代理。

另一个构建变量 `VITE_DEBUG` 与 origin 无关，只决定读取失败时是否换用 mock 数据，见第 6 节。

后端 CORS 当前允许若干本地开发 origin，没有面向任意生产域名的配置化清单。

## 5. 多窗口日志同步

[全局运行页](./run-view.md)挂载时建立日志 WebSocket 与事件流，Chat 页挂载时建立事件流（右栏收为抽屉或收起时也要接收记忆任务的事件）；离开页面不断开，记忆任务面板依赖事件流。Workspace store 通过 `hivememory_primary_window` localStorage heartbeat 判断主窗口。主窗口建立 WebSocket 与 RuntimeEvent 连接，再通过 `hivememory_workspace_logs` BroadcastChannel 广播新日志、事件和连接状态；次窗口只消费广播。

这个机制减少重复连接，但没有服务端 lease，也没有跨浏览器实例协调。浏览器冻结、localStorage 清理、时钟漂移或主窗口非正常终止时可能短暂出现无主或多主；它不能替代后端订阅身份和可靠消费语义。

## 6. Mock 降级边界

mock 数据只在 debug 模式下启用。debug 模式只在 Vite 构建或开发服务器显式设置 `VITE_DEBUG=true` 时开启（`utils/debugMode.ts`），`npm run dev` 与生产构建默认都不开启：

| 读取 | 默认：读取失败时 | debug 模式：读取失败时 |
|:---|:---|:---|
| Memory 列表（`hooks/useMemories.ts`） | 清空列表并显示错误 | 换用 `MOCK_MEMORIES`，不显示错误 |
| Agent 列表（`hooks/useAgentManagement.ts`） | 列表为空，页面显示后端给出的原因；没有后端原因时显示“Agent 列表读取失败”，连不上后端或凭据失效时只写“数据暂不可用” | 换用 `MOCK_AGENT_CONFIGS` |
| 主配置（`hooks/useSettings.ts`） | 配置保持未加载，常用与高级两页显示“设置暂不可用”；“恢复本节默认值”读取默认配置失败时草稿不变，以错误提示“没有读到默认值，这一节没有恢复” | 加载 `MOCK_CONFIG` 并保存 error；“恢复本节默认值”读取失败时取 `MOCK_CONFIG` 中这一节的值 |
| Memory task（`services/memoryTaskApi.ts`） | 错误原样抛出 | 列表、按 ID 读取与取消换用 mock task |

Session、Provider、Model、Topic 与 Chat 不使用同类数据回退。

前端不重复定义主配置的结构：`HiveMemoryConfig`（`types/config.ts`）是不展开字段的对象类型，设置页与配置校验都按字段路径读写（`readPath` 与设置页的字段表），后端测试核对字段表中的每条路径都存在于主配置。因此 `MOCK_CONFIG`（`constants/settings.ts`）按后端主配置的真实结构书写，只含前端读取的字段，取值为 `configs/config.yaml` 的默认值；`frontend/tests/settings-mock-config.test.mjs` 校验它为设置页的每个字段给出与控件相符的值，避免 debug 模式下设置页读到空字段。

Mock 的用途是让组件开发在脱离后端时仍有可见状态，不是把离线体验冒充真实系统，因此默认关闭，读取失败如实显示。debug 模式下页面没有 mock 标识，记忆、Agent 与配置的写操作仍指向后端并失败，只适合调界面。

## 7. 不变量与失败处理

- localStorage 只保存偏好、视图和可安全重新解释的选择，不保存权威业务实体；
- SSE、WebSocket 与 RuntimeEvent 的连接状态不等于业务终态；
- 顶部状态条的连接状况只由请求的实际结果更新（网络层失败、后端响应、凭据错误码），访问后端的新请求必须经 `backendFetch`，页面不自行推测连接问题；
- 乐观更新必须有回滚或显式 reconciliation；
- mock 数据只在 debug 模式下出现，不能参与真实写请求后再显示成功；
- transport 变更必须同时检查开发 proxy、整合部署、反向代理和多窗口行为；
- store migration 负责清理旧结构，不能让过期 model ID、消息块或 nav tab 在升级后复活。

## 8. 当前限制

- 没有 chat SSE 的 last-event-id 或断线续传；会话历史可重新读取，但后端重启后丢失；
- 没有统一 query cache、请求去重或 stale-while-revalidate 层；
- 基础身份选择当前固定为 `user_id=default + main_workspace`（`services/identity.ts`），没有认证 token、租户或 per-user storage namespace；
- 直连后端（`VITE_BACKEND_ORIGIN`）的请求不带管理凭据，后端 CORS 没有可配置的 origin 清单；
- 会话已有自动化行为测试；尚无完整的持久化迁移矩阵与跨窗口一致性测试。

后端 route 与 event 的唯一契约见[公开路由与事件](../contracts/routes-and-events.md)，Chat 事件如何进入页面见 [Chat 视图](./chat-view.md)。
