---
title: Frontend State Persistence and Transports
status: current
owner: frontend
scope: state-ownership-persistence-http-sse-and-websocket
code_paths:
  - frontend/src/stores/
  - frontend/src/services/
  - frontend/src/transports/
  - frontend/vite.config.ts
  - src/hivememory/server/routers/
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/README.md
last_reviewed: 2026-10-18
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

`chat-store` 的 persist 版本为 4，`partialize` 只保存 `currentSessionId` 与 `currentAgentId`。启动后从服务端读取会话历史；旧版本状态不会复活消息或旧 Topic 指针。读取 404 时清除选择，其他失败保留错误并阻止发送，直到成功加载或新建会话。会话历史的权威副本仅在后端进程内，浏览器不保存消息正文。

压缩进度是 SSE 的运行期投影，检查点是会话 GET 的快照，两者都不写入 localStorage。压缩结束后从服务端刷新检查点，不根据 token 或进度合成本地摘要；历史视图代次隔离晚到的读取，选择会话、清空和新一轮都会使旧读取失效。原始历史和边界标记的展示见 [Chat 视图](./chat-view.md#4-会话恢复与-topic-标注)。

浏览器的 localStorage 也应被视为一个需要版本治理的本地契约，而不是“随手缓存”。当前持久化 store 已采用 `name`、`version`、`migrate` 或 `partialize` 的组合来清理旧结构、缩小写入面并恢复可解释的小状态；新增 store 或改变字段时也必须遵循同一规则。优先保存主题、tab、折叠、筛选和短小的运行偏好，谨慎保存可重新解释的草稿，避免把过期的 Agent ID、Topic 指针、消息块或后端实体整棵复活。

## 2. 本地持久化矩阵

| Store | 持久化 | 明确不持久化 |
|:---|:---|:---|
| `chat-ui-store` | 主导航、主题、左右面板折叠、侧栏 tab、Settings 分类、记忆预检索开关 | toast、页面业务数据 |
| `chat-runtime-config-store` | model override、temperature/top_p/max_tokens、是否覆盖参数 | 某次请求终态 |
| `chat-store` | 当前 Session ID 与 Agent ID | messages、会话列表/历史、输入草稿、引用记忆、连接、process ID、run status、压缩进度/检查点、本轮失败码 |
| `memory-view-store` | search mode、类型、status filter、排序、grid/list | search query、selected tags、Memory 数据 |
| `memory-task-store` | 是否显示 terminal tasks | task、选中 task、请求状态 |
| `workspace-store` | filters、trace/span UI | logs、trace groups、RuntimeEvents、连接、主窗口状态 |
| `topic-store` | 无 | Topic 池、loading、error |

Settings 草稿使用组件 state/useDraft；OmniInput 草稿使用 chat store 的运行期字段，供无输出停止时回填，两者都不跨刷新恢复。Topic `settle`/`delete` 使用乐观移除并在失败时回滚；这里的 `settle` 是结束 Topic 生命周期并交给记忆生成，不是 Memory Library 的 `archive`。Memory delete 当前不回滚，是需要修正的不一致。

## 3. 当前传输

| 能力 | 传输 | 当前入口 | 终态/恢复语义 |
|:---|:---|:---|:---|
| 统一身份上下文 | HTTP 请求头 | `x-user-id` / `x-workspace-id`（`services/identity.ts`） | 所有身份相关请求统一携带；后端只从请求头读取身份选择 |
| Chat | fetch-based SSE | `POST /api/v1/chat` | `done` / `error`；无断线续传 |
| Stop | HTTP | `POST /api/v1/chat/stop` | best-effort 请求，等待流终态确认 |
| Sessions / Topics / Memories / Agents / Config / Registries / Tasks | HTTP JSON | 相对 `/api/v1/...` | 请求级成功失败；页面按能力 refetch 或乐观更新 |
| RuntimeEvent | EventSource SSE | `/api/v1/runtime-events/stream` | 支持 `last_event_id`/gap 的后端语义，浏览器只保留当前运行期 |
| Logs | WebSocket | `/api/v1/ws/logs` | ping/pong、有限重连，无历史恢复 |

Chat 使用 fetch 而不是原生 EventSource，是因为需要以 POST 发送请求体。RuntimeEvent 是 GET 流，使用 EventSource。日志需要双向 ping/pong 和持续推送，使用 WebSocket。

身份选择传输约定：Session、Memory、Agent、Topic、Chat 与 stop 请求统一携带 `x-user-id`/`x-workspace-id` 基础选择；后端不接受 Topic 等管理路由的 `?user_id=` query；Chat/stop 请求体不携带身份字段。管理请求（Session/Memory/Agent/Topic）默认依赖统一请求头，不需要额外参数。

### 3.1 HTTP 错误解析

`services/apiError.ts` 的 `readApiError()` / `apiErrorMessage()` 供 HTTP services 与 SSE 握手共用。后端 `error/detail/reason/issues` 以安全 detail 展示、code/reason 分支；兼容旧 detail 字符串、message 字符串、detail.message 与 422 校验列表。旧校验列表只读取字段位置和受控类别，不展示 input/ctx/validator 原文。未知对象、HTML 和空体使用带 HTTP status 的默认提示，正文读取的 AbortError 继续抛出。Session 等服务保留原有状态异常，RuntimeEvent disabled 503 仍按业务 DTO 处理。后端信封与迁移说明见[错误模型](../contracts/error-model.md#7-http-api-与客户端呈现)。

## 4. Origin 与代理

本地开发时，Vite 在 `127.0.0.1:5173` 运行，并将 `/api` 代理到 `http://localhost:8769`，包括 WebSocket upgrade。Docker/整合部署由 FastAPI 在 `8000` 同源提供前端和 API。

普通 HTTP services 全部使用相对 `/api`；日志与 RuntimeEvent 流的 URL 则由 `VITE_BACKEND_ORIGIN || window.location.origin` 生成。因此：

- 同源部署与 Vite proxy 是当前完整支持路径；
- 仅设置 `VITE_BACKEND_ORIGIN` 只会迁移日志和 RuntimeEvent，不会迁移 chat、memory、topic 等 HTTP 请求；
- 真正的前后端分域部署需要反向代理，或统一改造所有 service 的 base URL 与 CORS/credential 策略。

后端 CORS 当前允许若干本地开发 origin，没有面向任意生产域名的配置化清单。

## 5. 多窗口日志同步

Workspace store 通过 `hivememory_primary_window` localStorage heartbeat 判断主窗口。主窗口建立 WebSocket 与 RuntimeEvent 连接，再通过 `hivememory_workspace_logs` BroadcastChannel 广播新日志、事件和连接状态；次窗口只消费广播。

这个机制减少重复连接，但没有服务端 lease，也没有跨浏览器实例协调。浏览器冻结、localStorage 清理、时钟漂移或主窗口非正常终止时可能短暂出现无主或多主；它不能替代后端订阅身份和可靠消费语义。

## 6. Mock 降级边界

当前 mock 行为并不统一：

- Memory API 读取失败时始终回退 `MOCK_MEMORIES` 并清除 error；
- Agent 列表失败时回退 `MOCK_AGENT_CONFIGS`；
- Config 失败时回退 `MOCK_CONFIG`，同时保存 error；
- Memory task 只在 Vite development 模式回退 mock；
- Session、Provider、Model、Topic 与 Chat 不使用同类数据回退。

Mock 的合理用途是让组件开发仍有可见状态，不是把离线体验冒充真实系统。当前 Memory/Agent 页面缺少醒目的 mock 标识，且写操作仍指向后端，是最容易制造误解的边界。后续若统一降级策略，至少应区分 `loading / live / mock / stale / error`，并阻止或明确模拟不可兑现的写操作。

## 7. 不变量与失败处理

- localStorage 只保存偏好、视图和可安全重新解释的选择，不保存权威业务实体；
- SSE、WebSocket 与 RuntimeEvent 的连接状态不等于业务终态；
- 乐观更新必须有回滚或显式 reconciliation；
- mock 数据必须可辨认，不能参与真实写请求后再显示成功；
- transport 变更必须同时检查开发 proxy、整合部署、反向代理和多窗口行为；
- store migration 负责清理旧结构，不能让过期 model ID、消息块或 nav tab 在升级后复活。

## 8. 当前限制

- 没有 chat SSE 的 last-event-id 或断线续传；会话历史可重新读取，但后端重启后丢失；
- 没有统一 query cache、请求去重或 stale-while-revalidate 层；
- 多个页面的 mock 降级策略和错误可见性不一致；
- 基础身份选择当前固定为 `user_id=default + main_workspace`（`services/identity.ts`），没有认证 token、租户或 per-user storage namespace；
- `VITE_BACKEND_ORIGIN` 不是全局 API base；
- 会话已有自动化行为测试；尚无完整的持久化迁移矩阵与跨窗口一致性测试。

后端 route 与 event 的唯一契约见[公开路由与事件](../contracts/routes-and-events.md)，Chat 事件如何进入页面见 [Chat 视图](./chat-view.md)。
