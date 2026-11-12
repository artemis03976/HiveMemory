---
title: HiveMemory Frontend
status: current
owner: frontend
scope: frontend-current-design-and-index
code_paths:
  - frontend/src/
  - frontend/package.json
  - frontend/vite.config.ts
  - src/hivememory/server/app.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
related_docs:
  - docs/workspace/README.md
  - docs/todo/frontend-ui-polish.md
last_reviewed: 2026-11-11
---

# HiveMemory Frontend

HiveMemory Frontend 不是后端之外的另一套系统真相，而是人能够看见、介入和校正 HiveMemory 的工作界面。它一方面承载日常对话，另一方面把记忆引用、MTP 动作、子 Agent、后台记忆任务和运行日志投影成可观察状态。前端可以保存界面偏好、维护短期交互草稿，却不能用浏览器缓存替代 Topic、Memory、Config 或运行终态的后端所有权。

这个边界决定了前端的核心问题并非“如何展示更多信息”，而是如何同时满足三件彼此牵制的事情：让普通对话保持安静，让后台动作不再成为黑箱，又让开发者在需要时能够追到机器态细节。

## 1. 设计理念

### 1.1 透明、沉浸、可观测

- **透明**：系统选择了哪个话题、引用了哪些记忆、正在执行什么 MTP 动作、是否进入后台 finalize，都应由真实事件驱动并能够被用户感知；
- **沉浸**：观测信息通过卡片、侧栏和状态提示渐进展开，不用协议原文、日志洪流或频繁弹窗切断对话；
- **可观测**：开发者可以在 Chat 右栏查看引用记忆与记忆任务，在运行页查看 RuntimeEvent 与日志流，但观测旁路不能反过来决定业务成功。

### 1.2 人类态与机器态分离

MTP 和子 Agent 不是普通聊天文本。当前 SSE handler 将 `mtp_start`、`mtp_result` 与子 Agent 事件转换为结构化 `ContentBlock`，主正文继续承载人类可读回答，MTPCard 与 SubAgentCard 承载机器动作。这是旧设计中“翻译官”理念的当前实现：协议事实不被抹去，却也不再作为 XML 或控制文本直接污染阅读流。

这种分离有一条重要约束：前端只能根据后端明确发出的结构化事件展示动作，不能根据“思考中”之类的动画猜测模型内部推理，更不能把尚未收到终态的操作渲染为成功。

### 1.3 控制面应暴露真实能力

Memory Library、Agent Management 与 Settings 为用户提供人工干预入口，但按钮和表单只有在后端存在对应语义时才是控制面。后端没有对应语义的入口不做成看似可用的按钮：记忆 Pin 一律禁用并提示“尚未提供”；头像只是外观菜单的入口，不代表账户；设置页的每一项都绑定真实的配置字段，并写明保存后立即生效还是重启后生效。

### 1.4 魔法图书馆与紫色霓虹

视觉系统延续 Patchouli 魔法图书馆的隐喻：东方风格的命名（帕秋莉、人偶）与“少女祈祷中...”这类文案只用在标题、加载与空状态，按钮、表单与错误提示用直白的说法。颜色以紫色为主、樱花粉点缀；七曜色只表示状态（水、木、火、金四色）或记忆类型（七色，只上在类型图标块上），樱花粉不表示任何状态。这些隐喻的作用是让复杂状态具有稳定辨识度，而不是让装饰压过信息。

颜色、表面与阴影都是 `frontend/src/index.css` 中的语义令牌，深浅两套主题各自取值（[应用壳与视觉系统](./application-shell.md)）。组件为项目自有 React 组件；旧文档提到的 Shadcn UI 并不是当前依赖或组件基础。

## 2. 当前产品表面

`App.tsx` 由持久化的 `activeNavTab` 条件渲染五个主要页面，没有使用前端路由器：

| 页面 | 当前责任 | 当前事实来源 |
|:---|:---|:---|
| Chat | 会话管理与历史恢复、逐轮 Agent 选择、`#` 话题引用、MTP/子 Agent 卡片、逐轮话题标题、消息的复制/反馈/重新生成、引用记忆、记忆任务与本会话的模型配置 | sessions API、chat SSE、topics API、memory tasks、RuntimeEvent |
| Memory Library | 长期记忆的浏览、客户端筛选、语义检索、创建、编辑与删除；短期话题池的查看、结算、丢弃与在对话中引用 | memories API、topics API；读取失败时显示错误，debug 模式下回退 mock |
| Agent Management | Agent Profile 列表、创建、编辑、启用/停用与删除 | agents API + generic memories API；读取失败时提示，debug 模式下回退 mock |
| Settings | 主配置的常用与高级两层、外观（主题）、提供商凭证、模型注册表、接入与权限 | config/providers/models/access API |
| 运行（Terminal 入口） | 运行时事件按一次运行展示的列表与时间线，以及后端日志尾部；只看不改 | RuntimeEvent 流、logs WebSocket |

## 3. 应用边界

前端负责：

- 把 HTTP、SSE、WebSocket 和 RuntimeEvent 映射为可理解的页面状态；
- 保存主题、面板、筛选、当前 Agent 和 generation override 等界面偏好；
- 对明确支持的变更提供表单、确认和失败提示；
- 在不篡改业务语义的前提下执行乐观更新，以及 debug 模式（`VITE_DEBUG=true`）下的 mock fallback。

前端不负责：

- 保存聊天历史、Topic、Memory、Config 或 task 的权威副本；
- 判断 Gateway、Alice 或 Patchouli 的业务终态；
- 从日志或动画反推模型思维；
- 提供账户、权限隔离、移动端适配或离线工作保证；
- 把 mock 数据伪装成已经由真实后端确认的结果。

前端维护并传递用户导向身份选择上下文：基础选择为 `user_id + workspace_id`（默认 `default` + `main_workspace`），经 `services/identity.ts` 统一以请求头 `x-user-id`/`x-workspace-id` 携带；只有 Chat 等由具体 Agent 执行的操作才在请求体附加 `agent_id`（默认 `omni_doll`）。后端 `server/deps.py` 只将该选择解析为身份声明，经统一认证网关认证得到工作 actor，再由操作授权点组装 `IdentityScope`；保留值 `system` 由后端注入非 Agent action，前端不得把它作为可选 Agent。Chat 请求体仍接受 `session_id`，但它不进入 actor 身份，也不参与认证或资源授权。响应中的 `user_id`（如 MemoryResponse）是 owner 展示投影，前端不从响应反推下一次 actor 选择。前端尚无登录、租户或用户/Workspace 切换 UI，`user_id` 只是请求上下文，不是认证或授权凭证。

## 4. 当前文档

- [应用壳与视觉系统](./application-shell.md)：导航、布局与断点、令牌与主题、提示与连接状态条、基本可用性与部署方式；
- [Chat 视图](./chat-view.md)：会话管理与历史恢复、对话主流程、`#` 话题引用、SSE 事件、话题标题、消息操作、MTP、子 Agent 与右栏；
- [管理页面](./management-views.md)：Memory Library、Agent Management、Settings 与人工干预边界；
- [全局运行页](./run-view.md)：Terminal 入口的运行列表、事件时间线、后端日志与有限历史；
- [桌面客户端](./desktop-client.md)：Windows 安装版的托盘常驻、管理凭据、数据根、首次部署、删除记忆数据与打包升级；
- [状态、持久化与传输](./state-and-transports.md)：状态所有权、本地持久化矩阵、HTTP/SSE/WebSocket、连接状况和 mock 降级。

旧的 `FrontendDesign.md`、`MemoryGardenUI.md` 与 `frontend-state-persistence-research.md` 已移入 Archive，不再作为当前依据；它们保留设计演进和调研过程，当前事实均以本组文档为准。

## 5. 技术与验证入口

当前技术基线为 React 19、TypeScript 5.9、Vite 7、Tailwind CSS 4、Zustand 5、Motion、Lucide、react-markdown 与 highlight.js；Inter 与 Manrope 两款字体以 `@fontsource-variable` 包随构建打包。`react-resizable-panels` 虽然仍在依赖表中，但当前源码没有使用它，Chat 的左右两栏是固定宽度、可收起的布局，窗口较窄时改为抽屉。

| 责任 | 代码入口 |
|:---|:---|
| 应用入口与导航 | `frontend/src/App.tsx`、`components/GlobalNavBar.tsx` |
| 全局提示、连接状态条与页面头部 | `components/common/`、`transports/connectionStatus.ts` |
| Chat 视图与观测面板 | `components/ChatLayout.tsx`、`components/chat/` |
| 管理页面 | `components/MemoryLibrary.tsx`、`AgentManagement.tsx`、`SettingsPanel.tsx` |
| 全局运行页 | `components/runs/`、`stores/runs/` |
| 状态所有权 | `frontend/src/stores/`、`hooks/` |
| API 与流传输 | `frontend/src/services/`、`transports/` |
| 主题与视觉令牌 | `frontend/src/index.css`、`stores/ui/theme.ts` |
| 开发代理与构建 | `frontend/vite.config.ts`、`package.json` |

前端的可执行门槛是 `npm run lint`、`npm test` 与 `npm run build`，CI 的前端任务依次运行这三项。`frontend/tests/` 下是 Node 内置测试运行器的测试，`npm test` 运行全部，会话行为测试也可以由 `npm run test:session` 单独运行（[Chat 视图](./chat-view.md)）；后端路由行为由 `tests/unit/server/routers/` 覆盖。安装与运行方式见 [Help](../help/README.md)。

## 6. 当前总体限制

- 会话历史由后端进程内的 ConversationSession 持有，刷新页面可以恢复，后端重启后丢失；SSE 断线不续传；
- Memory、Agent 与 Settings 的 mock fallback 只在 debug 模式下启用，此时页面没有 mock 标识，看似可用但写操作仍会失败；
- Memory Pin 没有后端语义，按钮一律禁用；
- UI 是桌面工作台布局：没有路由级深链接和账户系统；断点只在 Chat 页与短期话题池处理，动效没有统一的时长与缓动，记忆库一次读取并渲染全部记忆，这些细节登记在 [Todo：前端细节优化](../todo/frontend-ui-polish.md)；
- 没有组件测试、视觉回归或无障碍的自动化门槛，界面在深浅主题与不同宽度下的表现靠人工核对。

这些限制意味着当前前端适合个人开发、系统观察与能力验证，而不是已经完成的多用户管理后台。未来功能只有在形成 Plan 并落地后，才能改写本目录的当前能力描述。
