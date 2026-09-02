---
title: HiveMemory Frontend
status: current
owner: frontend
scope: frontend-current-design-and-index
code_paths:
  - frontend/src/
  - frontend/vite.config.ts
  - src/hivememory/server/app.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-09-01
---

# HiveMemory Frontend

HiveMemory Frontend 不是后端之外的另一套系统真相，而是人能够看见、介入和校正 HiveMemory 的工作界面。它一方面承载日常对话，另一方面把记忆引用、MTP 动作、子 Agent、后台记忆任务和运行日志投影成可观察状态。前端可以保存界面偏好、维护短期交互草稿，却不能用浏览器缓存替代 Topic、Memory、Config 或运行终态的后端所有权。

这个边界决定了前端的核心问题并非“如何展示更多信息”，而是如何同时满足三件彼此牵制的事情：让普通对话保持安静，让后台动作不再成为黑箱，又让开发者在需要时能够追到机器态细节。

## 1. 设计理念

### 1.1 透明、沉浸、可观测

- **透明**：系统选择了哪个话题、引用了哪些记忆、正在执行什么 MTP 动作、是否进入后台 finalize，都应由真实事件驱动并能够被用户感知；
- **沉浸**：观测信息通过卡片、侧栏和状态提示渐进展开，不用协议原文、日志洪流或频繁弹窗切断对话；
- **可观测**：开发者可以在 Kernel Vision 中查看引用记忆、Memory Runtime task、RuntimeEvent 与日志流，但观测旁路不能反过来决定业务成功。

### 1.2 人类态与机器态分离

MTP 和子 Agent 不是普通聊天文本。当前 SSE handler 将 `mtp_start`、`mtp_result` 与子 Agent 事件转换为结构化 `ContentBlock`，主正文继续承载人类可读回答，MTPCard 与 SubAgentCard 承载机器动作。这是旧设计中“翻译官”理念的当前实现：协议事实不被抹去，却也不再作为 XML 或控制文本直接污染阅读流。

这种分离有一条重要约束：前端只能根据后端明确发出的结构化事件展示动作，不能根据“思考中”之类的动画猜测模型内部推理，更不能把尚未收到终态的操作渲染为成功。

### 1.3 控制面应暴露真实能力

Memory Library、Agent Management 与 Settings 为用户提供人工干预入口，但按钮和表单只有在后端存在对应语义时才是控制面。当前仍有若干仅具外观、尚未接线的入口，文档会明确列出，而不会把设计稿中的 Pin、真正语义搜索、账户头像或全部配置热更新写成现有能力。

### 1.4 日月交辉与水晶工作台

视觉系统延续 Patchouli 魔法图书馆的隐喻：日与月构成冷暖环境光，木、火、土、金、水承担成功、错误、历史、警告与运行中的语义色，半透明面板形成可分层的“水晶”质感。这些隐喻的作用是让复杂状态具有稳定辨识度，而不是让装饰压过信息。

当前颜色 token、暗色/浅色变量、`glass-panel` 与 `ghost-border` 在 `frontend/src/index.css` 中维护。组件为项目自有 React 组件；旧文档提到的 Shadcn UI 并不是当前依赖或组件基础。

## 2. 当前产品表面

`App.tsx` 由持久化的 `activeNavTab` 条件渲染四个主要页面，没有使用前端路由器：

| 页面 | 当前责任 | 当前事实来源 |
|:---|:---|:---|
| Chat | 对话、Agent 选择、Topic 概览、MTP/子 Agent 卡片、引用记忆与运行观测 | chat SSE、topics、memory tasks、RuntimeEvent、logs |
| Memory Library | 记忆浏览、客户端筛选、创建、编辑与删除 | memories API；失败时可能回退 mock |
| Agent Management | Agent Profile 列表、创建、编辑与删除 | agents API + generic memories API；失败时可能回退 mock |
| Settings | 通用配置草稿、Provider 凭证和 Model Registry 管理 | config/providers/models API |

全局导航还显示 Terminal 入口，但 `App.tsx` 当前没有对应渲染分支，选择后主区域为空。这是未完成入口，不是第五个已交付页面。

## 3. 应用边界

前端负责：

- 把 HTTP、SSE、WebSocket 和 RuntimeEvent 映射为可理解的页面状态；
- 保存主题、面板、筛选、当前 Agent 和 generation override 等界面偏好；
- 对明确支持的变更提供表单、确认和失败提示；
- 在不篡改业务语义的前提下执行乐观更新或开发期 mock fallback。

前端不负责：

- 保存聊天历史、Topic、Memory、Config 或 task 的权威副本；
- 判断 Gateway、Alice 或 Patchouli 的业务终态；
- 从日志或动画反推模型思维；
- 提供账户、权限隔离、移动端适配或离线工作保证；
- 把 mock 数据伪装成已经由真实后端确认的结果。

当前匿名界面请求使用 `user_id=default`，由后端入口解析为默认 `IdentityScope`；默认 Agent 为拥有完整权限的 `omni_doll`。前端尚无登录、租户或用户/Workspace 切换能力，`user_id` 只是请求上下文，不是认证或授权凭证。

## 4. 当前文档

- [应用壳与视觉系统](./application-shell.md)：导航、布局、主题、视觉隐喻与部署方式；
- [Chat 工作区](./chat-workspace.md)：对话主流程、SSE 事件、Topic、MTP、子 Agent 与 Kernel Vision；
- [管理页面](./management-views.md)：Memory Library、Agent Management、Settings 与人工干预边界；
- [状态、持久化与传输](./state-and-transports.md)：状态所有权、本地持久化矩阵、HTTP/SSE/WebSocket 和 mock 降级。

旧的 `FrontendDesign.md`、`MemoryGardenUI.md` 与 `frontend-state-persistence-research.md` 已移入 Archive，不再作为当前依据；它们保留设计演进和调研过程，当前事实均以本组文档为准。

## 5. 技术与验证入口

当前技术基线为 React 19、TypeScript 5.9、Vite 7、Tailwind CSS 4、Zustand 5、Motion/Framer Motion、Lucide、react-markdown 与 highlight.js。`react-resizable-panels` 虽然仍在依赖表中，但当前源码没有使用它，左右侧栏是固定宽度和折叠式布局。

| 责任 | 代码入口 |
|:---|:---|
| 应用入口与导航 | `frontend/src/App.tsx`、`components/GlobalNavBar.tsx` |
| Chat 与观测工作区 | `components/ChatLayout.tsx`、`components/chat/` |
| 管理页面 | `components/MemoryLibrary.tsx`、`AgentManagement.tsx`、`SettingsPanel.tsx` |
| 状态所有权 | `frontend/src/stores/`、`hooks/` |
| API 与流传输 | `frontend/src/services/`、`transports/` |
| 主题与视觉 token | `frontend/src/index.css` |
| 开发代理与构建 | `frontend/vite.config.ts`、`package.json` |

前端当前没有独立自动化测试套件，主要可执行门槛是 `npm run lint` 与 `npm run build`；后端路由行为由 `tests/unit/server/routers/` 覆盖。安装与运行方式见 [Help](../help/README.md)。

## 6. 当前总体限制

- chat message 与 `currentTopicId` 明确不持久化，刷新后不会恢复会话；后端也没有 session/message history API；
- 点击历史 Topic 只改变本地标题和高亮，不会加载历史消息，也不会把所选 Topic 送入下一次 chat 请求；
- Memory、Agent 与 Settings 在后端失败时采用不同程度的 mock fallback，部分页面可能看似可用但写操作仍会失败；
- Settings 的主配置类型仍采用旧扁平结构，与当前后端嵌套配置响应不一致，多组分类页尚不能视为可靠控制面；
- `#` 引用、消息复制/点赞/重试、Memory Pin 和顶层 Terminal 页面仍是未接线入口；
- UI 为固定桌面工作台布局，尚没有路由级深链接、正式响应式策略、账户系统或无障碍验收证据。

这些限制意味着当前前端适合个人开发、系统观察与能力验证，而不是已经完成的多用户管理后台。未来功能只有在形成 Plan 并落地后，才能改写本目录的当前能力描述。
