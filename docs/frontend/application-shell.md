---
title: Frontend Application Shell and Visual System
status: current
owner: frontend
scope: navigation-layout-theme-and-deployment
code_paths:
  - frontend/src/App.tsx
  - frontend/src/components/GlobalNavBar.tsx
  - frontend/src/components/ChatLayout.tsx
  - frontend/src/index.css
  - frontend/vite.config.ts
  - src/hivememory/server/app.py
related_contracts:
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-07-28
---

# 前端应用壳与视觉系统

应用壳把 HiveMemory 的多个观察和管理视角放进同一张工作台。它不试图用一套路由层模拟多个独立产品，而是让用户在 Chat、记忆、Agent 和配置之间快速切换，同时保留主题与工作区偏好。当前设计优先服务桌面开发与个人使用，尚未把 URL 导航、账户边界或移动端布局作为已交付能力。

## 1. 当前结构

```text
App
  ├─ GlobalNavBar (fixed 64px)
  ├─ ChatLayout
  │    ├─ ContextSidebar (288px / collapsed 48px)
  │    ├─ ChatWorkspace (flex)
  │    └─ KernelVision (320 / 440 / 600px, collapsed 48px)
  ├─ MemoryLibrary
  ├─ AgentManagement
  ├─ SettingsPanel
  └─ DynamicToast
```

`activeNavTab` 决定四个主页面中的哪一个被渲染。这里没有 React Router，也没有按 URL 恢复页面、浏览器前进后退或页面级深链接。GlobalNavBar 的 Terminal 按钮会把状态切为 `terminal`，但 App 没有该分支，因此当前会出现空白主区。

Chat 工作区也不是旧设计中的可拖拽四栏：左侧 Context Sidebar 和右侧 Kernel Vision 采用固定宽度，支持折叠；中央 Chat Workspace 使用剩余空间。`react-resizable-panels` 尚未被源码使用，不能把依赖存在等同于拖拽布局已经落地。

## 2. 为什么采用同屏工作台

HiveMemory 的价值不仅在最终回答，还在回答依赖了哪些记忆、触发了哪些动作、后台是否仍在沉淀。本设计因此把三种视角放在同一水平空间：

- 左侧回答“本次对话处于什么工作上下文”；
- 中央回答“人正在与哪个 Agent 交互”；
- 右侧回答“系统正在引用、生成或记录什么”。

侧栏可以折叠，是为了让日常对话保持沉浸；它们始终可返回，又使调试不必跳到另一套管理工具。固定宽度是当前实现的简化取舍，也意味着窄屏时左右控制面会明显挤压正文。

## 3. 视觉隐喻与语义色

### 3.1 日月交辉

背景以月之紫蓝表达记忆的深度与静态积累，以日之暖金表达当前焦点和用户主动输入。它不是项目能力的拟人化证明，而是一种稳定的空间区分：冷色构成工作台基底，暖色用于引导注意。

### 3.2 五行状态

`index.css` 中的当前 token 为：

| Token | 主要语义 |
|:---|:---|
| `magic-water` | 流式、连接、运行中 |
| `magic-wood` | 成功、完成、生效 |
| `magic-fire` | 错误、停止、删除 |
| `magic-earth` | 历史、弱化、非活跃 |
| `magic-metal` | 警告、挂起、需要注意 |
| `magic-moon` / `magic-sun` | 主题高光与关键引导 |

状态色应作为低透明度背景、边框和文字提示，而不是高饱和度实心色块。若同一业务状态在不同页面被映射为不同颜色，视觉隐喻就失去了帮助用户识别状态的意义。

### 3.3 水晶层次

`glass-panel` 和 `ghost-border` 把背景、主面板与局部卡片分层。毛玻璃只用于表达空间层次，不应在每一层重复叠加模糊；信息密度较高的日志、表格和表单仍以可读性优先。

当前暗色与浅色主题通过 `chat-ui-store.theme` 持久化，再由 `App` 同步到 `<html data-theme>`。浅色主题已经具备 CSS 变量和主要组件覆盖，但 Roadmap 仍将完整页面覆盖、自定义背景与等待反馈归入后续 Frontend Experience 工作。

## 4. 组件体系

当前组件为项目内自有 React 实现，样式主要由 Tailwind CSS 4 class、CSS token 和少量通用表单组件组成。Motion/Framer Motion 用于页面、消息和弹层过渡；Lucide 提供图标；Markdown 由 react-markdown、remark/rehype 与 highlight.js 渲染。

旧设计稿曾提出 Shadcn UI，但当前依赖和源码都没有使用 Shadcn/Radix。若未来引入新的组件基础，必须先明确它如何继承现有 token、主题和交互语义，而不是让两套视觉与状态规范并行。

## 5. 开发与部署形态

| 形态 | 前端入口 | 后端入口 | 连接方式 |
|:---|:---|:---|:---|
| 本地开发 | `127.0.0.1:5173` | `localhost:8769` | Vite 将相对 `/api` 代理到后端，并代理 WebSocket |
| Docker/整合部署 | `localhost:8000` | 同源 `localhost:8000` | FastAPI 挂载 `frontend/dist`，API、SSE 与 WebSocket 同源 |

生产构建只有在 `HIVEMEMORY_SERVE_FRONTEND=true` 且构建目录存在时才由 FastAPI 提供。静态服务对非 API 路径执行 SPA fallback，并对请求路径做 realpath/commonpath 检查；API 未匹配仍返回 404，不会被 `index.html` 吞掉。

大多数前端请求直接使用相对 `/api`，因此非同源部署需要外部反向代理。`VITE_BACKEND_ORIGIN` 当前只直接影响 Kernel 日志 WebSocket 与 RuntimeEvent URL，不能单独改变全部 HTTP API 的目标。

## 6. 不变量与矛盾检查

- 页面导航若新增一个 tab，必须同时存在真实渲染分支、可恢复状态和无障碍名称；
- 主题 token 是跨页面语义，不在局部组件随意重新定义同名状态色；
- 机器态信息通过结构化卡片和侧栏渐进披露，不回流为普通正文噪声；
- 前端部署口径必须区分开发端口 `5173/8769` 与 Docker 同源端口 `8000`；
- 新的后端 origin 配置只有覆盖 HTTP、SSE 与 WebSocket 全部传输后，才能被称为完整的分离部署支持；
- 任何“可拖拽、响应式、账户头像、终端页面”描述必须有当前组件和交互证据，不能只依据依赖、图标或设计稿。

## 7. 当前限制

- 无客户端路由、深链接和浏览器历史集成；
- 顶层 Terminal tab 未接线；
- 用户头像为固定外部图片 URL，不代表账户或身份能力；
- Chat 侧栏固定宽度，窄屏与移动端没有正式适配；
- 没有已配置的前端组件测试、视觉回归或可访问性自动化门槛；
- `react-resizable-panels` 留在依赖中但未使用，属于实现与依赖清理问题。

Chat 内部交互见 [Chat 工作区](./chat-workspace.md)，状态与部署连接细节见[状态、持久化与传输](./state-and-transports.md)。
