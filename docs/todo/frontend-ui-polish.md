---
title: Frontend UI Polish
status: todo
owner: frontend
scope: motion-responsiveness-and-adaptive-layout-after-the-ui-optimization-plan
code_paths:
  - frontend/src/App.tsx
  - frontend/src/index.css
  - frontend/src/hooks/useMemories.ts
  - frontend/src/components/ChatLayout.tsx
  - frontend/src/components/MemoryLibrary.tsx
related_docs:
  - ../frontend/README.md
  - ../archive/plans/frontend-ui-and-entry-wiring.md
last_reviewed: 2026-11-11
---

# 前端细节优化：动效、响应速度与自适应布局

## 现状与证据

前端 UI 优化计划完成了设计令牌与深浅主题、各页的排版与文案、设置页分层、基本的键盘与读屏可用性，以及空入口的接线。owner 2026-11-11 决定计划就此收尾，剩下的细节单独记为本 Todo：

- **动效**：没有统一的时长与缓动。进入与切换动画各页自行处理，记忆卡片、对话框与子 Agent 卡片用 `motion` 做入场，其余页面没有；“减少动效”时 CSS 动画与 Motion 动画都会去掉（`index.css`、`App.tsx` 的 `MotionConfig`），但没有一套统一的动效规范。
- **响应速度**：记忆图书馆一次取全整个记忆库并渲染全部卡片（`useMemories` 的 `fetchAllMemories`），记忆增多后首屏与筛选会变慢；切换导航时页面整体卸载再挂载（`App.tsx`），回到对话页会重新读取会话列表，设置页每次进入都重新读取配置。
- **自适应布局**：断点只按 1440 与 1024 两种宽度核对过。对话页在 1280 与 1024 两处收起侧栏，话题池在 1280 以下收窄列表；1024 以下（只出现在浏览器中）、超宽屏下的内容宽度，以及 Agent 页固定 300 宽的列表、设置页固定 240 宽的导航，都没有系统处理。

## 完成条件

- [ ] 动效：定出一组时长与缓动并在各页统一；列出哪些地方有进入与切换动画，减少动效时一并去掉。
- [ ] 响应速度：记忆库在记忆数增长后仍然流畅（分页或虚拟列表，二选一需 owner 决定）；页面切换时保留已经读取的数据，不重复读取。
- [ ] 自适应布局：确定支持的最小宽度与超宽屏的内容宽度，各页按同一组断点处理侧栏与列表。
- [ ] 以上在打包后的桌面端中由 owner 验收。
