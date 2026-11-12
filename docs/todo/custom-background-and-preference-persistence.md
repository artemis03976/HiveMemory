---
title: Custom Background And Preference Persistence
status: todo
owner: frontend
scope: custom-background-frosted-surfaces-and-desktop-ui-preference-persistence
code_paths:
  - desktop/src/main.js
  - frontend/src/index.css
  - frontend/src/stores/ui/appUiStore.ts
  - frontend/src/stores/ui/theme.ts
related_docs:
  - ../frontend/desktop-client.md
  - ../frontend/application-shell.md
  - ../frontend/state-and-transports.md
  - ../archive/plans/frontend-ui-and-entry-wiring.md
last_reviewed: 2026-11-11
---

# 自定义背景，以及桌面端界面偏好的持久化

## 现状与证据

- **自定义背景**：没有实现（候选清单 D-13 的剩余部分）。面板与卡片使用不透明的表面令牌；毛玻璃效果是为了让自定义背景透出，没有背景时不做，前端 UI 优化计划因此把它推迟到这里。
- **桌面端的界面偏好**：桌面客户端的窗口使用没有 `persist:` 前缀的会话分区（`desktop/src/main.js`），管理凭据的 cookie 与缓存因此不落盘。副作用是页面保存在浏览器存储中的界面偏好只在一次运行内有效：主题、Chat 两栏的收起、视图与筛选、当前会话与 Agent 的选择，在每次重启桌面客户端后回到缺省值（[桌面客户端](../frontend/desktop-client.md#10-已知限制)）。从源码目录以浏览器使用时不受影响。

## 影响

在桌面客户端中选择的深色或浅色主题，下次启动时回到跟随系统；之后的自定义背景图片同样需要跨重启保存，现有的存储方式保存不了。

## 完成条件

- owner 2026-11-11 决定：主题需要持久化，与自定义背景一起做。
- [ ] 选定界面偏好的保存位置，且不能让管理凭据落盘。可选的做法：偏好经主进程保存到数据根；或改用持久的会话分区，同时让管理凭据的 cookie 不随分区落盘。需要 owner 决定。
- [ ] 主题与其他界面偏好在桌面客户端重启后保留；从源码目录以浏览器使用时行为不变。
- [ ] 自定义背景：图片的选取与保存、深浅主题下文字的可读性，以及让背景透出的毛玻璃表面令牌。
- [ ] 在打包后的桌面端中由 owner 验收。
