---
title: Frontend Mock Fallback Disclosure
status: archived
archived_at: 2026-11-07
implemented_by: PR #189
superseded_by: docs/frontend/state-and-transports.md
owner: frontend
scope: frontend-mock-fallback-state-disclosure
code_paths:
  - frontend/src/utils/debugMode.ts
  - frontend/src/hooks/useMemories.ts
  - frontend/src/hooks/useAgentManagement.ts
  - frontend/src/hooks/useSettings.ts
  - frontend/src/services/memoryTaskApi.ts
related_docs:
  - docs/frontend/state-and-transports.md
  - docs/frontend/management-views.md
  - docs/help/troubleshooting.md
last_reviewed: 2026-11-07
---

# 统一前端 mock fallback 的状态披露

> 状态：Archived（2026-11-07，owner 决定 mock 数据只在 debug 模式下启用，原定的统一 source/mode 标识因此不再需要，处置见文末一节）。当前事实见[状态与传输](../../frontend/state-and-transports.md#6-mock-降级边界)第 6 节与[管理页面](../../frontend/management-views.md)。以下为登记时的原文。

## 问题与证据

前端部分服务在 API 不可用或开发环境下会退回 mock/fallback 数据，但当前页面不总能明确区分真实后端结果、缓存和 mock。文档已将其记录为实验工作台限制。

## 影响

- 开发者可能把 fallback 数据误认为真实 Memory、Topic 或 Agent 状态；
- API 故障会被“看起来正常”的空结果掩盖；
- UI 动画和成功提示可能与后端实际状态脱节。

## 完成条件

- 所有 mock/fallback 入口统一产生可识别的 source/mode 状态；
- 页面在开发环境显示 fallback 提示，不把 mock 写操作伪装成持久化成功；
- transport、store 和管理页面对 loading/error/empty/mock 四种状态有明确区分；
- 增加 API 失败、fallback 和恢复后的前端测试，并更新 Frontend/Help 文档。

## 处置（2026-11-07）

owner 决定 mock 数据只在 debug 模式下启用，PR #189：

- debug 模式只在 Vite 构建或开发服务器显式设置 `VITE_DEBUG=true` 时开启（`frontend/src/utils/debugMode.ts`）。Memory 列表、Agent 管理、Settings 与 memory task 只在 debug 模式下读取失败时换用 mock 数据；memory task 此前以 `import.meta.env.DEV` 判断，开发服务器默认就会换用；
- 默认读取失败如实显示：Memory 列表清空并显示错误，Agent 页提示“Agent 列表读取失败，请检查后端服务后刷新”，Settings 保持未加载并显示“配置加载失败”，“恢复默认”失败以错误 toast 提示，memory task 的错误原样抛出；
- 测试：`frontend/tests/debug-mode-mock.test.mjs` 在开发服务器下不设置 `VITE_DEBUG`，断言后端不可用时 memory task 读取如实失败。

完成条件中的统一 source/mode 状态、开发环境的 fallback 提示与 loading/error/empty/mock 的区分不再需要：默认不出现 mock 数据，debug 模式只用于调界面，页面不加 mock 标识。
