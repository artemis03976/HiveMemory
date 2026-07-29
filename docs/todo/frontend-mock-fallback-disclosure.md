---
title: Frontend Mock Fallback Disclosure
status: todo
owner: frontend
scope: frontend-mock-fallback-state-disclosure
related_docs:
  - docs/frontend/state-and-transports.md
  - docs/frontend/management-views.md
  - docs/help/troubleshooting.md
last_reviewed: 2026-07-29
---

# 统一前端 mock fallback 的状态披露

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
