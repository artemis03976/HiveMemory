---
title: Frontend Identity Ownership
status: todo
owner: frontend
scope: user-identity-state-and-switching
related_docs:
  - docs/frontend/state-and-transports.md
  - docs/contracts/subsystem-contracts.md
last_reviewed: 2026-07-29
---

# 建立前端身份状态所有权

## 问题与证据

前端目前没有独立的用户身份状态所有者，多条请求仍依赖固定 `user_id=default` 或调用参数。旧 TODO 提议增加持久化 `UserStore`，但仅新增一个可切换字符串不足以建立认证、租户隔离或安全边界。

当前状态与 transport 行为见[状态、持久化与传输](../frontend/state-and-transports.md)。

## 影响

- 不同页面或请求可能各自选择默认身份，难以保持一致；
- 未来接入登录或调试切换时，旧用户的消息、话题和缓存可能残留；
- UI 身份选择可能被误认为后端已经执行认证与授权。

## 完成条件

- 建立单一前端身份状态入口，并定义初始化、持久化和失效规则；
- 所有需要 Identity 的请求从同一入口派生，显式 override 只用于受控场景；
- 切换或登出时清理/隔离 chat、topic、memory cache 与进行中的 stream；
- 与后端认证/身份契约对齐，文档明确前端状态不是安全边界；
- 增加身份切换、状态清理和请求字段测试，并更新 `docs/frontend/state-and-transports.md`。

若后端认证方案会改变 token、session 或多租户模型，本项应升级为跨系统 Plan，而不是在前端单独实现长期架构。
