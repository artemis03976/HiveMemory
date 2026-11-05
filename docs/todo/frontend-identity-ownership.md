---
title: Frontend Identity Ownership
status: todo
owner: frontend
scope: user-identity-state-and-switching
code_paths:
  - frontend/src/services/identity.ts
  - frontend/src/constants/identity.ts
related_docs:
  - docs/frontend/README.md
  - docs/frontend/state-and-transports.md
  - docs/contracts/subsystem-contracts.md
last_reviewed: 2026-11-04
---

# 建立前端身份状态所有权

## 问题与证据

单一的前端身份入口已经建立：[`services/identity.ts`](../../frontend/src/services/identity.ts) 维护 `user_id + workspace_id` 基础选择，固定为默认用户与默认 Workspace；会话、chat、记忆、Agent、话题与附件的请求都经 `identityHeaders()` 携带 `x-user-id`/`x-workspace-id`，`agent_id` 只作为具体操作的参数（2026-11-04 核对）。旧 TODO 提议增加持久化 `UserStore`，但仅新增一个可切换字符串不足以建立认证、租户隔离或安全边界。

仍然缺少的是切换语义：选择没有持久化，也没有失效规则；`setIdentitySelection` 被调用时，chat、话题、记忆缓存与进行中的流不会清理或隔离。前端目前没有登录与 Workspace 切换 UI，这部分还没有使用方。当前状态与 transport 行为见[前端总览](../frontend/README.md)第 3 节与[状态、持久化与传输](../frontend/state-and-transports.md)。

## 影响

- 未来接入登录或调试切换时，旧用户的消息、话题和缓存可能残留；
- UI 身份选择可能被误认为后端已经执行认证与授权。

## 完成条件

- 建立单一前端身份状态入口（已完成），并定义初始化、持久化和失效规则；
- 所有需要 Identity 的请求从同一入口派生（已完成），显式 override 只用于受控场景；
- 切换或登出时清理/隔离 chat、topic、memory cache 与进行中的 stream；
- 与后端认证/身份契约对齐，文档明确前端状态不是安全边界；
- 增加身份切换、状态清理和请求字段测试，并更新 `docs/frontend/state-and-transports.md`。

若后端认证方案会改变 token、session 或多租户模型，本项应升级为跨系统 Plan，而不是在前端单独实现长期架构。
