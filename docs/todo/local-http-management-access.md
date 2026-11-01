---
title: 本机 HTTP 管理接口不证明调用方
status: todo
owner: system
scope: local-http-management-routes-caller-authentication
code_paths:
  - src/hivememory/server/app.py
  - src/hivememory/server/deps.py
  - src/hivememory/server/routers/
  - configs/system_principals.yaml
related_docs:
  - docs/help/configuration.md
  - docs/workspace/authentication.md
  - docs/todo/config-secrets-written-to-yaml.md
last_reviewed: 2026-11-01
---

# 本机 HTTP 管理接口不证明调用方

## 问题与证据

HTTP 接口按本地单用户部署的信任假设设计：

- **前端使用的 Workspace 路由**（记忆、话题、记忆任务、Agent、会话、附件、chat）：以 server 登记的 principal（`system.server_principal_id`）经统一认证网关接入。请求头中的用户、Workspace 与 Agent 只是声明，不证明调用方是谁。
- **不经认证网关的路由**：`config`、`providers`、`models`、`logs`、`runtime_events`，可以读写主配置、提供商密钥与模型注册表，并读取日志与运行时事件。
- **要求调用方出示凭据的入口**：只有外部 harness 使用的 `/api/v1/import/turns` 与 `/mcp`，凭据按 principal 登记。

本机任何进程都能调用前两类接口。外部 harness 接入之后，这类进程包括 harness 中带 shell 或网络工具的 agent。WSL2 默认的网络模式还会把 Windows 上的 `127.0.0.1` 转发到 WSL 中监听本机地址的服务，Windows 上的进程同样能访问。

证据：2026-11-01 dsh 接入试用中，Windows 桌面端里的 agent 经本机转发直接调用了 `/api/v1/config`，取得了明文密钥。接口自身的明文泄露已在同日改为打码；主配置保存把密钥写进 YAML 是另一个问题，见 [Todo](./config-secrets-written-to-yaml.md)。

## 影响范围

- 外部 harness 的 agent 在 Workspace 访问登记中只有四项记忆操作，但经 HTTP 管理接口可以绕过这份白名单：管理记忆、改写主配置、写入提供商密钥、读取日志；
- 本机其他程序同样可以这样做，本地单用户部署的信任假设不再成立。

可能的处理（未决定）：

- 为 HTTP 管理接口增加本机管理凭据，前端随页面取得，外部进程没有；
- 管理接口只监听另一个端口或 Unix socket，与 harness 使用的 `/mcp`、`/api/v1/import/turns` 分开；
- 收紧到只读，或移除不经认证网关的写接口。

这会改变整个 HTTP API 的信任模型，涉及前端、认证网关与部署方式，实施前先形成 Idea 或计划。

## 完成条件

- 决定 HTTP 管理接口的调用方认证方式，记入对应的 Idea 或计划；
- 外部 harness 的 agent 与本机其他进程不能在没有管理凭据的情况下读写记忆、配置、密钥与模型注册表；
- 更新[配置指南](../help/configuration.md#51-访问登记)与[认证与访问登记](../workspace/authentication.md)中关于信任假设的描述。
