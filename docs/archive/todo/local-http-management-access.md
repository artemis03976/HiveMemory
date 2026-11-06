---
title: 本机 HTTP 管理接口不证明调用方
status: archived
archived_at: 2026-11-05
implemented_by: 提交 31f39135、9069db27、952d209f（分支 feat/http-management-credential）
superseded_by: docs/workspace/authentication.md
owner: system
scope: local-http-management-routes-caller-authentication
code_paths:
  - src/hivememory/server/app.py
  - src/hivememory/server/deps.py
  - src/hivememory/server/routers/
  - configs/system_principals.yaml
related_docs:
  - docs/ideas/identity-and-access-model.md
  - docs/help/configuration.md
  - docs/workspace/authentication.md
  - docs/archive/todo/config-secrets-written-to-yaml.md
  - docs/archive/plans/http-management-credential.md
last_reviewed: 2026-11-05
---

# 本机 HTTP 管理接口不证明调用方

> 状态：Archived（2026-11-05，已修复防误用部分，处置见文末一节）。当前事实见[认证与访问登记](../../workspace/authentication.md#1-两阶段认证)第 1 节与[安装与启动](../../help/setup.md#43-启动后端)。以下为登记时的原文，链接已按归档位置改正。

## 问题与证据

HTTP 接口按本地单用户部署的信任假设设计：

- **前端使用的 Workspace 路由**（记忆、话题、记忆任务、Agent、会话、附件、chat）：以 server 登记的 principal（`system.server_principal_id`）经统一认证网关接入。请求头中的用户、Workspace 与 Agent 只是声明，不证明调用方是谁。
- **不经认证网关的路由**：`config`、`providers`、`models`、`logs`、`runtime_events`，可以读写主配置、提供商密钥与模型注册表，并读取日志与运行时事件。
- **要求调用方出示凭据的入口**：只有外部 harness 使用的 `/api/v1/import/turns` 与 `/mcp`，凭据按 principal 登记。

本机任何进程都能调用前两类接口。`hivememory-server` 入口还把监听地址固定为 `0.0.0.0`（[`server/__main__.py`](../../../src/hivememory/server/__main__.py)），同一网络中的其他机器也能连接；开发脚本 `scripts/hivememory-dev.sh` 默认监听 `127.0.0.1`。外部 harness 接入之后，这类进程包括 harness 中带 shell 或网络工具的 agent。WSL2 默认的网络模式还会把 Windows 上的 `127.0.0.1` 转发到 WSL 中监听本机地址的服务，Windows 上的进程同样能访问。

证据：2026-11-01 dsh 接入试用中，Windows 桌面端里的 agent 经本机转发直接调用了 `/api/v1/config`，取得了明文密钥。接口自身的明文泄露已在同日改为打码；主配置保存把密钥写进 YAML 是另一个问题，已于同日修复（[归档 Todo](./config-secrets-written-to-yaml.md)）。

## 影响范围

- 外部 harness 的 agent 在 Workspace 访问登记中只有四项记忆操作，但经 HTTP 管理接口可以绕过这份白名单：管理记忆、改写主配置、写入提供商密钥、读取日志；
- 本机其他程序同样可以这样做，本地单用户部署的信任假设不再成立。

这会改变整个 HTTP API 的信任模型，涉及前端、认证网关与部署方式。认证方式的选项与影响已移入[身份与访问体系 Idea](../../ideas/identity-and-access-model.md#http-管理入口的调用方证明)第 4 节（2026-11-05），本 Todo 保留问题、证据与完成条件。

## 完成条件

- 决定 HTTP 管理接口的调用方认证方式，记入对应的 Idea 或计划（已完成：owner 2026-11-05 决定采用管理凭据，现在先防误用，防刻意随 v1.0 桌面客户端完成，见[身份与访问体系 Idea](../../ideas/identity-and-access-model.md#http-管理入口的调用方证明)第 4 节）；
- 外部 harness 的 agent 与本机其他进程不能在没有管理凭据的情况下读写记忆、配置、密钥与模型注册表；
- 更新[配置指南](../../help/configuration.md#51-访问登记)与[认证与访问登记](../../workspace/authentication.md)中关于信任假设的描述。

## 追踪

- 2026-11-05：owner 决定采用管理凭据；防误用部分抽出计划 [HTTP 管理面的管理凭据](../plans/http-management-credential.md)，计划收尾时本 Todo 归档。

## 处置（2026-11-05）

按 owner 决定的管理凭据实施了防误用部分（[归档计划](../plans/http-management-credential.md)）：

- `/api/` 下除导入之外的 HTTP 与 WebSocket 路由都要求管理凭据，缺少或不一致时在路由之前拒绝，包括不经认证网关的 `config`、`providers`、`models`、运行时事件与日志；
- 凭据保存在 `configs/management.token`，后端首次启动时生成；前端经 Vite 代理或后端下发的 `HttpOnly` cookie 携带，前端源码不处理凭据；
- `hivememory-server` 默认只监听 `127.0.0.1`，Docker Compose 只在宿主的本机地址上开放端口；
- [配置指南](../../help/configuration.md#51-访问登记)与[认证与访问登记](../../workspace/authentication.md)的信任假设已改写。

完成条件中的第二项按防误用达成：外部 harness 的 agent 顺手调用管理接口时被拒。以同一系统用户运行、专门读取凭据文件的程序仍能通过，防刻意随 v1.0 的桌面客户端完成，见[身份与访问体系 Idea](../../ideas/identity-and-access-model.md#http-管理入口的调用方证明)第 4 节。
