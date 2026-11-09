---
title: HiveMemory Help
status: current
owner: project
scope: user-and-operator-guides
code_paths:
  - pyproject.toml
  - frontend/package.json
  - frontend/vite.config.ts
  - docker/
  - configs/
  - src/hivememory/server/
  - integrations/hivememory/
  - integrations/hivememory/node/dsh/
related_contracts:
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-11-08
---

# HiveMemory Help

本目录面向需要安装、运行、配置和排查 HiveMemory 的开发者与个人部署者。Help 回答“如何安全完成一项操作”，当前设计为什么这样划分则由 [Project](../PROJECT.md)、[System](../system/README.md)和各子系统文档维护。

## 当前指南

- [安装与启动](./setup.md)：Docker 一体化运行、本地开发、端口、health/readiness、验证，以及外部 harness 的接入入口；
- [接入 Claude Code](./claude-code-plugin.md)：Claude Code 插件的安装、凭据、配置、确认、排查与更新；
- [接入 DeepSeek Harness](./deepseek-harness-plugin.md)：dsh 插件的安装（Windows 桌面端与命令行）、凭据、配置、确认、排查与更新；
- [接入 Codex](./codex-plugin.md)：Codex 插件（与 Claude Code 合用插件目录）在 Windows 上的安装、hook 信任、凭据、配置、确认、排查与更新；
- [接入 pi](./pi-plugin.md)：pi 扩展（同一插件目录中的 pi 包）在 Windows 上的安装、凭据、配置、确认、排查与更新；
- [配置指南](./configuration.md)：配置来源、Provider 凭证、Model Registry、环境变量和生效边界；
- [故障排查](./troubleshooting.md)：Qdrant、模型预热、LLM、前端连接、mock 数据、日志与 Settings 常见问题。

## 选择运行方式

| 目标 | 建议入口 | 浏览器地址 |
|:---|:---|:---|
| 快速体验完整 Web UI | Docker Compose，应用与 Qdrant 一起启动 | `http://localhost:8000` |
| 修改后端或前端 | 本地 Python 服务 + Vite，Qdrant 可由 Docker 单独运行 | `http://127.0.0.1:5173` |
| 只验证 HTTP API | 本地 `hivememory-server` | `http://localhost:8769` |

开发端口与 Docker 端口属于两种启动形态，不是冲突配置。任何排障都应先确认自己运行的是哪一种。

旧入口已经移入 [`archive/legacy-docs/SETUP.md`](../archive/legacy-docs/SETUP.md)，只用于追溯历史启动方式；所有当前操作以本目录为准。
