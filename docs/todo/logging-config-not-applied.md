---
title: 日志配置段的级别、格式与输出没有生效
status: todo
owner: system
scope: logging-level-format-file-and-console-config-applied-at-startup
code_paths:
  - src/hivememory/config/app.py
  - src/hivememory/server/__main__.py
  - src/hivememory/server/app.py
  - src/hivememory/server/deps.py
  - scripts/hivememory-dev.sh
  - frontend/src/components/settings/GeneralSettings.tsx
related_docs:
  - docs/system/configuration.md
  - docs/frontend/management-views.md
  - docs/archive/todo/v0.7.0-code-convergence.md
last_reviewed: 2026-10-18
---

# 日志配置段的级别、格式与输出没有生效

## 问题与证据

`LoggingConfig`（`config/app.py`）声明了 `level`、`format`、`file_path`、`console_output`，全仓没有读取方。同一配置段只有 `websocket_*` 被读取，用于 `server/deps.py` 的 `init_websocket_log_broadcasting`。

两种启动方式都不使用这四个键：

- `python -m hivememory.server` 经 `server/__main__.py` 的 `_configure_logging`，写死 `INFO` 级别与固定格式（`basicConfig(..., force=True)`），并把 `hivememory` logger 设为 `INFO`；
- `scripts/hivememory-dev.sh` 直接运行 `uvicorn hivememory.server.app:app`，不经过 `__main__`，根 logger 没有 handler。2026-10-18 按 uvicorn 默认日志配置复现：`hivememory.*` 的 INFO 日志不输出，WARNING 由 Python 的兜底 handler 输出到 stderr。开启 WebSocket 日志广播后，被广播的命名空间挂有广播 handler，控制台输出是否随之改变需在处理时核实。

前端设置页的“通用”分类可以编辑这四个键，`POST /config` 会把它们写入 YAML；[System 配置](../system/configuration.md)的配置树也把 `logging` 列为日志输出配置。用户修改后不会产生任何效果。

## 影响范围

- 日志级别、格式、文件输出与控制台开关无法通过配置调整；
- 两种启动方式的日志行为不同，开发脚本下看不到 `hivememory` 的 INFO 日志。

[v0.7.0 代码收敛清单](../archive/todo/v0.7.0-code-convergence.md)第 2.11 条清理不生效的配置键时，按接线处理这一段，没有删除。可能的处理：在入口加载配置后按 `logging` 段配置根 logger（两种启动方式一致），文件输出与控制台输出由 `file_path`、`console_output` 决定；注意不要用 `force=True` 移除测试框架或 WebSocket 广播挂上的 handler。`POST /config` 是否热更新日志配置，与配置热更新的范围一致。

## 完成条件

- 两种启动方式都按 `logging` 段的级别、格式、文件与控制台设置输出日志，WebSocket 广播照常工作；
- 入口层测试覆盖级别、文件输出与关闭控制台输出，并恢复全局 logging 状态；
- [System 配置](../system/configuration.md)中日志配置的描述随实现更新。

## 追踪

- 2026-10-18：处理 v0.7.0 代码收敛清单第 2.11 条时发现：原条目只列出 `console_output`，同段的级别、格式与文件路径同样没有读取方；未绑定版本或 Issue。
