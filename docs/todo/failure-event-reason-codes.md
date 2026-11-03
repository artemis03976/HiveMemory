---
title: 部分失败事件没有稳定原因码
status: todo
owner: components
scope: failure-runtime-events-outside-memory-generation-lack-stable-reason-codes-or-carry-raw-exception-text
code_paths:
  - src/hivememory/components/scheduler/async_scheduler.py
  - src/hivememory/components/events/operations.py
  - src/hivememory/system/system.py
  - src/hivememory/patchouli/services/lifecycle.py
  - src/hivememory/alice/events.py
  - src/hivememory/alice/run/
  - src/hivememory/workspace/process/events.py
related_docs:
  - docs/governance/observability/logs-and-events.md
  - docs/components/observability.md
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-11-03
---

# 部分失败事件没有稳定原因码

## 问题与证据（2026-11-03）

事件契约要求失败事件的 `reason` 使用稳定原因码，异常类名只放在 `data.error_class`，事件不承载原始异常正文（[可观测性](../components/observability.md#1-runtimeevent-信封)第 1 节）。Import Bus、Gateway workflow 与 Work Queue 已经做到（[事件契约](../contracts/routes-and-events.md#44-失败与降级观测)第 4.4 节），下面几处没有：

| 事件 | 发出位置 | 现状 |
|:---|:---|:---|
| `maintenance.task.failed` | `AsyncMaintenanceScheduler._execute_task` | `reason` 与 `data.error` 为原始异常文本，没有异常类名 |
| 生命周期整理的失败 | `LifecycleFamiliar.run_gardening_once` | 捕获异常后返回带 `error` 的结果，失败以 info 级 `maintenance.task.completed` 出现 |
| `system.start_failed` / `system.stop_failed` | `HiveMemorySystem` 的启动与停止 | `reason` 与 `data.error` 为原始异常文本 |
| `subsystem.operation.failed` | `RuntimeOperationObserver`（当前唯一使用方是 Patchouli 的 shutdown drain） | `reason` 与 `data.error` 为原始异常文本 |
| `agent.run.failed` | Alice 的运行结果投影与运行服务的异常路径 | `reason` 为空，CPU 失败原因没有传入 |
| `chat.run.failed` | 任务进程的事件投影 | `reason` 为空，失败原因码放在 `message` |

生成任务的原因码属于 v0.7.3（[收敛 Idea](../ideas/generation-source-convergence.md) 第 4.3 节），不在本 Todo。

## 影响

失败事件之间无法按原因归类；原始异常文本进入公共观测信封，违反事件契约。

## 做法

**最简版本（默认）**（治理主题 W3）

| 事件 | `reason` | `data` |
|:---|:---|:---|
| `maintenance.task.failed` | `maintenance_task_failed` | `error_class`；删除 `error` |
| 生命周期整理的失败 | 不再捕获异常，交给调度器，按上一行发出 | 同上 |
| `system.start_failed` / `system.stop_failed` | `start_step_failed` / `stop_step_failed` | `error_class` 与原有的失败步骤；删除 `error` |
| `subsystem.operation.failed` | `operation_failed` | `error_class`；删除 `error` |
| `agent.run.failed` | CPU 失败原因码；没有原因码或异常路径为 `agent_run_failed` | 异常路径带 `error_class` |
| `chat.run.failed` | CPU 失败原因码、Workspace 错误码或 `task_process_failed` | 未预期的异常带 `error_class` |

- `message` 保持各处现有的固定说明；前端实时面板只显示 `message` 或 `reason`，不按它们分支。
- 原始异常不再进事件：调度器在发出事件时记录带堆栈的日志；系统启停与操作观测在发出事件后重新抛出异常。
- 生命周期整理的结果不再含 `error`；它的本地路由 `lifecycle.run_gardening_once` 当前没有调用方。

## 完成条件

- 上表事件带原因码与异常类名，不含构造异常时使用的文本；生命周期整理失败时调度器发出失败事件；
- 测试放在各自的现有测试文件：`tests/unit/components/scheduler/test_async_scheduler.py`、`tests/unit/patchouli/services/test_lifecycle.py`、`tests/unit/system/test_lifecycle.py`、`tests/unit/components/events/test_operations.py`、`tests/integration/alice/test_events.py`、`tests/unit/workspace/process/test_process_events.py`；
- 更新[事件契约](../contracts/routes-and-events.md#44-失败与降级观测)第 4.4 节、[可观测性](../components/observability.md)第 4 节与治理主题第 3、4 节。
