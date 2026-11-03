---
title: 维护任务空转也发事件
status: todo
owner: components
scope: maintenance-scheduler-emits-started-and-completed-on-every-no-op-tick
code_paths:
  - src/hivememory/components/scheduler/async_scheduler.py
  - src/hivememory/core/contracts/runtime_events.py
  - src/hivememory/patchouli/runtime/core.py
  - src/hivememory/infrastructure/model_service.py
related_docs:
  - docs/governance/observability/logs-and-events.md
  - docs/components/runtime-and-bus.md
  - docs/components/observability.md
last_reviewed: 2026-11-03
---

# 维护任务空转也发事件

## 问题与证据（2026-11-03）

`AsyncMaintenanceScheduler._execute_task` 每次运行都发 `maintenance.task.started` 与 `maintenance.task.completed`，不论这一周期有没有做事。只有 Patchouli 注册维护任务：

- 话题空闲扫描每 30 秒，返回本次结算的话题 ID 列表，没有时为空；
- 模型卸载检查每 60 秒，返回 None，看不出是否卸载了模型；
- 生命周期整理每 24 小时。

合计每分钟约 6 条事件。2026-11-03 读取事件缓冲时，最近 100 条全是前两个任务的心跳，1000 条的缓冲约 2.8 小时轮换一遍；前端实时面板同样被它们占满（[日志与运行时事件](../governance/observability/logs-and-events.md)第 1、3 节）。

## 影响

真正的失败与状态转换被心跳冲出缓冲，前端也难以从中找到它们。

## 做法

**最简版本（默认）**（治理主题 W2）

- 调度器不再发 started；回调返回空结果（None 或空容器）表示这一周期没有做事，不发 completed；其余结果照常发 completed；异常照常发 failed。约定写在调度器与 `MaintenanceTaskSpec` 的说明中。
- `MAINTENANCE_TASK_STARTED` 没有生产者后从 `RuntimeEventType` 删除，契约文档同步。前端按类型通用显示维护事件，没有依赖 started 的逻辑。
- `unload_if_idle` 返回是否卸载，`unload_idle_models` 返回本次卸载的模型名列表。

**暂不处理**：调度节奏、重入与跳过策略；失败事件的原因码（见[失败事件原因码 Todo](./failure-event-reason-codes.md)）。

## 完成条件

- 空结果不发事件，非空结果只发 completed，异常只发 failed，被跳过的周期不发事件；改写 `tests/unit/components/scheduler/test_async_scheduler.py` 中现有的成功与失败事件用例，不保留旧契约的测试；
- `unload_if_idle` 的返回值由 `tests/unit/infrastructure/test_model_service.py` 覆盖；
- 更新[运行时机制](../components/runtime-and-bus.md)第 2 节、[可观测性](../components/observability.md)第 1、4 节与治理主题第 3、4 节。
