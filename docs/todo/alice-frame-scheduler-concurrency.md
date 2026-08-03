---
title: Alice Frame Scheduler Task Locality
status: completed
owner: alice
scope: completed-frame-scheduler-and-cancel-state-isolation
related_docs:
  - docs/alice/orchestration.md
  - docs/alice/agent-runtime.md
  - docs/plans/identity-isolation-and-execution-safety.md
last_reviewed: 2026-08-03
---

## 已完成实现事实（Phase 4/6）

旧共享 `FrameScheduler._frame_stack` 已从活动执行路径中完全删除。当前每次 Alice run 都新建一个 `RunSession`，其中持有 frame registry、调度状态、CALL records、取消事件和 stream sequence；同时新建一个只服务该 session 的 `RunScheduler`，用唯一 `_drive()` 循环推进 root 与当前 callee。这个 RunScheduler 是 run-local active-frame 状态机，不是旧共享栈 FrameScheduler 的恢复。frame 构造由无状态 `FrameFactory` 完成，MTP 动词权限和迭代上限由 `FrameExecutionPolicy` 承载，CALL 权限不再依赖共享 frame depth。

# Alice FrameScheduler 与取消状态的运行隔离

## 历史问题与证据

此前 `FrameScheduler._frame_stack` 和部分 cancel 状态在 AliceRuntime 级别共享。并发 run 的 CALL 可能交错 push/pop，恢复时又没有完全核对返回 frame；这会让一个 run 的执行坐标、取消信号或子帧关系影响另一个 run。

## 历史影响

- 并发 Agent run 可能恢复到错误的 frame；
- cancel、iteration budget 和 CALL depth 的行为不再只由当前 run 决定；
- 问题难以通过单次顺序运行复现，却会破坏长期服务的隔离性。

## 已完成条件

- frame stack 与 resume frame API 已删除；cancel token、运行预算、frame registry 和 CALL record 均有明确的 run-local owner；
- `PENDING/RUNNABLE/RUNNING/WAITING/TERMINATED` 状态只保存在 RunSession，任一 session 最多一个 RUNNING frame；
- Alice 编排层只有 RunScheduler 调用 `AgentRuntime.run_frame()`，CallCoordinator 不再内嵌执行 callee；
- CALL 回填通过 `(caller_frame_id, action_id)` 的 `CallRecord` 与 `AgentRuntime.apply_call_response()` 校验，不再依赖 parent frame id；
- 两个及以上并发 run、交错 CALL、分别取消和异常恢复已有回归测试；
- 当前仍保持串行单层 CALL，不引入 fan-out、DAG 或递归 CALL；
- 并发隔离规则已同步到 Alice 当前文档；身份安全 Plan 仍保留跨用户授权与缓存隔离的后续范围。
