---
title: Alice Frame Scheduler Task Locality
status: todo
owner: alice
scope: frame-scheduler-and-cancel-state-isolation
related_docs:
  - docs/alice/orchestration.md
  - docs/alice/agent-runtime.md
  - docs/plans/identity-isolation-and-execution-safety.md
last_reviewed: 2026-07-29
---

# Alice FrameScheduler 与取消状态的运行隔离

## 问题与证据

当前 `FrameScheduler._frame_stack` 和部分 cancel 状态在 AliceRuntime 级别共享。并发 run 的 CALL 可能交错 push/pop，恢复时又没有完全核对返回 frame；这会让一个 run 的执行坐标、取消信号或子帧关系影响另一个 run。

## 影响

- 并发 Agent run 可能恢复到错误的 frame；
- cancel、iteration budget 和 CALL depth 的行为不再只由当前 run 决定；
- 问题难以通过单次顺序运行复现，却会破坏长期服务的隔离性。

## 完成条件

- frame stack、resume frame、cancel token 和运行预算改为 task-local 或明确的 run-local owner；
- 恢复前校验 run id、frame id、parent frame id 和 action id；
- 增加两个及以上并发 run、交错 CALL、分别取消和异常恢复的测试；
- 不改变单层星型 CALL 的当前拓扑约束；
- 将并发隔离规则同步到 Alice 当前文档和身份安全 Plan。
