---
title: System Observability
status: current
owner: system
scope: runtime-events-operations-and-health
code_paths:
  - src/hivememory/system/runtime/events.py
  - src/hivememory/system/runtime/operations.py
  - src/hivememory/system/contracts/runtime_events.py
  - src/hivememory/system/system.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
last_reviewed: 2026-07-29
---

# System 可观测性

HiveMemory 的观测设计解决的是“如何知道一次运行发生了什么”，不是“如何让观测系统替业务作决定”。因此 RuntimeEventBus 是独立的 best-effort 旁路，业务返回值、异常和权威状态才是业务正确性的来源。

这一取舍来自两个现实：观测消费者可能断开、变慢或丢失，而 chat、命令和记忆提交不能因为 UI 没有订阅就停住；同时，单次运行又需要足够的关联 ID、阶段和摘要，才能解释取消、降级和失败究竟发生在哪个边界。

更早的直接痛点是并发前台交互与后台维护日志会在单一时间线上相互穿插：按到达顺序平铺，既看不清一次任务的完整上下文，也容易让静默任务失败被大量组件日志淹没。这是当前事件保留 `trace_id`、`span_name` 与 `task_type`，并由前端按运行上下文分组/折叠的设计来由。它们不是为了伪造模型内部思维链，而是把系统已经观测到的阶段事实重新组织成可读运行轨迹。

业务总线与观测流分离也不只是代码目录选择。业务 RPC 必须给调用点确定结果，观测流则允许 best-effort、回放缺口和慢订阅者隔离；如果让观测发布参与提交、重试或成功判断，UI 断连就可能改变业务正确性。

## 1. RuntimeEvent 信封

`RuntimeEvent` 当前围绕以下信息组织：

- 标识与顺序：`event_id`、进程内 `sequence`、UTC `timestamp`；
- 追踪：`trace_id`、`span_name`、`task_type`；
- 来源：`source`、`subsystem`、`component`、`severity`；
- 关联：generation、agent run、task、agent、frame、topic、atom；
- 结果：`status`、`reason`、`message`、摘要化 `data`。

事件类型由 `RuntimeEventType` 维护，当前覆盖 chat、command、passive ingress、Gateway workflow/analysis、Agent run、memory task、maintenance、System lifecycle、subsystem operation 和 stream gap。

## 2. RuntimeEventBus 语义

- `emit()` 先补写 timestamp、sequence 和当前 trace context，再进入有界 ring buffer；
- sink 内部异常只记录 warning，不传播到业务调用方；
- `subscribe()` 支持 replay last 或从 `last_event_id` 之后回放；
- 找不到 `last_event_id` 时发布 `event.stream.gap`，说明回放窗口已不再覆盖请求点；
- 每个订阅者拥有有界队列，队列满时丢弃最旧事件并累计 dropped count；
- 不承诺跨进程连续、持久化、全量不丢或 exactly-once。

事件的顺序只在当前进程 RuntimeEventBus 内有意义。它不能被用来推断“没有看到失败事件就一定成功”，也不能替代 Patchouli 的 memory task 状态或 System 的 chat run 状态。

## 3. Scoped sink 与操作观测

SystemAssembler 为 System、Gateway、Patchouli、Alice 和具体 component 创建 scoped sink。scope 负责补充稳定的 subsystem/source/component，而不改变业务事件类型。

`RuntimeOperationObserver` 对单个 async operation 提供统一的 started/completed/failed 记录：

```text
observe(operation)
  -> emit started
  -> await operation
  -> success: emit completed + duration + summary
  -> exception: emit failed + reason + duration, re-raise
```

observer 不捕获后吞掉原异常，不执行重试，也不决定事务是否提交。业务代码必须先处理正确性，再把结果投影为安全摘要。

## 4. 生命周期、Scheduler 与 Passive 事件

System start/stop 发布 `system.starting/ready/start_failed/shutting_down/stopped/stop_failed`，事件 data 包含步骤列表、已完成步骤、失败步骤和耗时。Scheduler 发布 maintenance task started/completed/failed；Passive Ingress 发布事件接收、duplicate、memory context、sealed turn 提交和 retry 事实。

这些事件描述状态转换，但不会驱动转换。例如 `maintenance.task.failed` 不自动重试任务，`passive.turn.submit_failed` 不直接修改 outbox，`chat.run.cancelled` 也不由 UI event 反向取消 run；真正的业务控制分别由 scheduler、submitter 和 run registry 负责。

## 5. 健康与日志的关系

`HiveMemorySystem.health()` 汇总子系统健康和模型 ready；它是当前状态快照，不是历史审计。日志适合记录异常和调试上下文，RuntimeEvent 适合跨组件关联和 UI/运维流式消费，两者都不应成为第三份业务事实。

事件 `data` 必须摘要化：不把完整 Passive 消息、tool args、memory context、traceback、密钥或绝对路径放入公共观测信封。需要调试原始 cause 时应使用受保护的日志或专用诊断入口。

## 6. 设计矛盾检查

评审观测改动时，检查：

1. 事件丢失、订阅者异常或 sink 关闭时，业务返回值是否仍然正确？
2. 新事件是否有稳定 `RuntimeEventType`，还是通过 `data` 临时字符串模拟类型？
3. event data 是否包含无法脱敏的外部内容、权限信息或内部异常细节？
4. 事件是否被误当成命令、重试信号、提交确认或持久化审计？
5. sequence、trace 和 correlation ID 是否能解释一次用例而不建立第二套状态机？

## 7. 当前生产端边界与演进计划

RuntimeEvent 的消费语义已经稳定，但生产端尚未完成同等程度的收敛。Chat、Gateway workflow、Alice、memory task、Scheduler 和 System lifecycle 仍分别构造 `RuntimeEvent`，并保留多组 `_emit_*` 私有方法；`PassiveIngressEventEmitter` 与 `RuntimeOperationObserver` 已经证明“领域投影与业务主流程分离”可行，却还没有形成全项目统一的 Publisher/Emitter 模式。

这项重复不会改变当前 wire format 或业务正确性，但会让默认 severity、关联上下文、payload 白名单和异常隔离在多个生产域中漂移。后续重构应保持三个约束：事件发生时机仍由业务控制流显式决定；领域 emitter 只投影事实、不修改业务状态；底层 publisher 统一 scope、上下文、payload 安全转换和 best-effort 边界。详细范围与验收条件见 [RuntimeEvent 生产端发布抽象重构](../plans/runtime-event-publishing-refactor.md)。

在该计划落地前，不能把 `RuntimeEventPublisher`、完整 payload 类型化或全域 emitter 写成当前能力；直接生产点仍以代码和本文件描述的外部契约为准。

## 8. 验证入口

- `tests/unit/system/runtime/test_runtime_events.py`
- `tests/unit/system/runtime/test_operations.py`
- `tests/unit/system/test_lifecycle.py`
- `src/hivememory/system/contracts/runtime_events.py`
