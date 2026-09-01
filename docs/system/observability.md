---
title: System Observability
status: current
owner: system
scope: runtime-events-operations-and-health
code_paths:
  - src/hivememory/system/runtime/events.py
  - src/hivememory/system/runtime/publisher.py
  - src/hivememory/system/runtime/operations.py
  - src/hivememory/system/contracts/runtime_events.py
  - src/hivememory/system/system.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/architecture/workspace.md
  - docs/system/runtime-and-bus.md
last_reviewed: 2026-09-01
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

`workspace_id` 是可选的观测关联字段，用于把事件按资源归属域展示或筛选。它不代表完整的 `IdentityScope`，也不参与 EventBus 路由、订阅、sequence、授权、幂等判断或任何业务状态迁移；需要作出业务决定时必须回到领域返回值、异常或 Store 状态。

事件类型由 `RuntimeEventType` 维护，当前覆盖 chat、command、passive ingress、Gateway workflow/analysis、Agent run、memory task、maintenance、System lifecycle、subsystem operation 和 stream gap。

## 2. RuntimeEventBus 语义

- `emit()` 先补写 timestamp、sequence 和当前 trace context，再进入有界 ring buffer；
- sink 内部异常只记录 warning，不传播到业务调用方；
- `subscribe()` 支持 replay last 或从 `last_event_id` 之后回放；
- 找不到 `last_event_id` 时发布 `event.stream.gap`，说明回放窗口已不再覆盖请求点；
- 每个订阅者拥有有界队列，队列满时丢弃最旧事件并累计 dropped count；
- 不承诺跨进程连续、持久化、全量不丢或 exactly-once。

事件的顺序只在当前进程 RuntimeEventBus 内有意义。它不能被用来推断“没有看到失败事件就一定成功”，也不能替代 Patchouli 的 memory task 状态、Topic/AssetStore 状态或 System 的 chat run 状态。

## 3. Publisher、Scoped sink 与操作观测

SystemAssembler 创建唯一 root `RuntimeEventPublisher`，并为 Alice 注入 subsystem scope；Publisher 负责合并稳定的 subsystem/source/component 与 run/task context，把 Pydantic 或 Mapping payload 转为安全 dict，并在 sink 或 payload 转换失败时保持 best-effort。Alice 进一步通过 `AgentRunEventEmitter` 把 `agent.run.*` 领域事实投影到 Publisher。Gateway、Patchouli、Chat、Scheduler 与 System lifecycle 当前仍可继续使用 scoped sink，后续按计划渐进迁移。

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

System start/stop 发布 `system.starting/ready/start_failed/shutting_down/stopped/stop_failed`，事件 data 包含步骤列表、已完成步骤、失败步骤和耗时。Scheduler 发布 maintenance task started/completed/failed；Passive Ingress 发布事件接收、duplicate 与 memory context 事实。submission 的 queued/running/retry/succeeded/failed 统一使用 Work Queue Runtime 的 `WORK_*` 事件。

这些事件描述状态转换，但不会驱动转换。例如 `maintenance.task.failed` 不自动重试任务，`work.retry_scheduled` 不直接修改 work record，`chat.run.cancelled` 也不由 UI event 反向取消 run；真正的业务控制分别由 scheduler、Work Queue Runtime 和 run registry 负责。

## 5. 健康与日志的关系

`HiveMemorySystem.health()` 汇总子系统健康和模型 ready；它是当前状态快照，不是历史审计。日志适合记录异常和调试上下文，RuntimeEvent 适合跨组件关联和 UI/运维流式消费，两者都不应成为第三份业务事实。

事件 `data` 必须摘要化：不把完整 Passive 消息、tool args、memory context、traceback、密钥或绝对路径放入公共观测信封。需要调试原始 cause 时应使用受保护的日志或专用诊断入口。

Agent token/MTP/CALL 交互输出不属于 RuntimeEvent。它由 Alice 的 `FrameOutputSink -> AgentRunOutput -> AgentRunStreamAdapter` 链路投递，具有请求级背压和断流取消语义；RuntimeEventBus 则面向全局观测，允许慢订阅者丢弃旧事件。两者只共享 correlation ID，不自动桥接，也不把 token 或完整 tool args 复制进观测信封。

## 6. 设计矛盾检查

评审观测改动时，检查：

1. 事件丢失、订阅者异常或 sink 关闭时，业务返回值是否仍然正确？
2. 新事件是否有稳定 `RuntimeEventType`，还是通过 `data` 临时字符串模拟类型？
3. event data 是否包含无法脱敏的外部内容、权限信息或内部异常细节？
4. 事件是否被误当成命令、重试信号、提交确认或持久化审计？
5. sequence、trace 和 correlation ID 是否能解释一次用例而不建立第二套状态机？

## 7. 当前生产端边界与演进计划

RuntimeEvent 的消费语义已经稳定，生产端迁移则处于渐进阶段。统一 `RuntimeEventPublisher` 基础设施已经落地，Alice 的 `agent.run.*` 已迁移到 `AgentRunEventEmitter`，补齐了 `generation_id` 关联并删除 Agent run 主流程中的 envelope 构造；Chat、Gateway workflow、memory task、Scheduler、System lifecycle 与 Passive Ingress 尚未全部切换到这一模式。

这项重复不会改变当前 wire format 或业务正确性，但会让默认 severity、关联上下文、payload 白名单和异常隔离在多个生产域中漂移。后续迁移应保持三个约束：事件发生时机仍由业务控制流显式决定；领域 emitter 只投影事实、不修改业务状态；底层 publisher 统一 scope、上下文、payload 安全转换和 best-effort 边界。剩余范围与完成条件见 [RuntimeEvent 生产端迁移后续](../todo/runtime-event-producer-migration.md)。

因此不能把“Publisher 与 Alice emitter 已落地”写成“全域生产端重构已完成”，也不能假设所有 payload 已强类型化。尚未迁移的生产点仍以代码和本文件描述的外部契约为准。

## 8. 验证入口

- `tests/unit/system/runtime/test_runtime_events.py`
- `tests/unit/system/runtime/test_publisher.py`
- `tests/unit/alice/runtime/test_runtime_events.py`
- `tests/unit/system/runtime/test_operations.py`
- `tests/unit/system/test_lifecycle.py`
- `src/hivememory/system/contracts/runtime_events.py`
