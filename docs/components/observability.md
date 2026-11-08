---
title: Runtime Events and Observability
status: current
owner: components
scope: runtime-events-operations-and-health
code_paths:
  - src/hivememory/components/events/bus.py
  - src/hivememory/components/events/publisher.py
  - src/hivememory/components/events/operations.py
  - src/hivememory/server/logging_setup.py
  - scripts/runtime_event_timeline.py
  - src/hivememory/core/contracts/runtime_events.py
  - src/hivememory/system/system.py
  - src/hivememory/system/services/import_bus/events.py
  - src/hivememory/workspace/capability/operation_events.py
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/mcp_server/server.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/README.md
  - docs/components/runtime-and-bus.md
last_reviewed: 2026-11-07
---

# 运行时事件与可观测性

本文描述 `components.events` 中的运行时事件机制（RuntimeEventBus、sink、Publisher 与操作观测器）及其使用约束。事件信封模型 `RuntimeEvent` / `RuntimeEventType` 是依赖中立的契约，位于 `core.contracts.runtime_events`；`HiveMemorySystem.health()` 的健康汇总属于 System，只在第 5 节说明它与事件的关系。

HiveMemory 的观测设计解决的是“如何知道一次运行发生了什么”，不是“如何让观测系统替业务作决定”。因此 RuntimeEventBus 是独立的 best-effort 旁路，业务返回值、异常和权威状态才是业务正确性的来源。

这一取舍来自两个现实：观测消费者可能断开、变慢或丢失，而 chat、命令和记忆提交不能因为 UI 没有订阅就停住；同时，单次运行又需要足够的关联 ID、阶段和摘要，才能解释取消、降级和失败究竟发生在哪个边界。

更早的直接痛点是并发前台交互与后台维护日志会在单一时间线上相互穿插：按到达顺序平铺，既看不清一次任务的完整上下文，也容易让静默任务失败被大量组件日志淹没。这是当前事件保留 `trace_id`、`span_name` 与 `task_type`，并由前端按运行上下文分组/折叠的设计来由。它们不是为了伪造模型内部思维链，而是把系统已经观测到的阶段事实重新组织成可读运行轨迹。

业务总线与观测流分离也不只是代码目录选择。业务 RPC 必须给调用点确定结果，观测流则允许 best-effort、回放缺口和慢订阅者隔离；如果让观测发布参与提交、重试或成功判断，UI 断连就可能改变业务正确性。

## 1. RuntimeEvent 信封

`RuntimeEvent` 当前围绕以下信息组织：

- 标识与顺序：`event_id`、进程内 `sequence`、UTC `timestamp`；
- 追踪：`trace_id`、`span_name`、`task_type`；
- 来源：`source`、`subsystem`、`component`、`severity`；
- 关联：process、agent run、task、agent、frame、topic、atom；
- 结果：`status`、`reason`、`message`、摘要化 `data`。

失败 reason 使用稳定原因码，message 使用可公开的说明；异常类名只放在诊断字段 `data.error_class`，不用于恢复决策。各失败事件的原因码由[事件契约](../contracts/routes-and-events.md#44-失败与降级观测)维护，事件不承载原始异常正文。

`workspace_id` 是可选的观测关联字段，用于把事件按资源归属域展示或筛选。它不代表完整的 `IdentityScope`，也不参与 EventBus 路由、订阅、sequence、授权、幂等判断或任何业务状态迁移；需要作出业务决定时必须回到领域返回值、异常或 Store 状态。

事件类型由 `RuntimeEventType` 维护，当前覆盖 chat、Import Bus（`import.turn.*`）、Gateway workflow/analysis、Agent run、actor 操作（`actor.operation`）、话题（`topic.*`）、memory task、work、maintenance、System lifecycle、subsystem operation 和 stream gap；每个类型的生产方、分组键与 `data` 见[事件契约](../contracts/routes-and-events.md#4-runtimeevent-观测契约)第 4 节的事件目录。

### 1.1 一次运行

运行时事件按"一次运行"归组展示，一条事件按下列规则归组，先匹配的规则生效：

1. `memory.task.*` 与 `work.*` 按 `task_id` 归组（Work Queue 的 `task_id` 是 work id）：它们是队列中的任务，有自己的起止：`work.*` 的执行事件在 worker 中发出、trace 为 `system`，`memory.task.*` 沿用提交者的 trace，按 trace 归组会把一个任务拆开或并进提交它的运行。
2. 其余 `trace_id` 不是 `system` 的事件按 `trace_id` 归组。设置 trace 的运行有三种：chat 任务进程（`task-` 前缀）、一轮导入（`import-`）与一次 MCP 工具调用（`mcp-`）。运行内的 Gateway、Alice、actor 操作与新建话题的事件都从 contextvars 继承所在运行的 trace，不必各自携带运行标识。
3. 剩下的事件各自是单条运行：`maintenance.task.*`、`system.*`、`subsystem.operation.*`，以及空闲结算等不在任何运行内发生的 `topic.*`。

不同运行之间经 `topic_id` 互相链接：导入或对话写入话题，话题结算引起生成任务。生成任务与维护任务不设置 trace。同一个 trace 也出现在日志格式中（[System 配置](../system/configuration.md)），一次运行的日志与事件可以互相对应。

## 2. RuntimeEventBus 语义

- `emit()` 先补写 timestamp、sequence 和当前 trace context，再进入有界 ring buffer；
- sink 内部异常只记录 warning，不传播到业务调用方；
- `subscribe()` 支持 replay last 或从 `last_event_id` 之后回放；
- 找不到 `last_event_id` 时发布 `event.stream.gap`，说明回放窗口已不再覆盖请求点；
- 每个订阅者拥有有界队列，队列满时丢弃最旧事件并累计 dropped count；
- 配置了事件记录文件时，补全后的事件在进入缓冲与分发之后写成一行 JSON（第 5.1 节）；回放事件中的 `event.stream.gap` 只发给请求回放的订阅者，不进入缓冲，也不记录；
- 不承诺跨进程连续、全量不丢或 exactly-once；事件记录文件是旁路的记录，不回放到总线。

事件的顺序只在当前进程 RuntimeEventBus 内有意义。它不能被用来推断“没有看到失败事件就一定成功”，也不能替代 Patchouli 的 memory task 状态、Topic/AssetStore 状态或 System 的 chat run 状态。

## 3. Publisher、Scoped sink 与操作观测

SystemAssembler 创建唯一 root `RuntimeEventPublisher`，为每个生产方注入带作用域的 publisher；sink 只留在总线与 publisher 内部。Publisher 负责合并稳定的 subsystem/source/component 与 run/task context，把 Pydantic 或 Mapping payload 转为安全 dict，并在 sink 或 payload 转换失败时保持 best-effort。各生产方在注入的 publisher 上投影事件：

| 生产方 | 事件 |
|:---|:---|
| Alice（`AgentRunEventEmitter`） | `agent.run.*`、`agent.context.*` |
| chat 任务进程（`TaskProcessEventEmitter`，进程创建时绑定 process、trace、workspace 与 agent，确定话题后再绑定 topic） | `chat.run.*` |
| 操作入口（`ActorOperationEventEmitter`） | `actor.operation` |
| Patchouli 感知层 | `topic.*` |
| 生成任务 controller（`MemoryTaskEventEmitter`） | `memory.task.*` |
| Work Queue（`WorkQueueEventEmitter`） | `work.*` |
| Import Bus（`ImportBusEventEmitter`） | `import.turn.*` |
| Gateway workflow 与查询分析 | `gateway.*` |
| 维护调度器 | `maintenance.task.*` |
| System 生命周期 | `system.*` |
| 操作观测（`RuntimeOperationObserver`） | `subsystem.operation.*` |

只有 `components.events` 直接构造 `RuntimeEvent`，由架构测试守护：各域直接组装信封会让默认 severity、关联字段与 best-effort 边界漂移。

`RuntimeOperationObserver` 对单个 async operation 提供统一的 started/completed/failed 记录：

```text
observe(operation)
  -> emit started
  -> await operation
  -> success: emit completed + duration + summary
  -> exception: emit failed + operation_failed + error_class + duration, re-raise
```

observer 不捕获后吞掉原异常，不执行重试，也不决定事务是否提交。业务代码必须先处理正确性，再把结果投影为安全摘要。

## 4. 生命周期、Scheduler 与 Import Bus 事件

System start/stop 发布 `system.starting/ready/start_failed/shutting_down/stopped/stop_failed`，事件 data 包含步骤列表、已完成步骤、失败步骤和耗时。Scheduler 只为做了事的周期发布 `maintenance.task.completed`，回调抛出异常时发布 `maintenance.task.failed` 并记录带堆栈的日志；空转与被跳过的周期不发事件（[运行时机制](./runtime-and-bus.md#22-运行结果与事件)第 2.2 节）。生命周期整理不自行捕获失败。Import Bus 以 `ImportBusEventEmitter` 投影一次导入的结局：`import.turn.accepted`（交互应用完成后发出，带实际所在的话题与是否新建）、`import.turn.duplicate`、`import.turn.skipped`（没有输出）与 `import.turn.deferred`（可重试失败或应用失败，原因码为 `gateway_unavailable`、`queue_unavailable`、`apply_pending` 或 `apply_failed`）；事件只携带来源、外部会话与轮次标识、`interaction_id`、事件数量与结构化标签，`task_type` 为 `background`。System 停止时的 `system.stopped` 以 `import_bus_shutdown` 报告 Import Bus 等待的在途请求数。submission 的 queued/running/retry/succeeded/failed 统一使用 Work Queue Runtime 的 `WORK_*` 事件。

一轮导入在 `import-` trace 中进行，MCP 的每次工具调用在 `mcp-` trace 中进行（`span_name` 为工具名），结束时以 token 复原。导入时的 Gateway 被动路由、MCP 写入时为这一轮确定话题引起的路由，以及交互提交与新建话题的事件都因此归入所在的那一次运行（第 1.1 节）。

操作入口对它分派的每一次 actor 操作在结束时发一个 `actor.operation`：预检索、MTP 的检索、读取、RUN、写入、更新与 CALL，以及 MCP 的四个工具，都经过这一处。结局在 `status`：返回为 `completed`；领域错误为 `rejected`，`reason` 是错误码；其他异常为 `failed`。取消不发事件。预检索未获授权时任务进程照常继续，这时仍会留下一条 `rejected`。

Patchouli 感知层在新建话题时发 `topic.created`，在话题以结算结束时发 `topic.settled`（带结算原因与是否提交了生成），不经结算删除时发 `topic.deleted`。准入失败、话题保留时不发 `topic.settled`。话题折叠以 Work Queue 的 `work.*` 出现，不另发话题事件。

这些事件描述状态转换，但不会驱动转换。例如 `maintenance.task.failed` 不自动重试任务，`work.retry_scheduled` 不直接修改 work record，`chat.run.cancelled` 也不由 UI event 反向取消 run；真正的业务控制分别由 scheduler、Work Queue Runtime 和 run registry 负责。

## 5. 健康与日志的关系

`HiveMemorySystem.health()`（System）汇总子系统健康和模型 ready；它是当前状态快照，不是历史审计。日志适合记录异常和调试上下文，RuntimeEvent 适合跨组件关联和 UI/运维流式消费，两者都不应成为第三份业务事实。

事件 `data` 必须摘要化：不把导入轮次的消息、思考与工具参数或结果、tool args、memory context、traceback、密钥或绝对路径放入公共观测信封。需要调试原始 cause 时应使用受保护的日志或专用诊断入口。

Agent token/MTP/CALL 交互输出不属于 RuntimeEvent。它由 Alice 的 `FrameOutputSink -> AgentRunOutput -> SinkRunOutput` 链路推送到任务进程的输出通道，再由交付方订阅，具有请求级背压；交付方离开时以停止请求取消本次 run；RuntimeEventBus 则面向全局观测，允许慢订阅者丢弃旧事件。两者只共享 correlation ID，不自动桥接，也不把 token 或完整 tool args 复制进观测信封。

### 5.1 事件记录文件

运行时事件可以写入单独的事件记录文件，与日志文件分开：日志看原因（异常与堆栈），事件记录看经过（状态转换、原因码与关联标识），两边靠 `task_id`、`process_id` 等标识对应。

- **写入**：`RuntimeEventBus.emit` 把补全序号、时间戳与 trace 之后的事件写成一行 JSON，交给专用 logger `hivememory.runtime_events`。这个 logger 不向上传递，事件不进入日志文件与 WebSocket 日志广播；级别固定为 INFO，不随 `logging.level` 丢弃事件；没有输出时总线不序列化事件。
- **文件**：`logging.event_file_path` 非空时由入口装上文件输出，每行只有事件的 JSON，按本地日期切分，方式与日志文件相同（[System 配置](../system/configuration.md)）。为空时不记录。`runtime_events.enabled` 关闭时没有总线，也就没有事件可记。开发脚本把它设为 `.hivememory/logs/runtime-events.jsonl`。
- **内容**：与 SSE 下发的信封相同，`data` 只有摘要（第 5 节的约束同样适用）；一条违反约束的事件会被永久留在文件中。
- **用途**：它只追加写入，不提供查询接口，也不回放到总线。`scripts/runtime_event_timeline.py <字段>=<值> <文件...>` 按一个关联标识（`task_id`、`process_id`、`topic_id`、`trace_id` 或 `agent_run_id`）筛出事件，按写入顺序排成时间线：系统时钟可能回跳，时间戳不一定递增。

## 6. 设计矛盾检查

评审观测改动时，检查：

1. 事件丢失、订阅者异常或 sink 关闭时，业务返回值是否仍然正确？
2. 新事件是否有稳定 `RuntimeEventType`，还是通过 `data` 临时字符串模拟类型？
3. event data 是否包含无法脱敏的外部内容、权限信息或内部异常细节？
4. 事件是否被误当成命令、重试信号、提交确认或持久化审计？
5. sequence、trace 和 correlation ID 是否能解释一次用例而不建立第二套状态机？

## 7. 生产端的约束

全部生产端都在注入的 publisher 上投影事件（第 3 节）。新增或修改生产点时保持三个约束：事件发生的时机由业务控制流显式决定；领域 emitter 只投影事实、不修改业务状态；底层 publisher 统一 scope、上下文、payload 安全转换和 best-effort 边界。

`data` 仍是按事件类型约定的字典，没有逐类型的强类型模型；每个类型的字段以[事件契约](../contracts/routes-and-events.md#4-runtimeevent-观测契约)第 4 节为准。

## 8. 验证入口

- `tests/unit/components/events/test_bus.py`
- `tests/unit/server/test_logging_setup.py`
- `tests/unit/components/events/test_publisher.py`
- `tests/integration/alice/test_events.py`
- `tests/unit/workspace/process/test_process_events.py`
- `tests/unit/components/events/test_operations.py`
- `tests/unit/components/test_trace_context.py`
- `tests/unit/system/test_lifecycle.py`
- `tests/integration/workspace/test_operation_requests.py`（actor 操作事件）
- `tests/integration/patchouli/test_perception.py`（话题事件）
- `tests/unit/system/services/import_bus/test_ingestor.py`、`tests/integration/system/test_mcp_memory_access.py`（导入与 MCP 的 trace）
- `tests/unit/architecture/test_package_layers.py`（只有 `components.events` 构造信封）
- `tests/unit/system/services/import_bus/test_ingestor.py`
- `src/hivememory/core/contracts/runtime_events.py`
