---
title: Routes and Events
status: current
owner: system
scope: global-routes-and-events
code_paths:
  - src/hivememory/system/contracts/route_names.py
  - src/hivememory/system/contracts/events.py
  - src/hivememory/system/contracts/runtime_events.py
  - src/hivememory/system/runtime/bus/
  - src/hivememory/system/runtime/events.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/error-model.md
last_reviewed: 2026-08-05
---

# 公开路由与事件

本文是进程内跨子系统路由名、事件名和投递语义的规范入口。HTTP API 不属于本文范围。

全局总线的目标是解除对象图耦合：Gateway 或 Alice 可以请求 Patchouli 的公共能力，却不需要持有它的 Runtime、Service 或存储对象。它不是为了把单体进程伪装成分布式系统，也不提供网络消息中间件的可靠性语义。明确这一点很重要，否则调用方容易把一次普通函数交接误写成无人负责的事件流，或反过来依赖总线并未承诺的持久化、重试和 exactly-once。

## 1. 总线语义

`GlobalSystemBus` 继承 `AsyncSystemBus`，只服务跨子系统公开能力。

选择 RPC 还是 Pub/Sub，取决于发布者是否必须得到一个确定结果。需要数据、确认或失败传播时使用 RPC；只需告知其他所有者“某件事已经发生”，且发布者不依赖订阅结果时使用 Pub/Sub。两者都能减少直接对象依赖，但不能互相替代。

### 1.1 RPC

- `register(route, handler)`：一个 route 对应一个 handler；重复注册会覆盖并记录 warning；
- `unregister(route)`：移除 handler，不存在时为 no-op；
- `request(route, *args, **kwargs)`：调用 handler 并等待结果；
- 未注册 route：抛出 `KeyError`；
- handler 应为 async；当前实现为兼容测试和适配器，也会直接返回非 awaitable 值。

RPC 不提供持久化、重试、跨进程投递或版本协商。

RPC 的等待关系也意味着责任关系：handler 的返回值或异常直接决定本次调用能否继续。将需要确定结果的 prepare、retrieve 或 run 改为 Pub/Sub，会让顶层链路无法知道工作何时完成，也无法可靠决定 finalize 或 cleanup。

### 1.2 Pub/Sub

- `subscribe(event, callback)`：同一事件可以有多个 async subscriber；
- `publish(event, ...)`：按订阅列表逐个等待 callback；
- 没有订阅者：静默 no-op；
- 单个 subscriber 异常：记录错误并继续，不传播给 publisher；
- `unsubscribe`：只移除指定 callback。

Pub/Sub 是通知语义，不能用于要求调用方获得确定返回值的工作流。

订阅者失败不传播给 publisher，是为了让通知的次要消费者彼此隔离；代价是事件不能承担必须成功的业务提交。若业务正确性依赖某个 subscriber 完成，就应将那一步建模为 RPC 或由明确所有者编排，而不是继续增加隐式订阅顺序。

## 2. 规范路由名

`src/hivememory/system/contracts/route_names.py::RouteNames` 是 route 字符串的代码级唯一来源。`GlobalRoutes` 和各子系统 Routes 类只重导出这些常量。

一个公开调用的契约由三部分共同构成：route 字符串确定能力身份，handler 参数名和调用形式确定交接方式，公共模型确定输入输出语义。只同步其中一项仍可能让注册方与调用方在运行时分叉，因此参数重命名、模型字段变化和 route 重命名都属于契约变更。

### 2.1 Gateway

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `gateway.public.process` | `GatewayService.process` | message、Identity、ingress mode、可选 `request_timeout_ms` | `GatewayProcessResult` |

### 2.2 Patchouli Chat / Retrieval

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `patchouli.public.submit_interaction` | `PatchouliService.submit_interaction` | `InteractionPayload`、可选 topic | 提交结果 |
| `patchouli.public.memory.retrieve` | `MemoryManagementService.retrieve` | `RetrievalRequest` | `RetrievalResponse` |
| `patchouli.public.memory.retrieve_by_aliases` | `retrieve_by_aliases` | aliases、可选 Identity | `RetrievalResponse` |
| `patchouli.public.prepare_agent_run` | `PatchouliService.prepare_agent_run` | message、user/agent/session、`GatewayDecision`、检索/生成选项 | `PreparedAgentRun` |
| `patchouli.public.finalize_agent_run` | `PatchouliService.finalize_agent_run` | `PreparedAgentRun`、`AgentRunResult` | memory task 列表 |
| `patchouli.public.cleanup_prepared_agent_run` | `cleanup_prepared_agent_run` | `PreparedAgentRun` | 是否清理空话题 |
| `patchouli.public.record_memory_citation` | `record_memory_citation` | memory id、source | 记录结果 |

### 2.3 Patchouli Memory

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `patchouli.public.memory.create` | `create_memory` | `MemoryAtom` | `MemoryAtom` |
| `patchouli.public.memory.list` | `list_memories` | query、filters、limit、exclude、refresh | `list[MemoryAtom]` |
| `patchouli.public.memory.get` | `get_memory` | memory id、refresh | `MemoryAtom | None` |
| `patchouli.public.memory.update` | `update_memory` | memory id 与可选字段 | `MemoryAtom | None` |
| `patchouli.public.memory.delete` | `delete_memory` | memory id | `bool` |
| `patchouli.public.memory.record_feedback` | `record_feedback` | memory id、positive、source | 记录结果 |

### 2.4 Patchouli Tasks / Profiles / Topics / Readiness

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `patchouli.public.memory_task.list` | `list_memory_tasks` | 无 | task 列表 |
| `patchouli.public.memory_task.get` | `get_memory_task` | task id | task 或 `None` |
| `patchouli.public.memory_task.cancel` | `cancel_memory_task` | task id | `bool` |
| `patchouli.public.agent_profile.create` | `create_agent_profile` | `MemoryAtom` | `MemoryAtom` |
| `patchouli.public.agent_profile.list` | `list_agent_profiles` | limit | profile atom 列表 |
| `patchouli.public.get_agent_profile` | `get_agent_profile` | agent alias、Identity（自定义 alias 必需） | `AgentProfile`；显式缺失/越权/无效时抛结构化 MTP error |
| `patchouli.public.topic.list_active` | `list_active_topics` | Identity、include_empty | `tuple[TopicSnapshot, ...]` |
| `patchouli.public.topic.get_data` | `get_topic_data` | Identity、topic id | 可见 `TopicData | None` |
| `patchouli.public.manual_settle_topic` | `settle_topic` | 可选 topic id | memory task 或 `None` |
| `patchouli.public.evict_topic` | `evict_topic` | topic id | 结果 dict |
| `patchouli.public.models.warmup` | `warmup_models` | 无 | `None` |
| `patchouli.public.models.ready` | `is_models_ready` | 无 | `bool` |

### 2.5 Alice

| Route | Handler | 输入摘要 | 输出 |
|:---|:---|:---|:---|
| `alice.public.run_agent` | `AgentRunService.run_agent` | `AgentRunContext`、generation options | `AgentRunResult` |
| `alice.public.run_agent_stream` | `AgentRunService.run_agent_stream` 适配器 | `AgentRunContext`、generation options | async generator 对象 |

流式 route 返回的是当前 Agent run 的交互输出流。兼容事件名保持为 `token`、`mtp_start`、`mtp_result`、`sub_agent_start`、`sub_agent_end` 和 `done`；每个事件携带 run-local `stream_sequence`，frame/CALL 事件还携带 `agent_run_id/frame_id/action_id` 等关联字段。这条流使用有界队列和背压，调用方提前断开会取消当前 runner 并沿 task cancellation 收尾，因此它属于请求执行协议的一部分，不是 RuntimeEvent 观测 SSE 的别名。

## 3. 全局业务事件

当前 `GlobalEvents` 只包含 PendingAtom 结算通知：

| Event | Publisher | Subscriber | Payload |
|:---|:---|:---|:---|
| `alice.events.pending_atom.settled` | PatchouliBridge | AliceRuntime | `settlement` |
| `alice.events.pending_atom.failed` | PatchouliBridge | AliceRuntime | `pending_alias` |
| `alice.events.pending_atom.cancelled` | PatchouliBridge | AliceRuntime | `pending_alias` |

这些事件把 Patchouli local settlement 投影回 Alice 的运行时 PendingAtom 视图。前缀中的 `alice.events` 表示消费域，不表示 Alice 是发布者。

发布者必须是 PatchouliBridge，因为只有 Patchouli 能确认延迟物化最终是 settled、failed 还是 cancelled；Alice 只是把这个长期结算事实映射回仍然存活的运行时 alias。若由 Alice 自己发布结算，PendingAtom 就会从“尚待长期系统确认的意图”变成 Alice 自证成功，破坏记忆所有权边界。

## 4. RuntimeEvent 观测契约

RuntimeEvent 不通过 `GlobalSystemBus` 发布，而通过独立 `RuntimeEventSink`。信封字段分为：

独立旁路的原因是观测不能参与业务控制。RuntimeEvent 需要支持缓冲、回放、慢订阅者隔离和 best-effort 投递，这些语义与业务 RPC 和所有权通知不同；如果把它们混在同一事件流中，UI 断连或观测 sink 失败就可能反向阻塞一次正常 chat，业务消费者也可能误把丢失的观测事件当成未发生的业务事实。

- 标识与排序：`event_id`、进程内 `sequence`、UTC `timestamp`；
- 追踪：`trace_id`、`span_name`、`task_type`；
- 来源：`source`、`subsystem`、`component`、`severity`；
- 关联 id：generation、agent run、task、agent、frame、topic、atom；
- 描述：`status`、`reason`、`message`、`data`。

当前事件组：

| 事件组 | 类型前缀或成员 |
|:---|:---|
| Chat | `chat.run.*` |
| Command | `command.executed` |
| Passive Ingress | `passive.ingress.*`、`passive.memory.context.prepared`、`passive.turn.*` |
| Gateway | `gateway.workflow.*`、`gateway.step.completed`、`gateway.analysis.capability.completed` |
| Agent | `agent.run.*` |
| Memory Task | `memory.task.*` |
| Maintenance | `maintenance.task.*` |
| System lifecycle | `system.starting/ready/start_failed/shutting_down/stopped/stop_failed` |
| Subsystem operation | `subsystem.operation.*` |
| Stream continuity | `event.stream.gap` |

### 4.1 投递语义

- `RuntimeEventBus.emit()` 尽力而为，内部失败只记录 warning；
- 每次 emit 会覆盖 timestamp 并分配递增 sequence；
- 总线保留有界 ring buffer，支持 replay last 或从 `last_event_id` 后回放；
- 找不到客户端提供的 `last_event_id` 时，先生成 `event.stream.gap`；
- 每个订阅者队列有界，满时丢弃最旧事件；
- RuntimeEvent 不承诺跨进程连续、持久化或 exactly-once。

### 4.2 与 Agent 交互输出流的边界

`alice.public.run_agent_stream` 的交互输出与 `/runtime-events/stream` 是两条独立通道：

- 交互输出只属于一次 Agent run，承载 token、MTP、CALL 边界和最终 `done`，队列满时通过背压等待，断流会触发该 run 的取消；
- RuntimeEvent 是全局扁平观测流，承载 `agent.run.started/completed/cancelled/failed` 等生命周期摘要，允许缓冲、回放和慢订阅者丢弃旧事件；
- 两条流可以通过 `generation_id/agent_run_id` 关联展示，但不得自动互相桥接；
- RuntimeEvent 的缺失或 transport 故障不能改变 Agent 结果，交互输出也不替代结构化 `TurnEvent` 与权威 run 状态。

## 5. `SystemEvent` 的状态

`SystemEvent` / `SystemEventType` 是保留的冻结生命周期模型，当前有契约测试，但实际系统生命周期观测使用 `RuntimeEventType.SYSTEM_*`。在新的生产者出现前，不应把 `SystemEvent` 推断为正在运行的第二套事件流。

## 6. 设计矛盾检查

评审新的 route 或 event 时，应检查：

1. 调用方是否必须获得确定结果？如果是，为什么选择不会传播 subscriber 失败的 Pub/Sub？
2. 这是否只是观测信息？如果是，为什么要进入会影响业务顺序的 GlobalSystemBus，而不是 RuntimeEventSink？
3. route、handler 参数名和公共模型是否作为一个整体更新，还是出现了代码能注册、调用方却按旧语义传参的情况？
4. 新事件的发布者是否真正拥有被声明的事实，订阅者是否只维护投影而没有取得权威所有权？
5. 外部模块是否开始引用 local route，或为了绕过公共模型而直接持有另一个 Runtime？
6. 实现是否依赖未承诺的持久化、自动重试、跨进程连续或 exactly-once？
7. RuntimeEvent 丢失、重复或 sink 失败时，业务结果是否仍然完全由返回值、异常和权威状态决定？

## 7. 变更规则

- 新公开 route 必须先加入 `RouteNames`，再由子系统 Routes 类重导出；
- route handler 的参数名也是调用契约的一部分；
- 新跨子系统通知应先判断需要 RPC 返回值还是 Pub/Sub；
- 新观测事件加入 `RuntimeEventType`，不得通过 `data` 中的临时字符串模拟事件类型；
- 删除或重命名 route/event 前必须更新所有注册方、调用方、测试和本文；
- local route 不记录在本规范中，也不得被外部模块直接引用。

主要验证入口：`tests/unit/system/contracts/test_public_routes.py`、`tests/unit/system/runtime/test_runtime_events.py`、`tests/unit/system/runtime/bus/test_async_bus.py`。
