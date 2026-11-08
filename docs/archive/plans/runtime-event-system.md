---
title: Runtime Event 体系完善
status: archived
archived_at: 2026-11-07
implemented_by: PR #193
superseded_by: docs/components/observability.md
owner: components
target: 版本之外的工作（观测与日志由 owner 于 2026-11-03 定为不绑定版本；2026-11-07 决定全局运行页以事件为主视图，本计划是它的前提）
scope: run-grouping-actor-operation-and-topic-events-catalog-and-producer-convergence
updates:
  - docs/components/observability.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/mcp.md
  - docs/governance/observability/logs-and-events.md
  - docs/gateway/workflow.md
  - docs/workspace/capability.md
  - docs/patchouli/perception.md
  - docs/system/import-bus.md
  - docs/frontend/state-and-transports.md
  - docs/archive/ideas/runtime-event-system-and-run-view.md
  - docs/ideas/README.md
  - docs/ROADMAP.md
---

# Runtime Event 体系完善

本计划实施[正式的 Runtime Event 体系与全局运行页 Idea](../ideas/runtime-event-system-and-run-view.md)第 3.1 至 3.4 节：让每一次运行的事件能归到一组，补齐 actor 操作与话题生命周期的事件，事件目录只在事件契约一处维护，全部生产端经领域 emitter 投影（观测治理 W6）。运行页本身（Idea 第 3.5 节）在本计划完成、新事件成为当前事实之后另写计划。对 Idea 候选设计的调整见第 4.8 节。

## 1. 背景和问题

owner 于 2026-11-07 决定：全局运行页的主视图是运行时事件，后端日志作为第二个标签；为此要先正式建立完善的 runtime event 体系（Idea 第 2 节）。运行页要回答"每一次运行从开始到结束发生了什么"，现状在四个方面做不到：

1. **事件挂不到所属的运行上**。trace 只有 chat 任务进程会设置（`workspace/process/runner.py`）。导入与 MCP 访问的事件都落在 `trace_id=system`，其中导入时 Gateway 被动路由发出的 workflow 与 step 事件，只能靠 Workspace 与时间猜它属于哪一次导入。按 2026-11-04 至 11-07 的事件记录文件，每天 336 至 660 条事件，`gateway.step.completed` 一项就占三分之一以上，大部分来自导入。
2. **覆盖缺口**。actor 操作（预检索、MTP 的检索、读取、RUN、写入、更新、CALL，以及 MCP 的四个工具）都不发事件，写入与更新只能从随后的 `memory.task.*` 间接看到；话题的建立、结算与删除不发事件。
3. **定义了但没有生产方的类型**。`command.executed`、`gateway.workflow.cancelled`、`agent.run.status` 只在枚举里，代码中没有任何地方发出。
4. **生产端两种写法**。一部分经 `RuntimeEventPublisher` 上的领域 emitter 投影，一部分直接构造 `RuntimeEvent` 交给 sink；注入的也一半是 publisher、一半是 sink（第 3.1 节）。默认 severity、关联字段与 best-effort 边界因此在各域漂移。

## 2. 目标与非目标

### 2.1 目标

1. chat 任务进程、一次导入、一次 MCP 工具调用与一个生成任务，各自的事件都能按第 4.2 节的规则归为一组。
2. actor 操作与话题生命周期有事件（第 4.3、4.4 节）。
3. 事件目录只在[事件契约](../../contracts/routes-and-events.md#4-runtimeevent-观测契约)第 4 节维护：每个类型写明生产方、分组键、是否终态与 `data` 字段；删除三个没有生产方的类型。
4. 全部生产端经领域 emitter 投影到 `RuntimeEventPublisher`，组装根注入 publisher 而不是 sink；已有类型的 wire format 不变（第 6 节写明的两处例外除外）。
5. 前端的事件类型与回放对齐实际契约（第 4.9 节）。

### 2.2 非目标

- 运行页、Chat 右栏终端标签的移除与日志标签（另一份计划）。
- 事件存储、查询接口、回放到总线；内存缓冲与订阅者队列的大小不变。
- 事件参与业务判断：事件仍是 best-effort 旁路，不替代返回值、异常与 Store 状态。
- 交互提交另发事件、话题折叠发事件、MCP 认证失败发事件（理由见第 4.8 节与第 4.3 节）。
- 生成任务与维护任务设置 trace：它们按 `task_id` 或单条事件展示。

## 3. 当前状态和缺口

### 3.1 生产端

| 生产方 | 类型 | 写法 | 组装根注入 |
|:---|:---|:---|:---|
| chat 任务进程（`TaskProcessEventEmitter`） | `chat.run.*` | 领域 emitter，publisher | 根 publisher |
| Alice（`AgentRunEventEmitter`） | `agent.run.*`、`agent.context.*` | 领域 emitter，publisher | 根 publisher 的 `alice` 作用域 |
| 生成任务（`MemoryTaskEventEmitter`） | `memory.task.*` | 领域 emitter，自己用注入的 sink 新建 publisher | sink |
| Work Queue（`WorkQueueEventEmitter`） | `work.*` | 领域 emitter，自己新建 publisher 并写死 `system` / `work_queue` 标签 | sink |
| Import Bus（`ImportBusEventEmitter`） | `import.turn.*` | 领域 emitter，但直接构造 `RuntimeEvent` | sink |
| Gateway workflow 与查询分析 | `gateway.*` | 直接构造 `RuntimeEvent` | sink |
| 维护调度器 | `maintenance.task.*` | 直接构造 | sink |
| System 生命周期 | `system.*` | 直接构造 | 根 sink |
| 操作观测（`RuntimeOperationObserver`，只用于 Patchouli 关闭时的 drain） | `subsystem.operation.*` | 直接构造 | sink |

`ScopedRuntimeEventSink` 只补缺失的标签，生成任务与 Work Queue 写死的标签因此盖过了注入的作用域：全部 `work.*` 都显示 `subsystem=system`。

### 3.2 归属

- chat：`chat.run.*` 显式绑定进程的 `trace_id`；Gateway、Alice 与 actor 操作都在进程 task 中运行，从 contextvars 继承同一个 trace。`trace_id` 是把 Gateway 事件连到进程上的唯一键，没有测试守护。
- 导入：`ImportBusIngestor` 不设 trace。Gateway 的被动路由（`route_passive_turn`）、交互提交的 `work.queued` 与 `import.turn.*` 都是 `trace_id=system`；被动路由事件的 `task_type` 还取默认的 `foreground`。
- MCP：工具调用不设 trace。写入与更新经 `determine_topic` 回调调用 `ImportBusIngestor.place`，同样会触发 Gateway 被动路由。
- 生成任务：`memory.task.created` 继承提交者的 trace，此后的状态与终态事件在 worker 中发出，trace 为 `system`；它们都带 `task_id`。

### 3.3 覆盖

- **actor 操作**：`WorkspaceOperationEntry` 是 actor API 唯一的调用方（架构测试守护），知道请求类型、工作 actor 与结局，但丢弃了线程解析结果中的 `process_id`，也不发事件。预检索被拒时任务进程照常继续，没有任何记录。
- **交互提交**：两条链路已经在各自的结局处覆盖：chat 的 `chat.run.completed` 带本轮话题，提交失败是 `chat.run.failed` 的原因码；导入是 `import.turn.*`。
- **话题**：`PerceptionFamiliar` 没有拿到事件 sink（Patchouli 运行时持有，但构造感知服务时没有传入）。结算原因（`TriggerReason`）也没有进入生成任务，`memory.task.*` 只带 `topic_id`。

### 3.4 消费方

- 前端 `RuntimeEvent` 类型缺 `workspace_id`；`RuntimeEventType` 联合类型只列了 chat、agent、memory task 与 gap，其余类型照常到达，只是类型不对。
- 生成任务的来源后端会发 `SETTLE`，前端联合类型没有，映射时回退为 `WRITE`：Memory Runtime 面板把结算任务显示成写入任务。
- `/runtime-events/stream` 的回放点同时接受查询参数 `last_event_id` 与 `Last-Event-ID` 请求头，查询参数优先。前端把首次连接时的回放点写在 URL 里，浏览器自动重连时请求头带的是更新的事件，但仍按 URL 中旧的那个回放，重连后会重复收到事件。

## 4. 目标设计

### 4.1 owner 决定（2026-11-07）

1. Terminal 入口是全局的后端观测页（运行页），主视图是运行时事件，后端日志作为第二个标签。
2. 先正式建立完善的 runtime event 体系，即本计划。
3. 运行页只看有限的事件历史，事件记录文件不提供查询接口。
4. Chat 右栏的终端标签移除（随运行页的计划实施）。
5. actor 操作的事件维持 `actor.operation` 的命名，不套用通用的 `subsystem.operation`；一次操作只发一个事件（计划评审）。`subsystem.operation` 保持原样，仍是一次性、按耗时与结局度量的操作所用的通用类型，目前只有 Patchouli 关闭时的 drain 使用。
6. 检索与其他 actor 操作一样，每次都发 `actor.operation` 事件；第 4.9 节的两个现存缺陷（结算任务的来源、回放点的优先级）随本计划修复（计划评审）。

以下第 4.2 至 4.9 节是候选设计，不是 owner 决定，可在计划评审时调整。

### 4.2 一次运行与分组规则

一条事件按下列规则归组，先匹配的规则生效：

1. `memory.task.*` 与 `work.*` 按 `task_id` 归组（Work Queue 的 `task_id` 是 work id）：它们是队列中的任务，有自己的起止，跨越提交者与 worker 两个 context。
2. 其余 `trace_id` 不是 `system` 的事件，按 `trace_id` 归组。设置 trace 的运行有三种：chat 任务进程（`task-` 前缀，已有）、一次导入（`import-`，第 4.5 节）、一次 MCP 工具调用（`mcp-`，第 4.5 节）。
3. 剩下的事件各自成为单条运行：`maintenance.task.*`、`system.*`、`subsystem.operation.*`，以及空闲结算等不在任何运行内发生的 `topic.*`。

不同运行之间经 `topic_id` 互相链接：导入或对话写入话题，话题结算引起生成任务。页面显示链接，不合并运行。

各事件族的起止：

| 事件族 | 起 | 终 |
|:---|:---|:---|
| `chat.run.*` | `created` | `completed`、`cancelled`、`failed` |
| `agent.run.*` | `started` | `completed`、`cancelled`、`failed` |
| `gateway.workflow.*` | `started` | `completed`、`failed`；取消直接离开 Gateway，由所在运行的终态收尾 |
| `import.turn.*` | 无 | `accepted`、`duplicate`、`skipped`、`deferred` 之一 |
| `actor.operation` | 无 | 一次操作一个事件，`status` 为 `completed`、`rejected` 或 `failed` |
| `topic.*` | `created` | `settled`、`deleted` |
| `memory.task.*` | `created` | `completed`、`cancelled`、`failed` |
| `work.*` | `queued` | `succeeded`、`cancelled`、`dead_lettered`；`rejected` 单独出现 |
| `subsystem.operation.*` | `started` | `completed`、`failed` |
| `system.*` | `starting`、`shutting_down` | `ready`、`start_failed`、`stopped`、`stop_failed` |

这条规则写进[运行时事件与可观测性](../../components/observability.md)，运行页的计划依据它分组。

### 4.3 actor 操作事件

`WorkspaceOperationEntry` 对它分派的每一次 actor 操作在结束时发一个 `actor.operation` 事件，不发开始事件：线程提交（`execute`）、请求级提交（`execute_request`，即 MCP）与 CALL 派生（`derive_child_thread`）。`end_thread` 不是 actor 操作，不发。结局放在 `status`：

| `status` | severity | 何时 |
|:---|:---|:---|
| `completed` | info | 操作返回 |
| `rejected` | warning | 操作以领域错误拒绝：未获授权、线程已结束、目标不存在、意图派发被拒等；`reason` 是错误码 |
| `failed` | error | 其他异常；`reason=actor_operation_failed`，`data.error_class` 为异常类名，异常原样抛出 |

取消不发事件，与 Gateway 一致。预检索未获授权时任务进程照常继续，这时会出现一条 `rejected`，如实反映这次拒绝。

- **信封**：`subsystem=workspace`、`component=operation_entry`；`agent_id` 与 `workspace_id` 取工作 actor 的 Agent 与驻留 Workspace，只写标签，工作 actor 本身不进入事件；线程提交另带线程解析得到的 `process_id`；trace 从所在运行继承。
- **`data`**：`operation`（`WorkspaceOperation` 的值，如 `memory.search`）、`entry`（`thread` 或 `request`）、`duration_ms`，以及按操作的摘要，不含任何内容：

| 操作 | 摘要 |
|:---|:---|
| `memory.search` | `result_count` |
| `memory.read` | `requested_count`、`delivered_count`（解析为正式记忆或重定向的数量） |
| `memory.run` | `resolution`（解析结果的 kind） |
| `memory.write`、`memory.update` | `pending_alias` |
| `agent.call` | `callee_agent_id` |

- **位置**：emitter `ActorOperationEventEmitter` 放在 `workspace/capability/operation_events.py`，经构造参数注入操作入口。架构测试锁定了入口的方法集合与分派表，也不允许 `actor/` 下出现新的公开类，因此不新增方法，只在现有三个方法里调用 emitter；emitter 提供一个包住分派的观测方法，让入口的控制流保持平铺。线程提交改为保留线程解析结果，取出 `process_id` 与工作 actor。
- **MCP 认证失败**发生在入口之前，MCP 已有日志，不发事件。

### 4.4 话题事件

`PerceptionFamiliar` 发话题的三个事件，emitter `TopicEventEmitter` 放在 `patchouli/services/perception_events.py`，由 Patchouli 运行时在构造感知服务时注入：

| 类型 | 发出位置 | `data` |
|:---|:---|:---|
| `topic.created` | `_ensure_topic` 新建话题时 | `created_by`：`interaction`（应用交互时建立）或 `prepare`（为写入预先确定话题时建立） |
| `topic.settled` | `_settle_topic`，四种结算共用 | `reason`：`TriggerReason` 的值（`idle_timeout`、`manual_settle`、`shutdown`、`lru_eviction` 等）；`generation_submitted`：是否提交了生成，没有材料时为 false |
| `topic.deleted` | 不经结算直接删除话题（管理删除，`manual_delete`） | 无 |

信封带 `topic_id` 与 `workspace_id`，`subsystem=patchouli`、`component=perception`。在导入、MCP 写入或对话中建立的话题继承所在运行的 trace，归入那一次运行；空闲结算在维护任务中发生，是单条运行。结算提交的生成任务以同一个 `topic_id` 出现在 `memory.task.created` 中，运行页据此链接。

话题折叠以 Work Queue 的 `work.*` 出现，本计划不另发事件：折叠频繁，也不改变话题归属。

### 4.5 导入与 MCP 的 trace

- **导入**：`ImportBusIngestor.ingest` 在一轮导入的全程设置 trace（`generate_trace_id("import")`，`span_name=ImportBus.Turn`，`task_type=background`），结束时以 token 复原，写法与任务进程相同。被动路由的 Gateway 事件、交互提交的 `work.queued`、新建话题的 `topic.created` 与 `import.turn.*` 因此共用这个 trace。
- **MCP**：四个工具调用各自设置 trace（`mcp-` 前缀，`span_name` 为工具名，`task_type=foreground`：外部 agent 在等待结果）。`ImportBusIngestor.place` 只由 MCP 写入的话题确定回调调用，不另设 trace，继承工具调用的 trace。
- 生成任务与维护任务不设 trace，按第 4.2 节第 1、3 条展示。

trace 同时进入日志格式（`[trace_id/span_name]`），导入与 MCP 的日志从此也能按一次运行查找。

### 4.6 删除没有生产方的类型

- `command.executed`：命令只以 `data.command_id` 出现在 `chat.run.completed` 与 `gateway.workflow.completed` 中，命令在进程中何时、由谁运行仍是搁置问题（Q-5a）。
- `gateway.workflow.cancelled`：取消直接离开 Gateway、不发事件是有意的设计（[Gateway Workflow](../../gateway/workflow.md#7-观测不是控制面)第 7 节），所在运行的终态收尾。
- `agent.run.status`：Alice 只发起止事件。

### 4.7 生产端收敛

| 生产方 | 改为 |
|:---|:---|
| Gateway workflow 与查询分析 | `GatewayEventEmitter`（`gateway/events.py`），提供 workflow、step 与 capability 的方法 |
| 维护调度器 | `MaintenanceEventEmitter`（`components/scheduler/events.py`） |
| System 生命周期 | `SystemLifecycleEventEmitter`（`system/lifecycle_events.py`） |
| Import Bus | `ImportBusEventEmitter` 改为在 publisher 上投影 |
| 操作观测 | `RuntimeOperationObserver` 改为接收 publisher |
| 生成任务、Work Queue | 接收注入的 publisher，不再自己用 sink 新建；写死的标签值保持不变 |

组装根为每个子系统注入带作用域的 publisher，sink 只留在总线与 publisher 内部。emitter 只投影事实、不修改业务状态，事件发生的时机仍由业务控制流显式决定；publisher 的异常隔离统一了 best-effort 边界。

新增一条架构测试：只有 `components/events` 构造 `RuntimeEvent`，防止生产端回到直接组装信封。

### 4.8 相对 Idea 候选设计的调整

- **交互提交不另发事件**：Idea 第 3.3 节建议在 `submit_interaction` 发提交结局，但两条链路已经在各自的结局处覆盖（第 3.3 节），再发一份是同一事实的重复表达。
- **分组靠 trace，不把 `interaction_id` 提升为信封字段**：Idea 第 3.1 节以 `interaction_id` 作为导入的分组键，并把它列为非目标"不为了分组把 trace 扩展到所有入口"。实际上导入的大部分事件来自 Gateway，Gateway 不知道 `interaction_id`；为导入与 MCP 设置 trace，一处改动就让嵌套事件全部归组，日志也随之归组。trace 只扩展到这两个入口，不扩展到生成任务与维护任务。
- **actor 操作在操作入口发，不在 actor API 内发**：入口是唯一调用方，统一知道操作种类、`process_id` 与结局，也避开了 `actor/` 下的架构约束。
- **话题有 `deleted`**：Idea 写的"回收"在代码中是带 `lru_eviction` 原因的结算；不经结算直接删除另需一个终态。

### 4.9 消费方对齐

- 前端 `RuntimeEvent` 类型补上 `workspace_id`，`RuntimeEventType` 联合类型按事件目录列全。
- 生成任务来源加入 `SETTLE`，Memory Runtime 面板不再把结算任务显示成写入任务。
- `/runtime-events/stream` 改为 `Last-Event-ID` 请求头优先于查询参数，浏览器自动重连按最新的事件续传。
- Chat 右栏终端标签的事件视图不改，它按通用字段展示，新类型照常出现。

## 5. 受影响的系统、契约与数据

- **core**：`RuntimeEventType` 删除三个类型，新增 `actor.operation` 与三个 `topic.*`，共四个。
- **components**：`events`（`RuntimeOperationObserver` 接收 publisher）、`scheduler`（新 emitter）、`work_queue`（接收 publisher）。
- **workspace**：操作入口与新 emitter；组装根改为注入 publisher。
- **patchouli**：感知服务与新 emitter，运行时构造时注入；生成任务 controller 接收 publisher。
- **gateway**：workflow 与查询分析改用新 emitter。
- **system**：生命周期 emitter、Import Bus 的 trace 与 emitter、组装根。
- **mcp_server**：工具调用设置 trace。
- **server**：`/runtime-events/stream` 的回放点优先级。
- **frontend**：`types/workspace.ts`、`types/memoryTask.ts` 与 `stores/memory/runtimeEventTaskMapper.ts`。
- **契约**：事件契约第 4 节改为事件目录表；[MCP 记忆访问](../../contracts/mcp.md)写明工具调用的 trace 与 actor 操作事件。
- **数据**：没有持久化结构变化；事件记录文件出现新类型，导入与 MCP 的事件带上各自的 trace。

## 6. 兼容与迁移

- 已有类型的字段与取值不变，由各生产方现有的精确断言测试守护。两处例外都是本计划要的修正：导入与 MCP 运行内的事件（Gateway、`work.queued` 等）的 `trace_id` 从 `system` 变为该次运行的 trace，`task_type` 随 trace 设置；`/runtime-events/stream` 的回放点优先级。
- 删除的三个类型从未发出，没有消费方受影响；前端联合类型同步删除 `agent.run.status`。
- 事件记录文件的旧记录照常可读，`scripts/runtime_event_timeline.py` 已支持按 `trace_id` 与 `task_id` 筛选，不需要改。
- 没有配置项变化。

## 7. 测试和验收标准

需要守住的不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 每次 actor 操作在结束时发一个 `actor.operation` 事件：`completed` 带摘要，线程提交带 `process_id`；`rejected` 带错误码；`failed` 带 `error_class` 并原样抛出；取消不发 | 操作入口的现有测试 `tests/integration/workspace/test_operation_requests.py` |
| MCP 工具调用的 actor 操作事件不带 `process_id`；同一次写入的 actor 操作与 Gateway 事件共用一个 `mcp-` trace | `tests/integration/system/test_mcp_memory_access.py` |
| 话题建立、结算（带原因与是否提交生成）与删除各发一个事件 | `tests/integration/patchouli/test_perception.py` |
| 一次导入的 Gateway 事件与 `import.turn.*` 共用一个 `import-` trace；导入结束后 trace 复原 | `tests/unit/system/services/import_bus/test_ingestor.py` |
| 已有类型的 wire format 不变 | 各生产方现有测试保持断言不变（Gateway、调度器、System 生命周期、Import Bus、操作观测、生成任务、Work Queue） |
| 只有 `components/events` 构造 `RuntimeEvent` | 新架构测试，放在 `tests/unit/architecture/` |
| `Last-Event-ID` 请求头优先于查询参数 | `tests/unit/server/routers/test_runtime_events.py` |
| 结算任务在前端显示为结算来源 | `frontend/tests/` 中新增一个映射用例 |

- 契约有意变更的旧测试同步改写：`tests/unit/gateway/test_request_control.py` 断言取消时没有 `gateway.workflow.cancelled`，改为断言取消时 Gateway 不发终态事件，不再引用删除的类型。
- 手动验收：用本分支重启正在使用的后端，配置事件记录文件，然后：
  1. 前端对话一轮，含一次 WRITE；
  2. Claude Code 正常使用，产生一次导入、一次 MCP 检索与一次 MCP 写入；
  3. 等一个话题空闲结算；
  4. 分别用 `scripts/runtime_event_timeline.py trace_id=<…>` 与 `task_id=<…>` 查看，每一次运行的事件完整归组、有终态，结算的生成任务能经 `topic_id` 对应到话题。

验证命令：

```bash
pytest tests/unit/workspace tests/unit/architecture tests/integration/workspace tests/integration/patchouli/test_perception.py tests/unit/system tests/unit/gateway -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm test && npm run build
```

## 8. 风险与待决问题

- **事件量**：按事件记录文件，现在每天 336 至 660 条；MCP 请求每天 5 至 390 次（含初始化等非工具调用），chat 每轮多一条预检索事件。新增的 actor 操作与话题事件加上后，1000 条的内存缓冲仍约能覆盖一天，不调整大小；运行页的计划再按实际使用核对。
- **操作入口的结构**：入口的方法集合被架构测试锁定，观测只能放进现有方法。缓解：emitter 提供包住分派的观测方法，入口的控制流仍是平铺的一次调用。
- **trace 泄漏**：导入与 MCP 设置 trace 后必须在 `finally` 中以 token 复原，否则后续请求会继承旧 trace。用例覆盖导入结束后的复原。
- **wire format 回归**：生产端改写时字段漂移。缓解：现有生产方测试的精确断言保持不改；只有第 6 节的两处例外改写断言。
- **待决**：没有。计划评审中的决定见第 4.1 节第 5、6 条。

## 9. 完成时必须更新的当前文档

- [运行时事件与可观测性](../../components/observability.md)：第 1 节写入分组规则与 trace 的三种运行；第 3、4、7 节改为全部经领域 emitter，删去"渐进迁移"的现状描述。
- [事件契约](../../contracts/routes-and-events.md)第 4 节：事件组表改为事件目录（类型、生产方、分组键、是否终态、`data` 字段），失败原因码表补上 `actor.operation`。
- [MCP 记忆访问](../../contracts/mcp.md)：工具调用的 trace 与 actor 操作事件。
- [日志与运行时事件](../../governance/observability/logs-and-events.md)：W6 改为已完成。
- [Gateway Workflow](../../gateway/workflow.md)第 7 节：改用 emitter；取消不发事件的说明不再提到删除的类型。
- [能力层](../../workspace/capability.md)：操作入口发 actor 操作事件。
- [感知与短期话题](../../patchouli/perception.md)：话题事件。
- [Import Bus](../../system/import-bus.md)：一轮导入的 trace。
- [状态与传输](../../frontend/state-and-transports.md)：回放点的优先级。
- Idea 与 Ideas 索引：登记第 3.1 至 3.4 节已实施，运行页仍待计划。
- [ROADMAP](../../ROADMAP.md) 的观测与日志一行。

## 10. 实施记录

2026-11-07 在分支 `docs/runtime-event-system-plan` 上实施，与第 4 节的出入：

- **不另建 emitter 类**：Gateway workflow 与查询分析、维护调度器、System 生命周期与操作观测原本就把发事件集中在一个辅助方法里，只把其中的 sink 换成注入的 publisher，没有新建第 4.7 节表中的 `GatewayEventEmitter`、`MaintenanceEventEmitter` 与 `SystemLifecycleEventEmitter`。话题事件同样由 `PerceptionFamiliar` 的两个私有方法发出，没有单独的 `perception_events.py`。只有操作入口需要单独的 `ActorOperationEventEmitter`：入口的方法集合被架构测试锁定。
- **emitter 的位置**：`ActorOperationEventEmitter` 放在 `workspace/capability/operation_events.py`。workspace 包根下的模块只能依赖 core，因此不放在包根；操作入口的导入守护本意是"只用 actor API、不用 root API"，白名单加上这个事件模块。
- **线程已结束的拒绝**：`execute` 在分派之前解析线程，线程已结束时直接拒绝、不发事件；写入与更新在 actor API 内再次解析线程时被拒，会发 `rejected`。
- **前端 mock 任务**：来源类型对齐后，debug 模式的 mock 任务中的 `ARCHIVE`、`SPLIT` 改为 `SETTLE`、`MERGE`。

验证：

- `pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85`：3236 个通过，覆盖率 94.85%；`ruff check .`、`black --check .`、`mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py`（446 个文件）通过。
- 前端 `node --test tests/*.test.mjs` 41 个通过，`npm run lint` 与 `npm run build` 成功。
- 反向确认：结算任务来源、回放点优先级、MCP 的 trace 与导入的 trace 四个新用例，在改动之前的代码上都失败。
- 实际使用核对（2026-11-07，后端以合并后的 master 重启）：MCP 的检索、读取与更新各成一次 `mcp-` 运行，更新为这一轮确定话题引起的 Gateway 路由、新建话题与生成任务入队都在同一个 trace 下；Claude Code 的每轮回流各成一次 `import-` 运行，写入时已确定话题的那一轮回流沿用话题、不再路由；前端可以看到更新的事件（owner）。归档前没有观察到话题的空闲结算。
- owner 2026-11-07 决定：实际验收不阻止计划归档，实现合并、测试通过即归档，之后的实际使用观察另行跟进。

归档后的后续：同日 owner 决定删除 `gateway.step.completed`，各步结果并入 Gateway workflow 终态事件的 `data`（`step_durations_ms` 与 `fallbacks`），当前事实见[事件契约](../../contracts/routes-and-events.md#44-失败与降级观测)。
