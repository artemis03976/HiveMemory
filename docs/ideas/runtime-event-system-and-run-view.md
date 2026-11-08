---
title: Runtime Event System and the Global Run View
status: idea
horizon: candidate
owner: components
scope: runtime-event-catalog-coverage-producers-and-global-run-view
code_paths:
  - src/hivememory/core/contracts/runtime_events.py
  - src/hivememory/components/events/
  - src/hivememory/workspace/process/runner.py
  - frontend/src/components/chat/WorkspaceVision.tsx
  - frontend/src/components/chat/WorkspaceTerminalTab.tsx
  - frontend/src/stores/workspace/
related_docs:
  - docs/components/observability.md
  - docs/governance/observability/logs-and-events.md
  - docs/contracts/routes-and-events.md
  - docs/frontend/chat-view.md
  - docs/frontend/README.md
last_reviewed: 2026-11-07
---

# 正式的 Runtime Event 体系与全局运行页

本文讨论两件互相依赖的事：前端用什么展示"后端正在做什么"，以及为此需要把运行时事件建成怎样的体系。owner 已于 2026-11-07 决定运行页以事件为主视图（第 2 节），本文其余部分是候选设计与默认值，不是 owner 决定，可在计划评审时调整。

2026-11-07：第 3.1 至 3.4 节经计划 [Runtime Event 体系完善](../plans/runtime-event-system.md) 实施，当前事实见[运行时事件与可观测性](../components/observability.md#11-一次运行)与[事件契约](../contracts/routes-and-events.md#4-runtimeevent-观测契约)，计划对本文候选设计的调整见其第 4.8 节。运行页（第 3.5 节）仍待另写计划。

事件机制、信封与 best-effort 语义的当前事实见[运行时事件与可观测性](../components/observability.md)，日志写法与事件生产端收敛的治理见[日志与运行时事件](../governance/observability/logs-and-events.md)，本文不重复。

## 1. 背景：两路观测在前端互相冲突

### 1.1 前端现在怎样展示

- **入口**：全局导航的 Terminal 入口在 `App.tsx` 中没有渲染分支，选择后主区域为空（[前端索引](../frontend/README.md)）。实际的观测面板做在 Chat 页右栏 Workspace Vision 的"终端"标签里（[Chat 视图](../frontend/chat-view.md#53-workspace-terminal)第 5.3 节）。
- **日志视图**：后端日志经 `/api/v1/ws/logs` WebSocket 广播，前端按 `trace_id`、`span_name` 折叠。trace 只有 chat 任务进程会设置（`workspace/process/runner.py`），Import Bus、MCP 与后台维护的日志都落在 `system/main` 一组里。
- **事件视图**：同一标签可以切到事件，事件经 `/api/v1/runtime-events/stream` 进入，展示为平铺列表，每条只标出 `task_id`、`process_id`、`agent_run_id` 或 `trace_id` 中的一个。
- **记忆任务**：右栏的 Memory Runtime 标签把 `memory.task.*` 事件投影为任务列表，这是事件目前唯一按语义消费的地方（[Chat 视图](../frontend/chat-view.md#52-memory-runtime)第 5.2 节）。

### 1.2 冲突在哪里

- **分组做在了日志上**：[运行时事件与可观测性](../components/observability.md)开头写明，事件保留 trace 与关联字段，是为了让前端按运行上下文分组、折叠；实现却是日志按 trace/span 折叠、事件平铺。日志是自由文本，同一模块中英文混用，可能带异常正文，不适合作为界面数据；而 trace 只覆盖 chat 任务进程，日常使用的 plugin 模式（Claude Code、dsh 经 Import Bus 与 MCP）看不到分组。
- **位置放错了**：观测面板挂在 Chat 页，Chat 页是可选的 controller 式入口，而后端观测是全局的：导入、MCP 访问、生成任务与维护都不经过 Chat 页。

### 1.3 事件还不是一个完整的体系

运行页要以事件为主视图，事件需要能回答"每一次运行从开始到结束发生了什么"。现状离这一点还有距离：

- **覆盖缺口**：MCP 的检索、读取、写入与更新，以及 actor API 的其他记忆操作，本身不发事件，写入与更新只能从随后的 `memory.task.*` 间接看到；检索不发事件；话题的建立、折叠、结算与回收在感知层没有事件；主动链路的交互提交只体现在 `chat.run.*` 的结局里。
- **定义了但没有生产方**：`command.executed`、`gateway.workflow.cancelled`、`agent.run.status` 三个类型在 `RuntimeEventType` 中定义，代码中没有任何地方发出。
- **生产端两种写法**：Alice、chat 任务进程、Import Bus、生成任务与 Work Queue 经领域 emitter 投影事件；Gateway、Scheduler、System 生命周期与 Patchouli 的操作观测仍直接组装信封。默认 severity、关联字段与 payload 白名单因此在各域漂移（治理主题的 W6）。
- **关联不连贯**：`memory.task.*` 只带 `task_id` 与 `topic_id`，不带 `process_id`；一次导入的 `interaction_id` 只在 `data` 中。从一次导入或一轮对话追到它引起的生成任务，只能经由话题间接对应。

## 2. owner 决定（2026-11-07）

1. Terminal 入口（候选 Feature 清单 D-05）是全局的后端观测页，不是终端 CLI（issue #3）；它与前端 UI 优化一起，在桌面打包之前完成。本文称它为运行页。
2. 运行页的主视图是运行时事件，后端日志作为同一页面的第二个标签，用于开发诊断。
3. 这要求先正式建立完善的 runtime event 体系。
4. 运行页只看有限的事件历史，即内存缓冲与断线续传能覆盖的范围，对普通情况足够；事件记录文件不提供查询接口这一非目标不变。
5. Chat 右栏的终端标签移除。

## 3. 候选设计

以下各项都是默认值，不是 owner 决定。

### 3.1 一次运行的定义

运行页按"运行"分组展示事件。每个事件族有一个分组键和一对起止事件，嵌套的事件挂在所属运行之下：

| 运行 | 分组键 | 起止 |
|:---|:---|:---|
| 一轮对话（chat 任务进程） | `process_id` | `chat.run.created` 到 completed、cancelled 或 failed；`agent.run.*`（带 `process_id`）与 Gateway 事件（同一 `trace_id`）嵌套其中 |
| 一次导入 | `interaction_id` | 只有一个结局事件（`import.turn.*`） |
| 一个生成任务 | `task_id` | `memory.task.created` 到终态 |
| 一次维护 | 维护任务的 key | `maintenance.task.completed` 或 failed |
| 系统启停 | 无 | `system.*` |

不同运行之间经 `topic_id` 互相链接（导入或对话写入话题，话题结算引起生成任务），页面显示为链接，不合并成一个运行。分组键目前只在 `data` 中的（如导入的 `interaction_id`），默认提升为信封字段。

### 3.2 事件目录

- 每个事件族都有终态事件，运行页才能判断一次运行是否结束；没有生产方的类型，要么补上生产方，要么删除。
- 事件目录只在[事件契约](../contracts/routes-and-events.md#44-失败与降级观测)一处维护，写明每个类型的生产方、分组键与 `data` 字段。

### 3.3 补齐覆盖

新事件只在唯一入口发出，不在每个适配器各发一次：

- **记忆操作**：actor API 是全部记忆能力的唯一入口，预检索、MTP 与 MCP 都经过它；每次操作在入口发一个带结局的事件，不发 started。
- **交互提交**：root API 的 `submit_interaction` 是 chat 与 Import Bus 共用的唯一提交入口，在这里发提交的结局。
- **话题**：Patchouli 感知层发话题的建立、结算与回收；折叠频繁且不改变话题归属，默认不发。

检索在 plugin 模式下非常频繁，每次一个事件会更快挤出 1000 条的内存缓冲，评审时需要用真实使用的事件量核对缓冲大小。

### 3.4 生产端收敛

治理主题 W6 的范围并入本文：全部生产端经领域 emitter 投影到 `RuntimeEventPublisher`，不再直接组装信封；已有类型的 wire format 不变。

### 3.5 运行页与日志标签

- **运行页主视图**：按第 3.1 节分组，显示进行中与最近结束的运行，失败处显示原因码；可按子系统、severity 与 workspace 筛选；回放窗口不再覆盖时显示 `event.stream.gap`。
- **日志标签**：平铺的日志尾部，按级别、logger 与 `trace_id` 筛选，不再做 trace/span 折叠。
- **Chat 右栏**：移除终端标签，保留引用记忆与 Memory Runtime。移除与运行页上线放在同一次改动：先移除会让日志与事件在前端暂时没有任何视图，日志 WebSocket 客户端与日志 store 也会失去使用方。

### 3.6 非目标

- 不建事件存储、查询接口或回放到总线；
- 事件仍不参与业务判断，不替代领域返回值与 Store 状态；
- 日志不复写事件，事件不携带日志正文；
- 不为了分组把 trace 扩展到所有入口，分组靠第 3.1 节的关联字段。

## 4. 开放问题

当前没有阻塞计划的未决问题。第 3 节的默认值在计划评审时确认。

## 5. 升级为计划的条件

本文可以形成两份依次执行的计划：

1. **runtime event 体系**：第 3.1 至 3.4 节。观测治理属于版本之外的工作，可以随时开始。前提是事件目录与分组键已经冻结，并用一轮对话、一次导入和一次 MCP 写入在事件记录文件中核对时间线（`scripts/runtime_event_timeline.py`）。
2. **运行页**：第 3.5 节，属于前端 UI 优化与空入口接线方向，在第一份计划完成、新事件成为当前事实之后编写。

## 6. 相关文档

- [运行时事件与可观测性](../components/observability.md)：信封、总线语义、事件记录文件与生产端现状；
- [日志与运行时事件](../governance/observability/logs-and-events.md)：日志写法、W6 生产端收敛；
- [事件契约](../contracts/routes-and-events.md)：公开事件与原因码；
- [Chat 视图](../frontend/chat-view.md)：Workspace Vision 的当前面板。
