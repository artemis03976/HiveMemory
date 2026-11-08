---
title: Frontend Run View
status: current
owner: frontend
scope: global-run-view-run-grouping-timeline-and-backend-logs
code_paths:
  - frontend/src/components/runs/
  - frontend/src/stores/runs/
  - frontend/src/stores/workspace/
  - src/hivememory/server/routers/runtime_events.py
related_contracts:
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/components/observability.md
last_reviewed: 2026-11-07
---

# 全局运行页

导航底部的 Terminal 入口打开全局运行页。日常使用以 plugin 模式为主，导入、MCP 访问、生成任务与维护都不经过 Chat 页，所以观测入口放在全局，而不是挂在某个会话旁边。页面的主视图是运行时事件：一次对话、一轮导入或一次 MCP 工具调用各自显示为"一次运行"，选中后看它的事件时间线；第二个标签是后端日志。页面只看，不提供停止对话、取消任务或手动结算等控制操作。

运行时事件是 best-effort 的观测，不是业务账本（[运行时事件与可观测性](../components/observability.md)）。运行页显示的状态只陈述收到了哪些事件，事件缺失时不推断失败。

## 1. 页面结构

- **顶栏**：标题、"运行"与"后端日志"两个标签、当前标签对应的连接状态；"运行"标签另显示缓冲占用（"缓冲 412 / 1000"）与最早一条事件的时间，让人知道历史有多长。暂停后不再追加日志与事件，Memory Runtime 的任务投影照常更新；清空只清本页显示，不影响后端缓冲。
- **左栏**（380px）：运行列表与筛选，最新开始的在上。
- **右栏**：所选运行的详情。没有选中时显示概览：最近一小时各种类运行的次数，以及缓冲内失败、推迟、拒绝与未见终态的运行。
- 视口窄于 900px 时两栏改为切换：选中运行后只显示详情，带"返回列表"按钮。

## 2. 运行的派生

`stores/runs/groupRuns.ts` 的 `groupRuns(events, now)` 是纯函数，把事件列表派生为运行列表，不另设后端接口。归组规则与[一次运行](../components/observability.md#11-一次运行)一致，并加上几处只影响展示的合并：

| 种类 | 归组 | 标题 | 状态取自 |
|:---|:---|:---|:---|
| 对话 | `task-` 开头的 trace | `对话 · {agent}` | `chat.run.completed/cancelled/failed` |
| 导入 | `import-` 开头的 trace | `导入 · {来源 harness}`，副标题为外部轮次 | `import.turn.*`：接纳为完成，另有重复、跳过与推迟 |
| MCP | `mcp-` 开头的 trace | `MCP · {工具名}` | `actor.operation` 的完成、拒绝或失败 |
| 生成 | `memory.task.*` 的 `task_id` | `生成 · {写入/更新/结算/合并}`，副标题为待生成记忆的别名 | `memory.task.completed/cancelled/failed` |
| 维护 | 单条 `maintenance.task.*` | `维护 · {任务名}` | 自身 |
| 话题 | 不在运行内的单条 `topic.*`；话题折叠的 work | `话题结算 · {原因}`、`话题建立`、`话题删除`、`话题折叠` | 自身 |
| 系统 | 相邻的 `system.*` 及其间的 `subsystem.operation.*`；挂不上的 work | `系统启动`、`系统关闭`、`队列任务 · {lane}` | `ready`、`start_failed`、`stopped`、`stop_failed`；work 的终态 |

- **work 的挂接**：`work.*` 不单独归组。work id 为 `memory_generation:{task_id}` 的挂到同一 task 的生成运行；`interaction:{interaction_id}` 的挂到交互标识相同的导入（`data.interaction_id`），或进程标识相同的对话（对话的交互标识就是进程标识）；`topic_compaction:` 开头的各自合成一次话题折叠。挂不上的（例如它所属的运行已超出缓冲）自成一次系统运行，状态取 work 的终态。work id 的前缀改变时挂接失效，运行退化为单独显示，不会出错。
- **状态**：有终态事件时取终态；没有终态时为"进行中"，开始超过 10 分钟仍没有终态的显示为"未见终态"。不属于任何运行的单条事件是一次瞬时观测，没有进行中可言，显示为完成。页面每 15 秒按当前时间重算一次，进行中的耗时与未见终态随之更新。
- **耗时**：多条事件的运行取首尾事件的间隔，与终态事件自带的 `duration_ms` 取较大者；只有一条事件的运行只取它自带的 `duration_ms`，没有时显示"—"。
- **关联**：运行中出现的 `topic_id`（包括 Gateway 终态事件 `data.topic_id` 中路由到的话题）列为关联话题。生成任务事件沿用提交者的 trace，因此一次 MCP 写入或对话与它提交的生成运行互相链接：详情中显示"来自"与"生成任务"。
- **筛选**：种类（全部、对话、导入、MCP、生成、维护、话题、系统）单选；"只看进行中"与"只看异常"两个开关，异常指失败、推迟、拒绝与未见终态；文本框按标题、副标题、trace、task、话题与 agent 的子串匹配。筛选条件随 `chat-ui-store` 持久化（[状态与传输](./state-and-transports.md#2-本地持久化矩阵)），属于视图偏好。

## 3. 运行列表

每行左侧是种类图标，第一行是标题与状态，第二行是副标题，第三行是开始时间（今天只显示时分秒，悬停显示完整时间）、耗时与事件数。进行中的状态点脉动，失败行带淡红底色。

新运行出现在顶部，已选中的运行保持选中。列表滚离顶部后不再跟随，顶部出现"N 个新运行"的提示，点击回到顶部。列表获得焦点时，上下方向键移动选择，Esc 取消选择。

## 4. 运行详情与时间线

- **头部**：种类、标题、状态（非完成的结局带原因码）与总耗时；第二行是 trace 或 task 标识（点击复制）、agent、Workspace 与开始时间；第三行是关联话题、来源运行与生成任务，点击话题把列表筛选为带这个话题的运行，点击运行则选中它。
- **跳到日志**：带 trace 的运行（对话、导入、MCP）显示"查看这次运行的日志"，切到日志标签并以这个 trace 筛选。生成、维护与系统运行没有自己的 trace，不显示这个按钮。
- **时间线**：按事件顺序纵向排列，左侧是相对首个事件的时间（`+0ms`、`+1.36s`），右侧是一句中文描述、关键字段与事件类型；点击一行展开原始 JSON。
- **描述**：`stores/runs/describeEvent.ts` 为每个事件类型拼一句描述，只用事件的字段（原因码、计数、别名、阶段等），不显示事件的 `message`；未知类型回退为类型名。
- **Gateway 的步骤**：`gateway.workflow.completed` 与 `failed` 的 `step_durations_ms` 画成一组水平条，最长的一步高亮；`fallbacks` 非空时在条下标出降级的步骤与原因。

颜色只表示状态，沿用七曜状态色在前端已有的对应：进行中为水，完成为木，失败为火，推迟、拒绝与未见终态为金，取消、跳过与重复为 slate。种类不用颜色，由中性色的图标（`MessageSquare`、`ArrowDownToLine`、`Plug`、`Sparkles`、`Wrench`、`Hash`、`Power`）与标题前缀辨认：七曜色在其他页面已经表示状态，分给种类会让同一行出现两套颜色含义，异常也不再醒目。`primary` 只用于页面主标题（与其他页面一致）、选中、焦点、时间线上运行的起点与 Gateway 最慢一步的耗时条。

## 5. 后端日志

平铺的日志尾部，等宽字体，每行是时间、级别、logger（缩写到最后两段，悬停显示全名）、`[trace/span]` 与消息，带异常时在下面列出异常类型与消息。筛选有级别、logger 前缀、文本与 trace：前三项是 `workspace-store` 中持久化的日志筛选，trace 只在本页有效，以可移除的标签显示，从运行详情跳来时自动设置。

在底部时跟随新日志；滚离底部后不再跟随，底部出现"N 条新日志"，点击回到底部。日志不按 trace/span 折叠。日志 WebSocket 只推送连接之后的日志，没有历史。

## 6. 连接、历史与缓冲

- **连接时机**：运行页挂载时建立日志 WebSocket 与事件流；Chat 页的 Workspace Vision 挂载时也建立事件流。离开运行页不断开，Memory Runtime 仍需要事件流。多窗口沿用主窗口持有连接、经 BroadcastChannel 同步的机制（[状态与传输](./state-and-transports.md#5-多窗口日志同步)）。
- **有限历史**：前端连接前请求 `/runtime-events/status`，以返回的 `buffer_size` 作为 `/runtime-events/stream` 的 `replay_last`，所以打开或刷新页面时取回后端缓冲内的全部事件。前端上限 `maxBufferSize` 为 1000，与后端缓冲一致，超出时丢掉最旧的事件。历史只到后端缓冲为止，事件记录文件不提供查询接口。
- **断档**：续传时后端找不到回放点会先发 `event.stream.gap`，列表上方提示"事件流曾中断，{时间} 之前的部分事件已经不在缓冲中"；运行的状态照常显示，不因断档改写。
- **未启用**：`runtime_events.enabled` 为 false 时，"运行"标签显示"运行时事件未启用"与配置项名称，日志标签照常可用。事件流连接失败时显示提示与重试按钮。

## 7. 验证入口

分组、挂接、状态与描述的逻辑都在纯函数中，由 `frontend/tests/run-grouping.test.mjs` 覆盖；组件以 lint 与构建守护。`replay_last` 的回放行为由 `tests/unit/server/routers/test_runtime_events.py` 覆盖。
