---
title: 全局运行页
status: planned
owner: frontend
target: 版本之外的工作（前端 UI 优化与空入口接线方向，桌面打包之前；owner 2026-11-07 决定 Terminal 入口为全局运行页）
scope: terminal-entry-run-view-run-grouping-timeline-backend-logs-and-chat-panel-removal
updates:
  - docs/frontend/README.md
  - docs/frontend/run-view.md
  - docs/frontend/chat-view.md
  - docs/frontend/state-and-transports.md
  - docs/frontend/application-shell.md
  - docs/contracts/routes-and-events.md
  - docs/components/observability.md
  - docs/ideas/runtime-event-system-and-run-view.md
  - docs/ideas/README.md
  - docs/ROADMAP.md
---

# 全局运行页

本计划实施[正式的 Runtime Event 体系与全局运行页 Idea](../ideas/runtime-event-system-and-run-view.md)第 3.5 节：导航里的 Terminal 入口渲染一个全局运行页，主视图把运行时事件按"一次运行"展示，第二个标签是后端日志；Chat 右栏的终端标签移除。事件体系与分组规则已经由 [Runtime Event 体系完善](../archive/plans/runtime-event-system.md)落地（[运行时事件与可观测性](../components/observability.md#11-一次运行)第 1.1 节），本计划只做页面与一处回放参数。

## 1. 背景和问题

- **入口空白**：导航的 Terminal 入口在 `App.tsx` 中没有渲染分支，选择后主区域为空。
- **观测挂在 Chat 页**：日志与事件只在 Chat 右栏的"终端"标签里。日常使用以 plugin 模式为主，导入、MCP 访问、生成任务与维护都不经过 Chat 页。
- **两路观测的展示与设计意图相反**：日志按 trace/span 折叠，而 trace 之外的日志都落在 `system/main` 一组；事件只有平铺列表，每条只标出一个关联标识。
- **刷新后只剩 100 条**：事件流在新连接时只回放最近 100 条（`RuntimeEventBus.subscribe` 的默认值），后端内存缓冲里的其余事件拿不到。

## 2. 目标与非目标

### 2.1 目标

1. Terminal 入口渲染运行页，主视图是"运行"标签：左侧运行列表，右侧所选运行的事件时间线。
2. 运行按[分组规则](../components/observability.md#11-一次运行)在前端派生，标题、状态、耗时与关联话题由事件计算，不另设后端接口。
3. 第二个标签是后端日志：平铺的日志尾部，可按级别、logger、文本与 trace 筛选，并能从运行详情跳到这次运行的日志。
4. 打开或刷新页面时取回后端缓冲内的全部事件（有限历史）。
5. Chat 右栏移除终端标签，日志视图不再按 trace/span 折叠，相关的折叠状态与 reducer 删除。

### 2.2 非目标

- 事件记录文件的查询与历史检索（owner 决定运行页只看有限历史）。
- 新增事件或修改分组规则。
- 同方向的其他空入口：Settings 未对齐的类、消息的复制与点赞与重试、`#` 话题引用、头像、Agent 状态，另写计划。
- 浅色主题的重新配色：运行页只使用现有的主题变量，两套主题都能显示。
- 在运行页上做控制操作（停止对话、取消生成任务、手动结算）：页面只看，不改。

## 3. 当前状态

- **页面**：`App.tsx` 渲染 chat、database、agents、settings 四个分支；`NavTab` 已有 `terminal`，`GlobalNavBar` 在底部显示它的图标。
- **终端标签**（`components/chat/WorkspaceTerminalTab.tsx`）：日志视图以 `groupLogsByTrace` 折叠成 `TraceGroup`/`SpanGroup`；事件视图以 `EventRow` 平铺；工具栏有连接、暂停、自动滚动与清空，日志有级别与文本筛选。
- **数据**：`useWorkspaceStore` 持有 `logs`、`runtimeEvents`（上限 `ui.maxBufferSize` = 1000）、两路连接状态与多窗口同步（主窗口持有连接，经 BroadcastChannel 同步给其他窗口）。`addRuntimeEvent` 先把事件交给 Memory Runtime 的任务投影，再在未暂停时追加到列表。
- **连接时机**：终端标签挂载时建立日志 WebSocket，Workspace Vision 挂载时建立事件流；Memory Runtime 标签依赖事件流。
- **回放**：`/runtime-events/stream` 新连接回放最近 100 条，带回放点时回放其后的全部；`/runtime-events/status` 返回 `buffer_size`，前端连接前已经请求它。

## 4. 目标设计

### 4.1 owner 决定（2026-11-07）

1. Terminal 入口是全局运行页，主视图是运行时事件，后端日志作为第二个标签。
2. 运行页只看有限的事件历史，事件记录文件不提供查询接口。
3. Chat 右栏的终端标签移除。
4. 运行页的 UI 与交互细节由 agent 设计。

以下第 4.2 至 4.9 节都是这一授权下的候选设计，可在计划评审时调整。

### 4.2 页面结构

```text
┌────┬──────────────────────────────────────────────────────────────────────────┐
│ 导 │ 运行    [运行] [后端日志]          ● 事件流已连接 · 缓冲 412/1000 · 最早 16:02   ⏸ 🗑 │
│ 航 ├──────────────────────────┬───────────────────────────────────────────────┤
│    │ [全部][对话][导入][MCP]  │ MCP · memory_update            ✓ 完成  1.4s   │
│    │ [生成][维护][系统]       │ mcp-2ddbbcf9 · claude_code · main_workspace   │
│    │ ☐ 只看进行中  ☐ 只看异常 │ 关联：话题 a9164c89 → 生成任务 active:…        │
│    │ 🔍 按标识或话题筛选       ├───────────────────────────────────────────────┤
│    ├──────────────────────────┤ +0ms     Gateway 开始分析（被动记忆）           │
│    │▌MCP · memory_update  ✓   │ +1.36s   Gateway 完成：路由到新话题            │
│    │ 16:11:44 · 1.4s · 9 条    │          ▇▇▇▇▇▇▇▇▇▇▇ 话题路由 1.36s            │
│    │▌导入 · claude-code   ✓   │ +1.37s   建立话题（为写入预先确定）            │
│    │ 16:10:02 · 2.1s · 9 条    │ +1.38s   生成任务创建（更新）                  │
│    │▌生成 · 更新 rev_wip… ⟳   │ +1.39s   更新已接纳 rev_wip_status_…           │
│    │ …                         │                    [查看这次运行的日志 →]       │
└────┴──────────────────────────┴───────────────────────────────────────────────┘
```

- **顶栏**：页面标题"运行"、两个标签、事件流状态与缓冲占用（"缓冲 412/1000 · 最早 16:02"），暂停与清空。清空只清本页显示，不影响后端缓冲。
- **左栏**（固定宽度约 380px）：运行列表与筛选，最新的在上。
- **右栏**：所选运行的详情。没有选中时显示概览：最近一小时各类运行的次数、失败与推迟的运行。
- 窄屏（宽度小于 900px）时左右两栏改为上下切换：选中运行后显示详情，带返回按钮。

### 4.3 运行的归组与派生

纯函数 `groupRuns(events)` 把事件列表派生为运行列表，规则与[分组规则](../components/observability.md#11-一次运行)一致，并加两处只影响展示的挂接：

| 运行种类 | 归组键 | 标题 | 终态 |
|:---|:---|:---|:---|
| 对话 | `task-` 开头的 `trace_id` | `对话 · {agent_id}` | `chat.run.completed/cancelled/failed` |
| 导入 | `import-` 开头的 `trace_id` | `导入 · {data.source}`，副标题为外部轮次标识 | `import.turn.*` |
| MCP | `mcp-` 开头的 `trace_id` | `MCP · {span_name}`（工具名） | `actor.operation` |
| 生成 | `memory.task.*` 的 `task_id` | `生成 · {来源}`，副标题为 `label` 或待生成记忆的别名 | `memory.task.completed/cancelled/failed` |
| 维护 | 单条 `maintenance.task.*` | `维护 · {data.name}` | 自身 |
| 话题结算 | 不在运行内的单条 `topic.settled`/`topic.deleted` | `话题结算 · {原因}` | 自身 |
| 系统 | 相邻的 `system.*` 合成一次启动或关闭 | `系统启动` / `系统关闭` | `ready`、`start_failed`、`stopped`、`stop_failed` |

- **work.\* 的挂接**：work id 为 `memory_generation:{task_id}` 的挂到对应的生成运行；`interaction:{interaction_id}` 的挂到交互标识相同的导入（`data.interaction_id`）或对话（`process_id`）；`topic_compaction:` 开头的合成一次"话题折叠"运行。挂不上的 work 事件自成一次运行。
- **状态**：有终态事件时取终态（完成、失败、取消、推迟、重复、跳过）；没有终态时为"进行中"。超过 10 分钟仍没有终态的显示为"未见终态"，不推断为失败：事件是 best-effort，可能丢失或超出缓冲。
- **关联**：运行中出现的 `topic_id` 列为关联话题。详情中点击话题，列表筛出所有带这个话题的运行，例如导入、之后的结算与结算引起的生成。
- **性能**：只在事件列表变化时重算（selector 记忆化）；1000 条事件的分组是一次线性扫描。

### 4.4 运行列表

- **筛选**：种类（全部、对话、导入、MCP、生成、维护、话题、系统）单选；"只看进行中""只看异常"两个开关，异常指失败、推迟与未见终态；文本框按 trace、task、话题或 agent 的标识子串筛选。筛选条件存入 localStorage，属于视图偏好。
- **行**：左侧是种类图标；第一行是标题与状态徽标，第二行是开始时间（今天只显示时分秒，悬停显示完整时间）、耗时与事件数。进行中的显示脉动点，失败行整体淡红底色。
- **实时**：新运行出现在顶部；已选中的运行保持选中，不跳动。列表滚离顶部时不自动滚动，顶部出现"N 个新运行"的提示，点击回到顶部。
- **键盘**：上下方向键移动选择，Enter 展开详情（窄屏），Esc 取消选择。

### 4.5 运行详情与时间线

- **头部**：标题、状态、总耗时；第二行是 trace 或 task 标识（点击复制）、agent、Workspace；第三行是关联话题与挂接的生成任务，可点击跳转。
- **时间线**：按事件顺序纵向排列，左侧是相对首个事件的时间（`+0ms`、`+1.36s`），右侧是一句中文描述与关键字段；点击一行展开原始 JSON。失败与拒绝的行用警示色，并显示原因码与 `error_class`。
- **描述**：每个事件类型对应一句描述，由事件字段拼出，不显示正文：

| 事件 | 描述 |
|:---|:---|
| `chat.run.created` / `status` / `completed` / `cancelled` / `failed` | 对话开始 / 进入{阶段} / 对话完成 / 已停止（{原因}） / 对话失败：{reason} |
| `agent.run.*` | Agent 开始执行 / 完成（MTP {n} 轮） / 停止 / 失败：{reason} |
| `agent.context.compaction` / `budget_exceeded` | 上下文压缩 {before}→{after} tokens / 超出上下文预算 |
| `gateway.workflow.started` / `completed` / `failed` | Gateway 开始分析（{入口}） / Gateway 完成：{路由到新话题或已有话题、意图、是否检索} / Gateway 失败：{reason} |
| `gateway.analysis.capability.completed` | 查询分析完成 / 查询分析失败 |
| `actor.operation` | 检索 {n} 条 / 读取 {delivered}/{requested} / RUN {resolution} / 写入已接纳 {pending_alias} / 更新已接纳 {pending_alias} / CALL {callee}；被拒时为"{操作}被拒：{reason}" |
| `topic.created` / `settled` / `deleted` | 建立话题（应用交互时 / 为写入预先确定） / 话题结算（{原因}，{是否提交生成}） / 话题删除 |
| `import.turn.*` | 导入接纳（新话题 / 已有话题） / 重复，已忽略 / 没有输出，跳过 / 推迟：{reason} |
| `memory.task.*` | 生成任务创建（{来源}） / 开始生成 / 生成完成 → {canonical_alias} / 已取消 / 生成失败：{reason} |
| `work.*` | 入队 / 开始执行 / 重试排队 / 成功 / 失败 / 进入死信 / 取消 / 拒绝：{reason} |
| `maintenance.task.*` | 维护完成（{结果摘要}） / 维护失败 |
| `system.*`、`subsystem.operation.*` | 启动、就绪、关闭等阶段；关闭前 drain 完成（{结果}） |

- **Gateway 的步骤**：`gateway.workflow.completed` 的 `step_durations_ms` 画成一组水平条，最长的那一步（通常是话题路由）高亮；`fallbacks` 非空时在条旁标出降级的步骤与原因。
- **跳到日志**：带 trace 的运行显示"查看这次运行的日志"，切到日志标签并以这个 trace 筛选。生成与维护运行没有自己的 trace，不显示这个按钮。

### 4.6 后端日志标签

- 平铺的日志尾部，等宽字体，每行是时间、级别徽标、logger（缩写到最后两段，悬停显示全名）、`[trace]` 小标签与消息。
- 筛选：级别、logger 前缀、文本与 trace；trace 筛选以一个可移除的标签显示，从运行详情跳来时自动设置。
- 自动滚动与"N 条新日志"的提示同运行列表。原有的 trace/span 折叠删除。

### 4.7 连接、历史与缓冲

- **回放全部缓冲**：`/runtime-events/stream` 增加查询参数 `replay_last`（整数，取值范围 1 到缓冲大小，超出时截到缓冲大小）。不带回放点的新连接按它回放，缺省仍为 100。运行页连接时传入 `/status` 返回的 `buffer_size`，刷新后能看到后端缓冲里的全部事件。带回放点的续传不受影响。
- **前端上限**：`maxBufferSize` 保持 1000，与后端缓冲一致；超出时丢掉最旧的事件。顶栏显示占用与最早一条事件的时间，让人知道历史有多长。
- **断档**：收到 `event.stream.gap` 时，列表顶部显示一条提示："事件流曾中断，{时间}之前的部分事件已经不在缓冲中"。运行的状态照常显示，不因断档改写。
- **暂停**：暂停时不再追加事件，Memory Runtime 的任务投影照常更新（现有行为）；顶栏显示"已暂停"。
- **连接时机**：运行页挂载时建立事件流与日志 WebSocket；离开运行页不断开，其他页面的 Memory Runtime 仍需要事件流。多窗口沿用现有的主窗口同步。
- **未启用**：`runtime_events.enabled` 为 false 时，运行标签显示"运行时事件未启用"与配置项名称；日志标签照常可用。

### 4.8 Chat 右栏

Workspace Vision 移除终端标签，保留"引用记忆"与"记忆任务"两个标签，面板宽度按剩下的两种标签计算。持久化的 `workspaceVisionTab` 为 `terminal` 时迁移为 `context`，不让旧标签在升级后复活。`WorkspaceTerminalTab`、`groupLogsByTrace` 等折叠 reducer 与 `TraceGroup`/`SpanGroup` 类型随之删除。

### 4.9 视觉

- 沿用现有主题变量与组件写法（`glass-panel`、`ghost-border`、Manrope 与 Inter 字体、`font-mono` 的日志区）。
- **颜色只表示状态**，沿用七曜状态色在生成任务面板、MTP 卡片、Toast 与连接点上已有的对应：进行中 `magic-water` 脉动，完成 `magic-wood`，失败 `magic-fire`，推迟、拒绝与未见终态 `magic-metal`，取消、跳过与重复 `slate`。日志级别同样：INFO `magic-water`，WARNING `magic-metal`，ERROR `magic-fire`。
- **种类不用颜色**，由中性色的图标与标题前缀辨认，图标取 lucide-react：对话 `MessageSquare`，导入 `ArrowDownToLine`，MCP `Plug`，生成 `Sparkles`，维护 `Wrench`，话题 `Hash`，系统 `Power`，用在列表行、种类筛选、详情的种类徽标与概览中。七曜色在前端各处已经表示状态，分给种类会让同一行出现两套颜色含义（生成的绿像完成，话题的黄像推迟），异常也不再醒目。
- `primary` 只用于选中、焦点、时间线上的关键事件与 Gateway 最慢一步的耗时条，不表示状态。行底色保持中性，失败行除外。

## 5. 受影响的系统、契约与数据

- **frontend**：新页面 `components/runs/`（运行页、列表、详情、日志标签）；`App.tsx` 增加 terminal 分支；派生函数放在 `stores/runs/`（`groupRuns` 与事件描述）；`stores/workspace` 删除 trace 折叠、增加 trace 筛选与 `replay_last`；`WorkspaceVision` 删去终端标签；`appUiStore` 的持久化迁移。
- **server**：`/runtime-events/stream` 的 `replay_last` 参数。
- **契约**：[事件契约](../contracts/routes-and-events.md)写明 `replay_last`；事件本身不变。
- **数据**：没有持久化数据变化；localStorage 新增运行页的筛选偏好。

## 6. 兼容与迁移

- `replay_last` 缺省为 100，旧前端与其他订阅方不受影响。
- 持久化的 `workspaceVisionTab: "terminal"` 迁移为 `context`；workspace store 持久化的日志筛选保留，折叠状态本来不持久化。
- 删除的折叠 reducer 与类型没有其他使用方。

## 7. 测试和验收标准

| 不变量 | 归属层 |
|:---|:---|
| 分组规则：按 trace 归组的三种运行、按 task 归组的生成运行、work 的挂接与挂不上时自成运行、系统事件合成一次启动、状态取终态、没有终态时为进行中或未见终态 | `frontend/tests/` 新增 `groupRuns` 的用例 |
| 事件描述不含正文，只由字段拼出；未知类型回退为类型名 | 同一测试文件 |
| 持久化的 `terminal` 标签迁移为 `context` | `frontend/tests/` 中 UI store 的用例 |
| `replay_last` 回放指定条数，超出缓冲时截到缓冲大小，带回放点时不受影响 | `tests/unit/server/routers/test_runtime_events.py` |

- 前端组件以 lint 与构建守护，分组与描述的逻辑都在纯函数里，由上表的测试覆盖。
- 实际使用核对：在运行页上看一轮对话、一次导入、一次 MCP 写入与一次话题结算，刷新后仍在；从运行详情跳到日志。核对结果记入实施记录，不阻止归档。

验证命令：

```bash
cd frontend && npm run lint && npm test && npm run build
pytest tests/unit/server/routers/test_runtime_events.py -q
pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto -q
ruff check . && black --check .
```

## 8. 风险与待决问题

- **事件丢失时的状态**：best-effort 的事件可能缺终态。缓解："未见终态"只陈述事实，不推断失败。
- **缓冲覆盖的时间不定**：现在每天 300 至 700 条事件，1000 条约覆盖一天，事件多时更短。顶栏显示最早一条的时间，让人知道历史有多长。
- **work id 的格式**：挂接依赖 work id 的前缀（`memory_generation:`、`interaction:`、`topic_compaction:`），它们由 Patchouli 生成，属于现有事件的 `task_id` 取值。前缀改变时挂接失效，运行退化为单独显示，不会出错。
- **待决**：没有阻塞问题。可在评审时调整的有：两栏宽度与窄屏断点、"未见终态"的 10 分钟阈值、概览的内容。

## 9. 完成时必须更新的当前文档

- 新增运行页的当前事实文档 `docs/frontend/run-view.md`：页面结构、运行的派生、时间线描述、日志标签、连接与缓冲。
- [前端索引](../frontend/README.md)：Terminal 入口不再是空白页。
- [Chat 视图](../frontend/chat-view.md)：第 5.3 节删去，Workspace Vision 只剩两个标签。
- [应用外壳](../frontend/application-shell.md)：导航的五个页面。
- [状态与传输](../frontend/state-and-transports.md)：事件流的 `replay_last` 与运行页的连接时机。
- [事件契约](../contracts/routes-and-events.md)：`replay_last` 参数。
- [运行时事件与可观测性](../components/observability.md)：消费方改为运行页。
- Idea 与 Ideas 索引：第 3.5 节实施后，整份 Idea 归档。
- [ROADMAP](../ROADMAP.md) 的前端一节。
