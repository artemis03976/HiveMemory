---
title: 运行时事件没有持久的记录
status: todo
owner: components
scope: runtime-events-are-only-visible-in-the-frontend-live-panel-and-leave-no-record
code_paths:
  - src/hivememory/components/events/bus.py
  - src/hivememory/server/logging_setup.py
  - src/hivememory/config/app.py
  - scripts/hivememory-dev.sh
related_docs:
  - docs/governance/observability/logs-and-events.md
  - docs/components/observability.md
  - docs/system/configuration.md
last_reviewed: 2026-11-03
---

# 运行时事件没有持久的记录

## 问题与证据（2026-11-03）

RuntimeEvent 唯一的消费方是 SSE 路由，供前端 Workspace Vision 实时显示，页面关闭或刷新后什么也不留下。进程内缓冲 1000 条，被调度器心跳占满，约 2.8 小时就轮换一遍。日常以 plugin 模式使用时没有人打开这个面板：

- 2026-11-03 排查合并失败时，`work.failed` 事件本来带了异常类名，但取证时已经被冲掉；
- Gateway workflow 与查询分析的失败只发事件、不记日志，完全看不到。

详见[日志与运行时事件](../governance/observability/logs-and-events.md)第 1、3 节。

2026-11-03 起维护任务空转不再发事件（[归档 Todo](../archive/todo/maintenance-event-noise.md)），缓冲不再被心跳占满；事件仍只在进程内，页面关闭或进程重启后不留记录，本 Todo 的问题仍在。

## 影响

失败只在有人恰好开着前端时可见，事后无法排查；失败前后的状态转换（同时在跑哪些任务、各自走到哪一步）也无从复盘。

## 做法

owner 2026-11-03 决定：运行时事件写入单独的事件日志文件，与 `backend.log` 分开（治理主题 W1）。同日早些时候的决定是"并入后端日志、不另建事件存储"，改定的理由：

- 事件文件每行一个事件的 JSON，结构统一，可以直接筛选与分组；`backend.log` 是自由文本，混有第三方库的输出，写法也不统一；
- 记录全部事件而不只是失败，才能看到失败前后的经过；
- 事件信封本来就不带正文、参数与密钥，整条写入是安全的；
- 不受日志级别的影响，取证时直接复制文件。

它仍不是事件存储：只追加写入、按天切分，不提供查询接口，也不回放到事件总线。

**最简版本（默认）**

- **写入位置**：`RuntimeEventBus.emit` 补全序号、时间戳与 trace 之后，把事件的 JSON 交给专用 logger `hivememory.runtime_events`，一个事件一行。这个 logger 不向上传递，事件不进入 `backend.log` 与 WebSocket 日志广播；没有输出时不序列化。写入失败与总线其余部分一样只记 warning，不影响缓冲与订阅。
  - 不用订阅者：订阅者的有界队列满时会丢事件，还需要启停。
  - 不另包一层 sink：总线在 `emit` 里才补全序号与时间戳，包在总线外面拿不到。
- **文件输出**：新增配置项 `logging.event_file_path`，默认为空，不写文件。配置后由入口的日志配置装上按本地日期切分的文件输出，格式只有消息本身，切分方式与 `backend.log` 相同。`runtime_events.enabled` 关闭时没有总线，也就没有事件可写。
- **开发脚本**：`scripts/hivememory-dev.sh` 把它设为 `.hivememory/logs/runtime-events.jsonl`，与 `backend.log` 并列。
- **不复写**：`backend.log` 不复写事件。日志看原因（异常与堆栈），事件文件看经过（状态转换、原因码与关联标识），两边靠 `task_id`、`process_id` 等标识对应。
- **离线关联**：新增脚本 `scripts/runtime_event_timeline.py`，按一个关联标识（`task_id`、`process_id`、`topic_id`、`trace_id` 或 `agent_run_id`）从事件文件中筛出事件，按时间排成一次任务、进程或话题的时间线。

**暂不处理**：

- 进程内的关联跟踪：维护在途任务表，实时发现开始后一直没有终态的任务。离线复盘中确实出现悬挂的任务时再做。
- 保留天数：与 `backend.log` 一样全部保留。
- 前端刷新后回放历史事件。
- 复盘时发现缺关联标识的事件，属于治理主题 W6。

## 完成条件

- 配置事件文件后，每个经过总线的事件写成一行 JSON，带总线补全的序号与时间戳；`backend.log` 中没有这些行；没有配置时不写文件；
- 测试：
  - 总线写给专用 logger 的内容放在 `tests/unit/components/events/test_bus.py`；
  - 入口的文件输出与不进入根 logger 放在 `tests/unit/server/test_logging_setup.py`；
  - 时间线脚本是开发工具，仓库没有脚本测试的先例，以实际生成的事件文件手动验证；
- 更新[可观测性](../components/observability.md)第 2、5 节、[System 配置](../system/configuration.md)的日志配置与治理主题第 3、4、5 节。

