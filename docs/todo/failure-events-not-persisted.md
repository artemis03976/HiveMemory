---
title: 警告与失败事件没有持久的观察者
status: todo
owner: components
scope: warning-and-error-runtime-events-are-only-visible-in-the-frontend-live-panel
code_paths:
  - src/hivememory/components/events/bus.py
  - src/hivememory/components/events/publisher.py
  - src/hivememory/system/assembler.py
related_docs:
  - docs/governance/observability/logs-and-events.md
  - docs/components/observability.md
last_reviewed: 2026-11-03
---

# 警告与失败事件没有持久的观察者

## 问题与证据（2026-11-03）

RuntimeEvent 唯一的消费方是 SSE 路由，供前端 Workspace Vision 实时显示，页面关闭或刷新后什么也不留下。进程内缓冲 1000 条，被调度器心跳占满，约 2.8 小时就轮换一遍。日常以 plugin 模式使用时没有人打开这个面板：

- 2026-11-03 排查合并失败时，`work.failed` 事件本来带了异常类名，但取证时已经被冲掉；
- Gateway workflow 与查询分析的失败只发事件、不记日志，完全看不到。

详见[日志与运行时事件](../governance/observability/logs-and-events.md)第 1、3 节。

## 影响

失败只在有人恰好开着前端时可见，事后无法排查。

## 做法

owner 2026-11-03 决定：持久的观察者并入后端日志，不另建事件存储（治理主题 W1）。

**最简版本（默认）**

- `components/events` 新增一个包装 sink：先交给内层 sink，再把 `warning` 与 `error` 级事件写一行日志（logger `hivememory.runtime_events`，级别与 severity 一致），带事件类型、`reason`、`message`、`data.error_class` 与非空的关联标识；不写 `data` 的其他字段。日志写入失败不影响内层的发出。`scoped` 返回包住自己的 `ScopedRuntimeEventSink`。
- 组合根用它包住 `event_sink`（总线或 Null sink），之后的派生都经过它；RuntimeEvent 关闭时失败也照样写入日志。
- 不用订阅者：总线没有启停生命周期，订阅者的有界队列满时会丢事件。
- 不新增配置项。

**暂不处理**：同一失败在已自行记日志的位置会出现两行，去重属于治理主题 W5。

## 完成条件

- `warning` 与 `error` 级事件写入对应级别的日志，`info` 级不写，内层 sink 总是收到事件；经 `scoped` 派生的 sink 同样经过；
- 组合根装配后，RuntimeEvent 关闭时领域的失败事件仍写入日志；
- 测试：该 sink 的单元测试放在 `tests/unit/components/events/`，装配的代表用例放在 `tests/unit/system/test_assembler.py`；
- 更新[可观测性](../components/observability.md)第 2、5 节与治理主题第 3、4 节。
