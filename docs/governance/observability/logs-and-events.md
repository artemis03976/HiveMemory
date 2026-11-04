---
title: Logs and Runtime Events
status: governance
owner: components
scope: logging-conventions-runtime-event-failure-reporting-and-convergence
code_paths:
  - src/hivememory/server/logging_setup.py
  - src/hivememory/components/events/
  - src/hivememory/components/scheduler/async_scheduler.py
  - src/hivememory/core/contracts/runtime_events.py
related_docs:
  - docs/components/observability.md
  - docs/contracts/routes-and-events.md
  - docs/system/configuration.md
last_reviewed: 2026-11-03
---

# 日志与运行时事件

本文是跨版本的观测治理主题：日志怎样写，失败怎样被看见，以及现状与这两点的差距。RuntimeEvent 的机制、信封与 best-effort 语义见[运行时事件与可观测性](../../components/observability.md)，失败事件的公开原因码见[事件契约](../../contracts/routes-and-events.md#44-失败与降级观测)第 4.4 节，日志配置段见 [System 配置](../../system/configuration.md)。本文不重复这些内容。

owner 2026-11-03 决定把观测与日志作为版本之外的工作，随时可以开始。运行时事件的持久记录是单独的事件日志文件，与后端日志分开：同日早些时候决定并入后端日志，评估结构化查询与完整经过的需要后改定；它只追加写入、按天切分，不是事件存储。

## 1. 为什么需要

日常使用以 plugin 模式为主（Claude Code、dsh），没有人打开前端的 Workspace Vision。2026-11-03 排查 4 次合并失败时，两条通道都没有给出原因（[收敛 Idea](../../ideas/generation-source-convergence.md#25-合并失败)第 2.5 节）：

- **事件**：`work.failed` 带了异常类名，但事件流唯一的消费方是前端的实时面板，刷新即丢失。进程内缓冲 1000 条被调度器心跳占满（话题空闲扫描每 30 秒、模型卸载检查每 60 秒各一对 started/completed），约 2.8 小时就轮换完，取证时已经查不到。
- **日志**：生成任务失败只有一句通用文本，合并中止的原因只在异常的 `details` 中，没有被任何一条通道带出。

## 2. 当前最低约束

### 2.1 日志与事件的分工

- 日志记录异常与调试上下文，事件记录跨组件的状态转换，供关联与流式消费；两者都不是业务事实（[可观测性](../../components/observability.md#5-健康与日志的关系)第 5 节）。
- 排查时日志看原因，事件记录文件看经过，两边靠领域标识对应（[可观测性](../../components/observability.md#51-事件记录文件)第 5.1 节）；日志不复写事件。
- 事件只携带稳定原因码、可公开的说明与摘要化数据，异常类名放在 `data.error_class`；原始异常正文、消息、工具参数与结果、记忆正文、密钥和绝对路径不进事件。需要原始 cause 时看日志。

### 2.2 日志写法

适用于新写与改动的代码；存量的收敛见第 4 节 W5。

- **格式化**：使用 `%` 占位符，`logger.info("…: topic_id=%s", topic_id)`，不使用 f-string。参数在级别过滤之后才格式化，全仓库只有一种写法。
- **语言**：沿用所在模块已有的语言，同一模块不混用。
- **级别**：
  - `error`：需要人处理的失败：任务失败、未预期的异常、启动或关闭失败；
  - `warning`：降级、可恢复的失败、被拒绝的请求；
  - `info`：一次性的生命周期与有意义的状态转换，不记录每个调度周期的空转；
  - `debug`：排查用的细节。
- **关联字段**：失败、降级与状态转换的日志带上适用的领域标识，写成 `key=value`：`task_id`、`topic_id`、`interaction_id`、`process_id`、`memory_id`、`intent_id`。只写标识，不写内容。
- **异常**：未预期的异常带堆栈（`logger.exception` 或 `exc_info=True`）；领域错误（`DomainError`）记录 `code` 与 `details`，异常文本只有错误码，原因在 `details` 中。
- **内容**：任何级别都不记录用户消息、记忆正文、工具参数与结果、提示词和密钥；需要时只记录长度或标识。
- **一次失败记一次**：在做出失败判断的那一层记录，向上传播时不重复记录。

## 3. 成熟度与已知缺口（2026-11-03）

| 方面 | 现状 |
|:---|:---|
| 日志配置 | 按天切分、只替换自己装上的输出、uvicorn 与 `warnings` 并入根 logger，已经成熟（`server/logging_setup.py`）。默认格式 `%(asctime)s - %(name)s - %(levelname)s - %(message)s` 不含 `TraceInjectFilter` 已经注入的 `trace_id` 与 `span_name` |
| 日志写法 | 约 300 处日志调用，约 120 处 f-string、约 89 处 `%` 格式；中文约 190 处、英文约 111 处；关联标识只在部分消息中随手写出 |
| 事件的消费与记录 | 前端实时面板，以及配置后单独写入的事件记录文件；开发脚本写到 `.hivememory/logs/runtime-events.jsonl`，`scripts/runtime_event_timeline.py` 按关联标识排时间线 |
| 失败事件的原因码 | 失败事件都有稳定原因码，异常类名在 `data.error_class`（[事件契约](../../contracts/routes-and-events.md#44-失败与降级观测)第 4.4 节）；生成任务的原因码由失败分类按异常给出（2026-11-03） |
| 只有事件没有日志 | Gateway workflow 与查询分析的失败只发事件，不记日志 |
| 内容进日志 | Gateway L1 拦截器在 debug 级别记录原始用户消息 |

## 4. 工作包

| 编号 | 工作包 | 状态 |
|:---|:---|:---|
| W1 | 运行时事件写入单独的事件日志文件，配离线时间线脚本 | 已完成（2026-11-03，[归档 Todo](../../archive/todo/runtime-events-not-persisted.md)） |
| W2 | 维护任务空转不发事件 | 已完成（2026-11-03，[归档 Todo](../../archive/todo/maintenance-event-noise.md)） |
| W3 | 失败事件带稳定原因码（生成任务之外） | 已完成（2026-11-03，[归档 Todo](../../archive/todo/failure-event-reason-codes.md)）；生成任务的原因码随 v0.7.3 于同日完成（[归档计划](../../archive/plans/v0.7.3-generation-failure-reason-codes.md)），见[收敛 Idea](../../ideas/generation-source-convergence.md) 第 4.3 节 |
| W4 | 默认日志格式带 `trace_id` 与 `span_name` | 未排期 |
| W5 | 存量日志按领域收敛到第 2.2 节的写法 | 未排期；先处理 Gateway 拦截器记录原始用户消息、调度器的 f-string、任务进程中不带 `process_id` 的失败日志 |
| W6 | RuntimeEvent 生产端收敛 | 未排期：Gateway、Scheduler 与 System lifecycle 仍各自组装信封，按领域迁移到窄领域 emitter 或 scoped publisher，迁移不得改变 wire format；由原 RuntimeEvent 生产端迁移 Todo 转入（2026-11-03） |

W1～W3 已于 2026-11-03 完成。W4～W6 在动到对应领域、或 owner 排期时登记为 Todo 或计划。

## 5. 长期目标与非目标

- **目标**：排查一次失败只需要后端日志与事件文件，日志看原因（异常与堆栈），事件文件看经过（状态转换、原因码与关联标识）；日志与事件对同一失败给出同一个原因码；新代码按第 2.2 节写，存量逐域收敛。
- **非目标**：事件文件不提供查询接口，也不回放到事件总线；不建事件存储、审计日志或跨进程的事件回放；事件仍是 best-effort 的旁路，不参与业务判断。
