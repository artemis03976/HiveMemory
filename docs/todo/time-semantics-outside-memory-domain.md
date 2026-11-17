---
title: Memory 域之外的时间语义统一
status: todo
owner: project
scope: wall-clock-relative-time-naive-epoch-fields-and-scattered-time-sources-outside-the-memory-domain
code_paths:
  - src/hivememory/utils/time.py
  - src/hivememory/infrastructure/rate_limiter.py
  - src/hivememory/core/models/topic.py
  - src/hivememory/core/models/interaction.py
  - src/hivememory/engines/retrieval/
  - src/hivememory/server/app.py
  - src/hivememory/agent_runtime/mtp/runtime.py
  - src/hivememory/patchouli/runtime/core.py
  - src/hivememory/agent_runtime/tools/clock.py
related_docs:
  - docs/archive/plans/v0.7.0-time-semantics-and-controllable-clock.md
  - docs/architecture/data-model.md
last_reviewed: 2026-11-16
---

# Memory 域之外的时间语义统一

2026-11-16 由《全项目时间使用统一规范》Idea 转来（owner 决定），Idea 删除前的最后版本见提交 `08f2c354`。

## 问题

Memory 领域的时间语义已在 v0.7.0 统一（[归档计划](../archive/plans/v0.7.0-time-semantics-and-controllable-clock.md)）：持久化的业务时间一律 UTC-aware，取时经局部注入的 now，耗时用 monotonic 测量。Memory 域之外仍是统一之前的写法，该计划的附录 D 列了当时的清单。系统时钟回拨或 NTP 跳变时，用墙钟计算的相对时间会给出错误的等待、过期判定与负的耗时；新代码也无从判断某处能不能用本地时间。

## 证据（2026-11-16 核对代码）

- **墙钟当作相对时间**：[限流器](../../src/hivememory/infrastructure/rate_limiter.py)的令牌桶用 `time.time()` 计算补充与过期。
- **耗时用墙钟**：检索（`engines/retrieval/engine.py`、`retriever.py`）、Server（`server/app.py`）、MTP runtime 与 Patchouli runtime 中有 `time.time()`。
- **话题域的时间字段是 naive epoch 秒**：[`TopicData.last_update`](../../src/hivememory/core/models/topic.py) 与话题块的 `created_at`（`datetime.now().timestamp()`），`StreamMessage.timestamp` 同样如此；字段形态与业务时间分不开。
- **取时入口分散**：[`utils.time`](../../src/hivememory/utils/time.py) 自称生产代码取得业务时间的唯一入口，生产代码中仍有 20 处内联的 `datetime.now(UTC)`，另有 7 处 naive 的 `datetime.now()`。
- **展示域没有成文的约定**：`sys_clock` 以本地时区展示时间是合理的，但没有一份允许使用本地时间的清单。

## 可能的方向（未决定）

- monotonic 时钟沿用 `Callable[[], float]` 的注入方式，收口限流器、调度器、解析预算、Gateway deadline 与 Qdrant 等待；
- 话题域字段逐个判定是 UTC 业务时间还是 monotonic 运行状态，idle 一类的判定改为注入时钟；
- 耗时统计一律改为 monotonic 或 `perf_counter`；
- 内联的 `datetime.now(UTC)` 收口到 `utc_now()`；
- 允许使用本地时间的展示点写成清单，再由静态扫描守住：裸 `datetime.now()`、以 `time.time()` 计算耗时、清单之外的本地时区。

## 动手前要确认

- 话题状态是否需要跨重启恢复：要恢复的话，idle 语义必须改为 UTC 业务时间加重算，monotonic 不能持久化；
- 限流器的桶语义改用 monotonic 之后，外部可见的行为是否变化；
- `StreamMessage.timestamp` 删除还是迁到 UTC，这涉及公共模型；
- 前端是否消费数值形态的后端时间字段。

## 完成条件

- 上面各处改为 UTC 业务时间、注入的时钟或 monotonic，展示点写成清单并由静态检查守住；
- 或者某一部分扩展为跨系统的改动、升级为计划，届时在[归档索引](../archive/todo/README.md)登记去向。
