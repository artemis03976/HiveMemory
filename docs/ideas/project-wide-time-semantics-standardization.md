---
title: Project-Wide Time Semantics Standardization
status: idea
owner: project
scope: whole-project-time-domain-unification-monotonic-and-utc
related_docs:
  - docs/archive/plans/v0.7.0-time-semantics-and-controllable-clock.md
  - docs/archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md
  - docs/architecture/data-model.md
last_reviewed: 2026-09-24
---

# 全项目时间使用统一规范（设想）

## 1. 背景与已有基础

v0.7.0 A2-P 已把 **Memory 领域**的时间语义统一完成并归档（见归档 Plan [A2-P Memory Time Semantics](../archive/plans/v0.7.0-time-semantics-and-controllable-clock.md)）：

- 持久化业务时间一律 UTC-aware，naive 值在模型与 codec 边界 fail closed；
- 四个 Memory 时间字段（`created_at`/`updated_at`/`lifecycle.last_accessed_at`/`lifecycle.decay_anchor_at`）职责互不替代，衰减唯一基准是 `decay_anchor_at`；
- 可控取时采用局部 `now: Callable[[], datetime]` 注入（默认 `utils.time.utc_now`），不引入全局时钟；同一业务操作只取一次 now；
- `TimeFormatter` 是纯展示组件，只消费 UTC-aware 输入；
- 旧存量已于 2026-09-23 按 `--source-tz` 显式迁移到 schema `"2.1"`，兼容入口已移除。

Memory 域之外的时间使用仍是迁移前的旧状态，集中记录在 A2-P 归档 Plan 附录 D（未迁移时间调用清单）。本 idea 设想把它们按同一套语义分类收口，但不承诺任何时间表。

## 2. 观察到的问题

1. **墙钟被用作相对时间**：`infrastructure/rate_limiter.py` 用 `time.time()` 做滑动窗口 TTL、`system/services/passive/turn_buffer.py` 用 naive epoch 判 idle——系统时钟回拨或 NTP 跳变会直接产生错误的等待/过期判定。
2. **topic 域时间字段语义未分类**：`LogicalBlock.created_at`、`TopicData.last_update`、`TopicAssetBinding.bound_at` 与 perception 投影仍是 naive epoch 秒；它们是进程内状态（重启即失），但字段形态与业务时间无法区分。
3. **耗时统计用墙钟**：检索、Server 中间件、MTP runtime、Patchouli runtime 的 `time.time()` 差值在时钟跳变时产生负耗时。
4. **时间来源入口分散**：`utils.time.utc_now` 自称"唯一生产时间入口"，但仍有 16 处内联 `datetime.now(UTC)`；monotonic 侧 `time.monotonic`/`perf_counter` 以模块级直接引用散布在 scheduler、workspace access、ParseBudget 与 qdrant client。
5. **展示时间未固化**：`sys_clock`（MTP 展示）与 `log_handler`（日志本地时区）是合理展示域，但没有成文的 allowlist 约定，新代码无法判断"这里能不能用本地时间"。

## 3. 可能路径（非承诺顺序）

1. **monotonic 域统一**：定义 monotonic 时钟的注入约定（沿用 `Callable[[], float]` 先例），收口 rate limiter、passive idle、scheduler、workspace access TTL、ParseBudget、Gateway deadline 与 qdrant 等待。
2. **topic 域字段分类**：按 A2-P 归档 Plan 附录 A 的方法逐字段判定 UTC 业务时间 vs monotonic 运行状态；`is_idle` 类判定改为注入时钟。
3. **耗时统计统一**：`time.time()` 差值全部改 monotonic/perf_counter（耗时不属于业务时间，已有先例：A2-P 把 WorkEvent 时长从 datetime 差值改为 monotonic 测量）。
4. **展示 allowlist**：把 `sys_clock`、`log_handler` 及未来人类可读时间点固化为显式 allowlist，纳入静态检查。
5. **静态门禁**：扫描生产源码中的裸 `datetime.now()`（非 UTC）、以 `time.time()` 计算耗时、未经 allowlist 的本地时区用法——复用 A2-P 验收时验证过的扫描方式。
6. **`datetime.now(UTC)` 内联收口**：把现存 16 处内联调用统一到 `utc_now()`，使"唯一生产时间入口"的声明成立。

## 4. 主要未知项

- passive buffer 与 topic 状态是否需要跨重启恢复（当前均为进程内，重启即失；若未来持久化，idle 语义必须改为 UTC 业务时间+重算，monotonic 不可持久化）；
- rate limiter 的桶语义在 monotonic 下的迁移边界（外部可见行为是否变化）；
- `StreamMessage.timestamp`（公共导出、无生产写入者）是删除字段还是迁 UTC——涉及公共 API 决策；
- 前端是否消费任何后端时间字段的 numeric 形态（当前核查结论是没有，需在动手时复核）。

## 5. 升级为 Plan 的条件

- 附录 D 清单逐项复核完成（代码位置与风险仍是现状）；
- 每个治理子域（monotonic 统一 / topic 分类 / 耗时 / allowlist / 门禁）有明确 owner 与验收标准；
- 与 A2（缓存）及 A3（Session/Topic）计划的时间相关契约无冲突。

在升级为 Plan 之前，本文件不构成任何排期承诺；新代码的时间使用仍应遵循 A2-P 已确立的约定（UTC-aware 业务时间、局部 now 注入、monotonic 做相对时长）。
