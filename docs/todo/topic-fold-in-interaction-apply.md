---
title: Topic Folding Inside Interaction Apply
status: todo
owner: patchouli
scope: perception-compact-timing-and-interaction-submission-timeout
code_paths:
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/control/interaction_submission.py
  - src/hivememory/engines/perception/relay_controller.py
related_docs:
  - docs/patchouli/perception.md
  - docs/system/import-bus.md
  - docs/ideas/short-term-memory-and-context-compaction.md
  - docs/todo/import-bus-real-use-observation.md
last_reviewed: 2026-10-27
---

# 话题折叠计入交互提交的处理时限

## 问题

话题折叠（compact）是交互应用的后置步骤。`PerceptionFamiliar.apply_interaction` 把 block 写入话题、journal 记为 `INTERACTION_APPLIED` 之后，仍在持有话题 lease 的同一次调用里等待 `_compact_topic_if_needed`，折叠返回后才记录本地完成并完成 journal。交互提交队列把这次调用作为一个工作项执行，折叠的耗时因此计入提交队列的单次处理时限，也计入导入与主动链路对“应用完成”的等待。

规则摘要（`SimpleRelay`）是纯计算，这个问题不显现；LLM relay 在折叠时调用 librarian 模型，输入是被折叠各轮的用户消息与回复，耗时随折叠量与模型速度变化，可能超过时限。

Owner 判断（2026-10-27）：话题折叠的耗时不应计入提交队列的写入。交互写入话题即完成提交，折叠是话题自身的维护。

## 证据

- `InteractionSubmissionQueue` 的 lane 策略在代码中固定为 `timeout_seconds=30.0`、`max_attempts=3`，没有配置项。
- `WorkQueueRuntime` 以 `asyncio.timeout` 包裹 handler，超时按普通失败重试。重试时 journal 处于 `INTERACTION_APPLIED`，跳过写入、路由回已记录的话题并再次执行折叠。
- `LLMRelayController` 只在 `ModelProviderError` 时回退为规则摘要；时限触发的取消不经过这条回退，折叠以失败结束。
- 尚未实测：仓库配置（`fold_token_threshold: 131072`、relay 为 `llm`、librarian 为 `deepseek-chat`）下一次折叠的实际耗时。

## 影响

- **导入链路**：Import Bus 在应用完成后才确认，等待时限为 `import_bus.apply_wait_seconds`（默认 30 秒）。折叠较慢时导入先得到 503 `apply_pending`；三次尝试都超时后交互为 failed，导入返回 500 `import_bus.apply_failed`，连接器在同一轮连续三次 500 后写入拒绝记录并跳过这一轮，而这一轮的 block 实际已经写入话题。
- **主动链路**：任务进程封口后的提交等待同一队列的应用结果，折叠超时使 finalize 以 `interaction_apply` 阶段失败。
- **连续失败**：折叠失败后话题仍超过阈值，之后写入同一话题的每次应用都会再次触发折叠，可能接连失败。
- 折叠期间话题 lease 一直被占用（[感知与短期话题](../patchouli/perception.md#8-当前限制)已记录），超时与重试不会缩短占用。

## 完成条件

- 交互写入话题与 journal 后即为应用完成：提交队列的处理时限、Import Bus 的确认与主动链路的 finalize 都不再包含折叠耗时；
- 折叠失败（含超时）不使已写入的交互判为应用失败，也不使同一话题的后续应用接连失败；
- 折叠写回与同一话题的写入、结算仍不交错；
- 在归属层测试中覆盖：慢速或超时的折叠不改变交互应用结果，折叠失败后话题内容完整、之后仍能再次折叠；
- 更新[感知与短期话题](../patchouli/perception.md)中的折叠时机与当前限制。

## 待决问题

- 折叠移出应用后由谁触发：应用完成后由 Familiar 另起任务，还是由维护调度器扫描超过阈值的话题；
- 折叠是否需要独立时限，超时后回退为规则摘要，还是保留原 blocks 留待下次；
- 折叠与结算（idle、LRU、关停、手动）在同一话题上的先后与 lease 竞争。

## 相关文档

- [感知与短期话题](../patchouli/perception.md)：折叠时机、Relay 与当前限制；
- [Import Bus](../system/import-bus.md)：应用完成后确认、错误分类与连接器重发；
- [短期记忆与上下文压缩](../ideas/short-term-memory-and-context-compaction.md)：Topic folding 的整体优化方向；
- [Import Bus 真实使用观察](./import-bus-real-use-observation.md)：折叠频率与导入失败比例的观察。
