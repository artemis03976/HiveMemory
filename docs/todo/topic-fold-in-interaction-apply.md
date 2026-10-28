---
title: Topic Folding Inside Interaction Apply
status: todo
owner: patchouli
scope: perception-compact-timing-and-interaction-submission-timeout
code_paths:
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/control/interaction_submission.py
  - src/hivememory/patchouli/control/interaction_apply_journal.py
  - src/hivememory/patchouli/system.py
  - src/hivememory/engines/perception/relay_controller.py
  - src/hivememory/infrastructure/work_queue/in_memory_store.py
related_docs:
  - docs/patchouli/perception.md
  - docs/system/import-bus.md
  - docs/ideas/short-term-memory-and-context-compaction.md
  - docs/todo/import-bus-real-use-observation.md
last_reviewed: 2026-10-27
---

# 话题折叠计入交互提交的处理时限

## 问题

话题折叠（compact）目前是交互应用的后置步骤：

- `PerceptionFamiliar.apply_interaction` 把 block 写入话题，journal 记为 `INTERACTION_APPLIED`；
- 之后仍在持有话题 lease 的同一次调用里等待 `_compact_topic_if_needed`；
- 折叠返回后，才记录本地完成并完成 journal。

交互提交队列把这次调用作为一个工作项执行。折叠的耗时因此计入提交队列的单次处理时限，也计入导入与主动链路对“应用完成”的等待。

规则摘要（`SimpleRelay`）是纯计算，这个问题不明显。LLM relay 在折叠时调用 librarian 模型，输入是被折叠各轮的用户消息与回复，耗时随折叠量和模型速度变化，可能超过时限。

## 处理方向（owner，2026-10-27）

- 折叠的耗时不计入提交队列对交互的处理时限：交互写入话题即完成提交，折叠是话题自身的维护。
- 话题接收数据与折叠解耦：折叠进行期间，提交队列暂停向该话题提交；同一话题的后续交互在队列中等待折叠结束，等待不计入它们的处理时限。

分析：另一种做法是折叠期间照常写入、写回时合并。它不需要等待，但写回前要校验话题在折叠开始之后的变化；当初把折叠放进 lease 内执行，正是为了去掉这类校验。阻塞的做法保留了这一点。

## 证据

- `InteractionSubmissionQueue` 的 lane 策略在代码中固定为 `timeout_seconds=30.0`、`max_attempts=3`，没有配置项。
- `WorkQueueRuntime` 以 `asyncio.timeout` 包裹 handler，时限从 handler 开始执行时计算。超时抛出的 `TimeoutError` 不在 `InteractionSubmissionHandler.classify_failure` 的可重试类型中（只有 `ConnectionError`、`TransientInteractionSubmissionError` 与 `TopicBusyError` 会重试），工作项一次即判为 failed，不经队列重试。
- 提交方重新提交已失败的交互时，队列以新的工作项执行。`apply_interaction` 发现 journal 处于 `INTERACTION_APPLIED`，就跳过写入、路由回已记录的话题，并再次执行折叠。
- `LLMRelayController` 只在 `ModelProviderError` 时回退为规则摘要；时限触发的取消不经过这条回退，折叠以失败结束。
- 提交队列的 lane 按排序键串行（`ordered_by_key`）：
  - 排序键为 `topic:{话题 ID}`，新建话题的提交为 `topic:NEW_TOPIC`；
  - 同一排序键上，第一个未到终态的工作项会挡住其后的全部工作项，无论它是否就绪；
  - 在队列中等待的时间不计入处理时限。
- Patchouli 停止时，先无时限地排空提交队列，再结算全部话题。
- 尚未实测：仓库配置（`fold_token_threshold: 131072`、relay 为 `llm`、librarian 为 `deepseek-chat`）下一次折叠的实际耗时。

## 影响

- **导入链路**：Import Bus 在应用完成后才确认，等待时限为 `import_bus.apply_wait_seconds`（默认 30 秒）。
  - 折叠超过处理时限时，工作项判为 failed，导入返回 500 `import_bus.apply_failed`；若导入的等待先到期，则返回 503 `apply_pending`。
  - 连接器重发时，Import Bus 重新路由并提交，应用会再执行一次折叠。
  - 同一轮连续三次得到 500 后，连接器写入拒绝记录并跳过这一轮，而这一轮的 block 实际已经写入话题。
- **主动链路**：任务进程封口后的提交会等待同一队列的应用结果，折叠超时使 finalize 以 `interaction_apply` 阶段失败。
- **连续失败**：折叠失败后话题仍超过阈值，之后写入同一话题的每次应用都会再触发折叠，可能接连失败。
- 折叠期间话题 lease 一直被占用（[感知与短期话题](../patchouli/perception.md#8-当前限制)已记录），超时与重试不会缩短占用。

## 候选设计

以下不是 owner 决定，实施时可以调整。

- **折叠作为独立工作项**：
  - 应用在写入 block 与 journal 后即完成；
  - 写入后话题超过折叠阈值时，向同一 lane 追加一个折叠工作项，排序键与该话题的提交相同（`topic:{话题 ID}`）；
  - 折叠项执行期间（含重试等待），同一话题的后续提交按现有的按键串行停在队列中，各自的处理时限从开始执行时计算。
- **去重**：同一话题同时只保留一个未结束的折叠项，已有待执行或执行中的折叠项时不再追加。
- **时限**：
  - lane 只有一个 handler 和一个时限，因此由 handler 按载荷类型分派，交互应用与折叠各自施加时限；
  - 通用队列组件不变。
- **折叠超时或失败**：
  - 已完成的交互不受影响；
  - 折叠超时后回退为规则摘要，使话题回到阈值以下，避免同一话题的每次写入都再触发一次慢折叠；
  - 另一选项是保留原 blocks，留待下次折叠。
- **与结算的关系**：
  - 折叠项执行期间照旧持有话题 lease。结算各来源的行为不变：空闲结算跳过本轮，LRU 改选其他话题，手动结算报忙；
  - 折叠项开始执行时，若话题已被结算删除，则直接结束。
- **关停**：Patchouli 开始停止后，折叠项直接结束、不再生成摘要，避免排空队列时多等一次模型调用；随后的结算会带走话题的完整 blocks。
- **journal**：应用不再有写入之后的本地义务，`LOCAL_COMPLETED` 阶段和“重新提交时补跑折叠”的分支可以删除。
- **并发**：lane 最多同时执行 4 个工作项，长时间的折叠会占用执行槽；单用户场景影响有限，实施后观察。
- **导入等待**：同一话题的下一轮导入可能要等前一个折叠结束，等待超过 `import_bus.apply_wait_seconds` 时，按现有规则返回 503 `apply_pending`，由连接器保留游标重发。这属于可重试的推迟，不计入连接器的连续失败。

## 完成条件

- 交互写入话题与 journal 后即为应用完成：该交互的处理时限、Import Bus 的确认与主动链路的 finalize 都不包含折叠耗时；
- 折叠进行期间，同一话题的后续提交在队列中等待折叠结束，等待不计入它们的处理时限；
- 折叠失败（含超时）不使已写入的交互判为应用失败，也不使同一话题的后续应用失败；
- 折叠写回与同一话题的写入、结算仍不交错；
- 归属层测试覆盖：
  - 慢速或超时的折叠不改变交互应用结果；
  - 折叠期间，同一话题的后续提交在折叠结束后应用，且不因等待而超时；
  - 折叠失败后话题内容完整，之后仍能再次折叠；
- 更新受影响的事实文档：
  - [感知与短期话题](../patchouli/perception.md)中的折叠时机与当前限制；
  - [Import Bus](../system/import-bus.md)中关于应用等待含折叠模型调用、写入后失败的重新提交的描述。

## 追踪

- 2026-10-27：启用 LLM relay 时登记；同日 owner 决定处理方向，在分支 `fix/topic-fold-in-interaction-apply` 实施。未绑定 Issue。

## 相关文档

- [感知与短期话题](../patchouli/perception.md)：折叠时机、Relay 与当前限制；
- [Import Bus](../system/import-bus.md)：应用完成后确认、错误分类与连接器重发；
- [短期记忆与上下文压缩](../ideas/short-term-memory-and-context-compaction.md)：Topic folding 的整体优化方向；
- [Import Bus 真实使用观察](./import-bus-real-use-observation.md)：折叠频率与导入失败比例的观察。
