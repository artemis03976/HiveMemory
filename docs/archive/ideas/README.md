---
title: Archived Ideas
status: current
owner: project
scope: archived-idea-index
last_reviewed: 2026-11-12
---

# Archived Ideas

本目录保存已经实现、被替代或不再跟踪的 Idea，只用于追溯；当前行为从对应事实文档进入，活动的 Idea 见 [Ideas](../../ideas/README.md)。每条登记一行：归档日期、实现所在的 PR（未合并时为提交）与替代它的当前文档；设计推导、owner 决定与未跟踪事项的去向见各 Idea 自身的归档说明。

- [写入意图（PendingAtom）的后续方向](./pending-intent-migration.md)：2026-11-16，PR #221、#213；[读取视图与写入意图](../../workspace/read-view-and-write-intents.md)，生成材料移入 [v0.8 写入链路 Idea](../../ideas/memory-flow-write-path.md)
- [Chat Run 取消与生命周期后续设计](./chat-run-lifecycle-follow-ups.md)：2026-11-16，取消的最小闭环已落地，其余候选不再跟踪；[任务进程](../../workspace/task-process.md)
- [复合意图分解](./composite-intent-decomposition.md)：2026-11-16，未实施，不再跟踪；[Gateway 分析](../../gateway/analysis.md)
- 全项目时间使用统一规范：2026-11-16 转为 [Todo](../../todo/time-semantics-outside-memory-domain.md)，删除前的最后版本见提交 `08f2c354`
- [统一语言设置与外部 Agent 文字的英文版](./language-setting-unification.md)：2026-11-12，PR #220；[全局 i18n](../../system/i18n.md)
- [日常实例与开发实例分开](./daily-and-dev-instance-separation.md)：2026-11-12，PR #219；[安装与启动](../../help/setup.md#47-与桌面安装版同时开发)、[桌面客户端](../../frontend/desktop-client.md)
- [正式的 Runtime Event 体系与全局运行页](./runtime-event-system-and-run-view.md)：2026-11-07，PR #193、#195；[运行时事件与可观测性](../../components/observability.md#11-一次运行)、[全局运行页](../../frontend/run-view.md)
- [主动写入与被动提取的收敛](./generation-source-convergence.md)：2026-11-04，PR #176、#177、#178；[Patchouli 生成](../../patchouli/generation.md)、[合并版本冲突 Todo](../../todo/merge-version-conflict.md)
- [话题结算的多条提取、查重与更新](./settlement-multi-memory-extraction-and-dedup.md)：2026-11-01，PR #154、#155；[Patchouli 生成](../../patchouli/generation.md)、[查重 Todo](../../todo/generation-dedup-decision-signal.md)
- [工具解析链路重写](./tool-call-parsing-chain.md)：2026-10-30，PR #147；[数据模型](../../architecture/data-model.md)、[感知与短期话题](../../patchouli/perception.md)
- [Import Bus：只向网络输入的记忆来源](./import-bus.md)：2026-10-30，PR #137，后续方向不再跟踪；[Import Bus](../../system/import-bus.md)
- [执行线程的正式定义](./execution-thread-definition.md)：2026-10-19，PR #133；[执行线程](../../workspace/execution-threads.md)
- [Chat Attachments 初步设计备忘](./workspace-mvp-chat-attachments-design.md)：2026-09-27，PR #97；[Chat 附件链路](../../system/attachments.md)
