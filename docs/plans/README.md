---
title: Plans
status: current
owner: project
scope: implementation-plan-rules-and-navigation-by-version
last_reviewed: 2026-11-27
---

# Plans

本目录存放已经绑定版本或里程碑、能够独立实施和验收，但尚未成为当前事实的计划。本索引是**版本内的计划导航**：按版本列出计划的状态与链接，不记录设计决定；计划的写法与准入见文末的“规则”。设计见 [Ideas](../ideas/README.md)，版本规划摘要见 [ROADMAP](../ROADMAP.md)，已完成的计划见 [Archived Plans](../archive/plans/README.md)。

## v0.7.3

已收尾，计划与收尾审计见 [Archived Plans](../archive/plans/README.md#v073)。

## v0.8.0

已收尾，计划与收尾审计见 [Archived Plans](../archive/plans/README.md#v080)。版本目标：生成核心的升级，摘要见 [ROADMAP](../ROADMAP.md) 第 3 节；设计见总纲[记忆核心：短期到中期的流转](../ideas/memory-flow-short-to-mid-term.md)；v0.8 的评测由试验场承载，见[孵化场、试验场与垃圾回收场](../ideas/experiment-field.md)第 6 节。

当前没有活动计划。

- 记忆类型的收敛、AgentProfileStore、生成的记忆指向交互轮次、馆长 agent、试验场第一版与写入时的更新：已完成，见 [Archived Plans](../archive/plans/README.md#v080)。

## v0.8.1

版本目标：短期记忆，摘要见 [ROADMAP](../ROADMAP.md) 第 3.4 节；设计见[短期记忆：话题的进展及其维护](../ideas/short-term-memory-and-context-compaction.md)。

- [话题折叠的重构](./v0.8.1-topic-fold-refactor.md)：折叠作为系统内部的材料压缩，写进展记录共用工作视图，折叠摘要不再给前台与 Gateway，折叠摘要结构化。

- 结算拆成沉淀与退役、话题跨重启保存、短期记忆代码整理、话题的存储介质、话题的进展记录、短期记忆代码可读性整理、压缩后推送与按句柄拉取、拉取当前话题与压缩后只推指针：已完成，见 [Archived Plans](../archive/plans/README.md#v081)。

## 已发布的版本

已发布版本的计划都已归档，见 [Archived Plans](../archive/plans/README.md)。

## 未绑定版本

当前没有活动计划。身份与访问主线是版本之外的工作，见[身份与访问体系 Idea](../ideas/identity-and-access-model.md)；观测与日志的其余工作包以治理主题跟踪，见[日志与运行时事件](../governance/observability/logs-and-events.md)。已完成的计划见 [Archived Plans](../archive/plans/README.md)。

## 规则

Plan 描述一个可独立验收的功能、重构或迁移，至少包括：背景和问题；目标与非目标；当前状态和缺口；目标设计；受影响的系统、契约和数据；兼容与迁移策略；测试和验收标准；风险与待决问题；完成时必须更新的当前文档。

“测试和验收标准”写需要守住的不变量、各自的归属层与验证命令，不逐条预列测试函数、参数化矩阵或“某符号不再存在”式检查；实施时验收测试写入归属层的现有测试文件，不形成以计划命名的测试集合，规则见[测试设计规范](../governance/testing/test-design-standards.md)第 4 节。Plan 是实施的工作依据，落地之后不承担当前设计说明。

进入 `plans/` 的条件：

- 计划只在能够立即执行时建立：阻塞问题已经决定、依赖已经成为事实（owner，2026-10-30）；
- 绑定明确的版本或里程碑（不用 `target: unscheduled`），能作为一个独立切片完成和验收，依赖与非目标已经基本明确；
- 包含迁移、测试、风险与完成后的文档更新清单，状态只能是 `planned` 或 `active`；
- 工作完成后立即合并当前事实并删除计划（[文档治理规范](../DOCUMENTATION.md)第 2.8、10 节）。

不能立即实施的方向留在 Idea，计划之间不预设先后，一个版本由多份计划依次完成。未排期的治理主题进入 Governance，尚需验证价值、样本或所有权的候选进入 Ideas，小范围缺陷与技术债进入 Todo。

计划之间满足三条约束，防止多份计划互相引用对方仍在变化的设计、一处变更联动修改一组计划：

1. **同一方向只有一份生效计划，不设总分结构**：不用 `parent_plan`，`plans/` 中不建立协调、冻结清单或宪章类文档；新工作与现有计划重叠时，先改写或退役原计划，不另开并行计划。
2. **设计依据只来自稳定来源**：当前事实文档、代码、ADR、此前归档并冻结的计划、已冻结的 Research 报告，以及作为背景的 Idea。已删除的计划不作为依据，它仍然有效的内容在收尾时已经写进事实文档或 ADR。Research 只提供外部证据与当时的适配分析，不证明 HiveMemory 已具备对应能力，也不替代本项目的设计决定。不以另一份活动计划的章节作为依据，也不把自身事项交给另一份活动计划收口。
3. **计划之间只允许顺序依赖**：`depends_on` 以整份计划为粒度，只表达先后；后继计划在前驱完成收尾、相关设计晋升为事实文档之后，依据事实文档编写或修订。

尚未开始实施的计划，如果方向仍然成立但前提已经改变，退回 Idea（[Ideas](../ideas/README.md)的“退回与归档”）；作废且不再讨论的计划直接删除（[文档治理规范](../DOCUMENTATION.md)第 10 节）。
