---
title: Topic Folding, Actor Context and Raw Evidence
status: planned
design_status: placeholder
owner: patchouli-alice-system
target: topic-folding-context-and-raw-evidence-milestone
scope: topic-compaction-intra-turn-context-and-raw-evidence
related_docs:
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/ideas/long-running-agent-intra-turn-context-folding.md
  - docs/todo/page-folding-cross-ingress-follow-ups.md
  - docs/plans/v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md
  - docs/plans/v0.7.0-a3-conversation-session-and-topic-projection.md
  - docs/plans/v0.7.0-external-memory-service-and-actor-interaction.md
updates:
  - docs/patchouli/perception.md
  - docs/patchouli/artifacts.md
  - docs/patchouli/generation.md
  - docs/alice/agent-runtime.md
  - docs/system/passive-ingress.md
  - docs/contracts/subsystem-contracts.md
  - docs/ROADMAP.md
last_reviewed: 2026-09-18
---

# 话题折叠、Actor 上下文与原始证据统一改造计划（占位）

本文为后续专项工作预留独立计划入口，目标里程碑为“话题折叠、上下文边界与原始证据管理”。当前仅记录问题、覆盖范围和设计待办；具体发布版本、实施顺序及详细方案待后续设计阶段确定，不自动纳入 v0.7.0 的发布范围。

`planned` 表示已决定将这些联动内容作为独立计划推进，`design_status: placeholder` 表示设计尚未展开。本文不是可直接执行的实施方案，也不表示两份 Idea 中的候选模型、默认值或失败策略已经被采纳。

## 1. 背景与目标

当前话题折叠同时影响 Patchouli 的短期材料工作集、Alice 使用的上下文和后续记忆生成输入。外部 harness 又可能自行管理对话压缩，需要重新划分资源材料维护与执行上下文管理的责任，并统筹折叠原文保留、长 turn 处理和记忆生成之间的关系。

本计划拟统一承接以下设计来源：

- [Page Folding Raw Evidence](../ideas/PatchouliPageFoldingRawEvidenceDesign.md)：折叠原文保存、证据引用及高保真记忆生成。
- [长时间运行 Agent 的 Turn 内上下文折叠](../ideas/long-running-agent-intra-turn-context-folding.md)：长 turn、分段、checkpoint、上下文预算和外部 harness 协作。
- [Page Folding 跨入口后续技术债](../todo/page-folding-cross-ingress-follow-ups.md)：跨入口上下文所有权、容量、来源与证据缺口。

两份 Idea 保留为设计推导来源；后续在本文中逐项确认采用、调整或不采用的内容。

## 2. 预期覆盖范围

| 工作域 | 后续设计需要覆盖的内容 |
|:---|:---|
| 话题压缩与折叠算法 | RelayController 的定位、触发策略与摘要算法分离、token/块数预算、摘要自身增长、超大块和保留后缀 |
| 资源与执行边界 | Patchouli Topic 工作集、Actor prompt history、原始证据与记忆生成输入的所有权；执行接力摘要与记忆摘要的用途差异 |
| 原始证据 | 保存时机、存储归属、引用与覆盖范围、折叠前后的材料交接、容量、保留期限、删除和写入失败 |
| 长 turn 与运行中折叠 | segment/checkpoint 候选模型、事件顺序、覆盖关系、工具调用原子边界、seal/cancel/failure 及恢复范围 |
| Alice 与外部 harness | 自主管理上下文、外部摘要/checkpoint 的来源、混合来源 Topic，以及显式委托上下文管理的适用范围 |
| Generation 与生命周期 | 普通 settlement、高保真分块处理、去重与来源关联、幂等、背压、关闭及持久化恢复的承诺边界 |

以上是设计覆盖范围，不预先要求落地全部候选机制；最终实现切片和非目标在详细设计时逐项冻结。具体类名、字段、存储实现、同步或异步执行方式、可靠性等级和默认策略均未确定。

A3 演进后的 InteractionPayload 表示一次完成或明确封口的交互；本计划的长 turn 分片/checkpoint 是运行中预算与覆盖机制，与交互封口拥有不同生命周期。后续应引用 interaction_id 与 TurnEvent 范围建立证据与折叠覆盖关系，避免另建一套来源 ID。资源 Topic 工作集、Session 历史和 Actor prompt history 分离是输入边界；本计划继续设计算法、原文保留与质量，不重新接管外部 harness 压缩。

## 3. 与现有计划的关系

- [v0.7.0 计划 A 协调入口](./v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md)提供整体边界；[A3 Session 与 Topic 投影计划](./v0.7.0-a3-conversation-session-and-topic-projection.md)冻结 Session/InteractionPayload/TurnEvent 与 Topic/LogicalBlock 的交接。本计划只在此基础上继续设计折叠、长 turn 和原始证据，不重新定义会话容器。
- [v0.7.0 计划 B](./v0.7.0-external-memory-service-and-actor-interaction.md)提供外部接入与交互契约；本计划需与其对齐上下文所有权、来源和可能的 checkpoint/上下文服务交接。
- 本次占位不改变 A 系列/B 已有验收出口，不将完整原文保全、长 turn 管理或上下文托管自动设为 A3、A4 或 B 的完成前提。具体依赖、交付顺序和版本调整在详细设计时一并评估。
- 完整历史导入、特定厂商 connector、沙箱及通用工作流平台不因本文建立而并入范围；本计划只需明确与它们相关的交接。

## 4. 后续补齐清单

- [ ] 刷新代码与行为基线，收集 Alice、外部 harness 和长对话的代表性样本。
- [ ] 明确所有权、生命周期、目标数据流及两份 Idea 的逐项承接决定。
- [ ] 冻结折叠算法、摘要用途、证据保留、预算和失败策略。
- [ ] 定义跨系统接口、版本/覆盖关系、幂等与恢复承诺。
- [ ] 补齐兼容与迁移方案，包括现有 Topic、配置、摄入重试、settlement 和外部调用方。
- [ ] 制定质量、延迟、容量、隔离、证据完整性和失败恢复的测试与验收标准。
- [ ] 划定实施阶段、非目标、回滚路径和发布版本，并同步 ROADMAP 与 A/B 交接。
- [ ] 核对最终需更新的事实文档；仅在实现、测试和审查收尾后晋升设计并归档本计划。

完成上述设计补齐后，再据此启动实现。本次占位不修改生产代码，不声明已解决现有折叠或证据丢失问题。
