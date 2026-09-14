---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-09-14
---

# Plans

本目录只存放已经绑定明确版本或里程碑、能够独立实施和验收，但尚未完全成为当前事实的功能、重构与迁移计划。已完成的计划移入 [Archived Plans](../archive/plans/README.md)，不在此目录继续作为活动实施入口。

v0.6.2 W1 Chat Attachments 实现与验收已完成，当前行为以 [Workspace 架构](../architecture/workspace.md) 与 [Chat 附件链路](../system/attachments.md) 为事实入口，Plan 本体保留在此目录等待归档。Workspace 资源体系与 Agent 执行边界重构已独立排入 v0.7.0；旧的 Workspace Runtime 聚合与缓存所有权迁移保留 v0.6.2 的已完成历史。其他候选工作与版本顺序见 [ROADMAP](../ROADMAP.md)，实施前分别建立正式 Plan。

| 当前计划 | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.7.0 Workspace 资源体系与 Agent 执行边界](./v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md) | Planned | 建立独立于 Alice 执行的进程级 WorkspaceRuntime，统一资源读取、Profile/Atom 派生缓存、授权重验与失效；以 Passive Ingress 为首个外部 Actor 适配通道，收缩 Alice/AgentRuntime，明确 PendingAtom 暂留和 MTP 适配边界 |

已完成计划与实施历史：

| Plan | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.6.2 W0 Workspace MVP（归档）](../archive/plans/v0.6.2-workspace-mvp.md) | Archived | 已完成 `WorkspaceIdentity`、端到端 scope、双 Workspace 隔离、进程内 WorkspaceAssetStore、两级状态机和 SemanticBuffer binding；当前事实见 [Workspace 架构](../architecture/workspace.md) |
| [v0.6.2 Identity 投影收敛（归档）](../archive/plans/v0.6.2-identity-projection-cleanup.md) | Archived | 已完成服务入口统一 `IdentityScope`、`InteractionTurnSnapshot` actor 值对象化、读侧兼容属性收口与 `system` 保留 actor 语义；当前事实见 [Workspace 架构](../architecture/workspace.md) 与 [System 应用服务](../system/application-services.md) |
| [v0.6.2 Workspace Runtime 聚合与缓存所有权迁移（归档）](../archive/plans/v0.6.2-workspace-runtime-cache-migration.md) | Archived | 已完成 Workspace-aware cache key（atom cache 按 `(WorkspaceIdentity, alias)`、profile cache 按完整授权坐标分区）；后续经 [ADR-0004](../architecture/decisions/0004-execution-path-derived-caches.md) 将派生缓存所有权归还 AliceRuntime、聚合解体，PendingAtomRuntime 保持 Alice 所有；当前事实见 [Workspace 架构](../architecture/workspace.md)、[System 组合根](../system/composition.md)、[ADR-0004](../architecture/decisions/0004-execution-path-derived-caches.md) 与 [Alice](../alice/README.md) |
| [v0.6.2 W1 Chat Attachments](./v0.6.2-w1-chat-attachments.md) | Implemented（待归档） | 已在 W0 公共契约上实现附件上传、确定性解析、representation READY/FAILED、Chat 选择与 lease、AttachmentCompiler、Topic binding 与按需 Artifact promotion；当前事实见 [Chat 附件链路](../system/attachments.md) |

`v0.6.2` 的 Workspace 工作拆分为 W0/W1 两个独立交付切片。W0 已完成并归档；W1 Chat Attachments 已按 Plan 完成实施与验收，把 W0 的稳定公共契约作为硬前置，落地了上传、解析、AttachmentCompiler、Topic binding 与 Artifact promotion。历史数据批量转换也不在 W0/W1 主链路中执行，需要未来另立脚本或 Plan。

[Workspace MVP 初步设计](../ideas/workspace-mvp-chat-attachments-design.md)继续保存 W1 Chat Attachments 的设计推导和开放问题；W0 的历史实施顺序、代码落点、兼容策略、删除/lease/settle 矩阵和测试出口见归档 Plan，当前事实不再由 Plan 或 Idea 承担。

## 准入规则

- 必须填写明确 `target` 或在标题/范围中绑定当前版本；
- 必须具有独立目标、非目标、迁移方式、测试和验收出口；
- 未排期治理主题进入 [Governance](../governance/README.md)；
- 尚需验证价值、样本或所有权的候选进入 [Ideas](../ideas/README.md)；
- 范围较小的缺陷和技术债进入 [Todo](../todo/README.md)；
- 已完成或被替代的实施稿进入 [Archived Plans](../archive/plans/README.md)。
