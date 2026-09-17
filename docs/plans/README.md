---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-09-16
---

# Plans

本目录只存放已经绑定明确版本或里程碑、能够独立实施和验收，但尚未完全成为当前事实的功能、重构与迁移计划。已完成的计划移入 [Archived Plans](../archive/plans/README.md)，不在此目录继续作为活动实施入口。

v0.6.2 W1 Chat Attachments 实现与验收已完成，当前行为以 [Workspace 架构](../architecture/workspace.md) 与 [Chat 附件链路](../system/attachments.md) 为事实入口，W1 Plan 已移入 archive/plans，保留实施历史。v0.7.0 拆为 Workspace 资源体系与内部执行边界、外部记忆服务与 Actor 交互契约两份计划；旧的 Workspace Runtime 聚合与缓存所有权迁移保留 v0.6.2 的已完成历史。其他候选工作与版本顺序见 [ROADMAP](../ROADMAP.md)，实施前分别建立正式 Plan。

| 当前计划 | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.7.0 计划 A：Workspace 资源体系与 Agent 执行边界](./v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md) | Active | 建立独立于 Alice 的 WorkspaceRuntime，统一管理员、Alice 与外部 Actor 的访问边界和既有 application/GlobalSystemBus 调用；迁移 Profile/Atom 派生缓存与失效，收缩执行侧；不新增平行业务 port/provider 层 |
| [v0.7.0 计划 B：外部记忆服务与 Actor 交互契约](./v0.7.0-external-memory-service-and-actor-interaction.md) | Planned | 建立被动对话与主动资源交互的外部协议，明确身份、来源、领域提交与结果查询；以无 Alice 的参考客户端验收，并用历史样例验证后续导入的契约边界 |
| [话题折叠、Actor 上下文与原始证据统一改造](./topic-folding-context-and-raw-evidence.md) | Planned / 占位 | 独立里程碑，统筹话题折叠算法重构、原始证据和长 turn 上下文两份 Idea；详细设计与发布版本待补齐 |

A 可独立实施和验收，不等待 B 的外部协议与客户端；B 消费 A 的统一访问边界与公开 application 契约，最终使用真实组件完成集成。三类调用主体的例子见计划 A 第 5.6.5 节。两份计划共同构成 v0.7.0 的发布范围。具体 harness connector、执行基座与完整历史导入仍按 ROADMAP 后续排期推进。

已完成计划与实施历史：

| Plan | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.6.2 W0 Workspace MVP（归档）](../archive/plans/v0.6.2-workspace-mvp.md) | Archived | 已完成 `WorkspaceIdentity`、端到端 scope、双 Workspace 隔离、进程内 WorkspaceAssetStore、两级状态机和 SemanticBuffer binding；当前事实见 [Workspace 架构](../architecture/workspace.md) |
| [v0.6.2 Identity 投影收敛（归档）](../archive/plans/v0.6.2-identity-projection-cleanup.md) | Archived | 已完成服务入口统一 `IdentityScope`、`InteractionTurnSnapshot` actor 值对象化、读侧兼容属性收口与 `system` 保留 actor 语义；当前事实见 [Workspace 架构](../architecture/workspace.md) 与 [System 应用服务](../system/application-services.md) |
| [v0.6.2 Workspace Runtime 聚合与缓存所有权迁移（归档）](../archive/plans/v0.6.2-workspace-runtime-cache-migration.md) | Archived | 已完成 Workspace-aware cache key（atom cache 按 `(WorkspaceIdentity, alias)`、profile cache 按完整授权坐标分区）；后续经 [ADR-0004](../architecture/decisions/0004-execution-path-derived-caches.md) 将派生缓存所有权归还 AliceRuntime、聚合解体，PendingAtomRuntime 保持 Alice 所有；当前事实见 [Workspace 架构](../architecture/workspace.md)、[System 组合根](../system/composition.md)、[ADR-0004](../architecture/decisions/0004-execution-path-derived-caches.md) 与 [Alice](../alice/README.md) |
| [v0.6.2 W1 Chat Attachments（归档）](../archive/plans/v0.6.2-w1-chat-attachments.md) | Archived | 已在 W0 公共契约上实现附件上传、确定性解析、representation READY/FAILED、Chat 选择与 lease、AttachmentCompiler、Topic binding 与按需 Artifact promotion；当前事实见 [Chat 附件链路](../system/attachments.md) |

`v0.6.2` 的 Workspace 工作拆分为 W0/W1 两个独立交付切片。W0 已完成并归档；W1 Chat Attachments 已按 Plan 完成实施、验收和归档，把 W0 的稳定公共契约作为硬前置，落地了上传、解析、AttachmentCompiler、Topic binding 与 Artifact promotion。已有 V1 Memory/Artifact 的批量转换由独立的[迁移归档 Plan](../archive/plans/v0.6.2-v1-memory-legacy-migration.md)承接并已完成；历史对话导入属于后续路线图。

[Workspace MVP 初步设计](../ideas/workspace-mvp-chat-attachments-design.md)继续保存 W1 Chat Attachments 的设计推导和开放问题；W0 的历史实施顺序、代码落点、兼容策略、删除/lease/settle 矩阵和测试出口见归档 Plan，当前事实不再由 Plan 或 Idea 承担。

## 准入规则

- 必须填写明确 `target` 或在标题/范围中绑定当前版本；
- 必须具有独立目标、非目标、迁移方式、测试和验收出口；
- 未排期治理主题进入 [Governance](../governance/README.md)；
- 尚需验证价值、样本或所有权的候选进入 [Ideas](../ideas/README.md)；
- 范围较小的缺陷和技术债进入 [Todo](../todo/README.md)；
- 已完成或被替代的实施稿进入 [Archived Plans](../archive/plans/README.md)。
