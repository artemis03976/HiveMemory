---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-09-17
---

# Plans

本目录只存放已经绑定明确版本或里程碑、能够独立实施和验收，但尚未完全成为当前事实的功能、重构与迁移计划。已完成的计划移入 [Archived Plans](../archive/plans/README.md)，不在此目录继续作为活动实施入口。

v0.6.2 W1 Chat Attachments 实现与验收已完成，当前行为以 [Workspace 架构](../architecture/workspace.md) 与 [Chat 附件链路](../system/attachments.md) 为事实入口，W1 Plan 已移入 archive/plans，保留实施历史。v0.7.0 计划 A 现作为协调入口，并拆为 A1–A6 六个可独立验收的执行计划；计划 B 继续负责外部协议、Passive 升级和参考客户端。旧的 Workspace Runtime 聚合与缓存所有权迁移保留 v0.6.2 的已完成历史。其他候选工作与版本顺序见 [ROADMAP](../ROADMAP.md)，实施前分别建立正式 Plan。

| 当前计划 | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.7.0 计划 A：Workspace 资源平面重构协调计划](./v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md) | Active | 维护 A1–A6 的共同边界、依赖和发布出口；不再重复维护子计划的完整实施细节 |
| [A1 Workspace 访问边界与授权](./v0.7.0-a1-workspace-access-boundary.md) | Active | 统一 CallerPrincipal、Workspace admission、operation grant、四阶段检查和可信上下文传播 |
| [A2 Patchouli 共同 API 与业务职责收敛](./v0.7.0-a2-patchouli-unified-api.md) | Active | 从 System、Alice、外部 Actor 的共同操作收敛 Patchouli application API，退出旧复合领域路径 |
| [A3 WorkspaceRuntime 与派生缓存迁移](./v0.7.0-a3-workspace-runtime-and-caches.md) | Planned | 建立进程级 WorkspaceRuntime，迁移 Profile/Atom cache，统一 revision、epoch、失效和关闭 |
| [A4 Conversation Session 与 Topic 投影边界](./v0.7.0-a4-conversation-session-and-topic-projection.md) | Planned | 冻结 Session/Segment/Part 与 Topic/LogicalBlock 的分层、路由、顺序、来源和结构化投影时机 |
| [A5 共享 Pending 与主动记忆写入](./v0.7.0-a5-pending-memory-intents.md) | Planned | 提供跨 Actor 的 Pending 读写一致性、独立 memory_intent.submit、完整 Topic 资料和结算解析 |
| [A6 Actor 适配与集成收口](./v0.7.0-a6-actor-adapters-and-integration.md) | Planned | 切换 Alice、Passive、管理和 MTP 消费者，接入稳定装配，完成旧旁路退出与 shutdown 验收 |
| [v0.7.0 计划 B：外部记忆服务与 Actor 交互契约](./v0.7.0-external-memory-service-and-actor-interaction.md) | Planned | 建立被动对话与主动资源交互的外部协议，明确身份、来源、领域提交与结果查询；以无 Alice 的参考客户端验收，并用历史样例验证后续导入的契约边界 |
| [话题折叠、Actor 上下文与原始证据统一改造](./topic-folding-context-and-raw-evidence.md) | Planned / 占位 | 独立里程碑，统筹话题折叠算法重构、原始证据和长 turn 上下文两份 Idea；详细设计与发布版本待补齐 |

A1–A5 可在独立组合和真实 application 链中分别验收，A6 负责跨计划消费者切换和生产收口；A 系列不等待 B 的外部客户端。三类调用主体如何收敛到同一业务链路的例子集中在 [A2](./v0.7.0-a2-patchouli-unified-api.md)。A4 的 Session/Topic 交互模型与 [话题折叠专项占位计划](./topic-folding-context-and-raw-evidence.md)交接，但折叠算法不自动纳入 A4。B 消费 A1/A2/A4/A5 的稳定能力并用真实组件集成；两组计划共同构成 v0.7.0 的发布范围。具体 harness connector、执行基座与完整历史导入仍按 ROADMAP 后续排期推进。

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
