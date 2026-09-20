---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-09-20
---

# Plans

本目录只存放已经绑定明确版本或里程碑、能够独立实施和验收，但尚未完全成为当前事实的功能、重构与迁移计划。已完成的计划移入 [Archived Plans](../archive/plans/README.md)，不在此目录继续作为活动实施入口。

v0.6.2 W1 Chat Attachments 实现与验收已完成，当前行为以 [Workspace 架构](../architecture/workspace.md) 与 [Chat 附件链路](../system/attachments.md) 为事实入口，W1 Plan 已移入 archive/plans，保留实施历史。v0.7.0 计划 A 现作为协调入口，并拆为 A1–A6 六个可独立验收的执行计划（A1 已完成并归档）；计划 B 继续负责外部协议、Passive 升级和参考客户端。旧的 Workspace Runtime 聚合与缓存所有权迁移保留 v0.6.2 的已完成历史。其他候选工作与版本顺序见 [ROADMAP](../ROADMAP.md)，实施前分别建立正式 Plan。

| 当前计划 | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.7.0 计划 A：Workspace 资源平面重构协调计划](./v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md) | Active | 维护 A1–A6 的共同边界、依赖和发布出口；不再重复维护子计划的完整实施细节 |
| [A2 Workspace 资源读取、Runtime 与派生缓存](./v0.7.0-a2-workspace-resource-reads-and-caches.md) | Active | 完整 MemoryAtom/Profile 定义读取、Workspace 共享缓存与写入失效；命中逐次资源授权且不回源，不等待 Pending 状态扩展 |
| [A3 Conversation Session 与 Topic 投影边界](./v0.7.0-a3-conversation-session-and-topic-projection.md) | Planned | 新增 Session，演进 InteractionPayload/TurnEvent；交付 Topic 生命周期、交互/资料公共路由与授权结果查询 |
| [A4 共享 Pending 与主动记忆写入](./v0.7.0-a4-pending-memory-intents.md) | Planned | 基于前置读取和 Topic 能力交付共享 Pending、主动提交、完整引用解析与结算 |
| [A5 Patchouli 共同 API 与业务职责收敛](./v0.7.0-a5-patchouli-unified-api.md) | Active | 消费 A1–A4 成果核对全局 API，补检索/使用报告差额，形成旧服务职责退出清单 |
| [A6 Actor 适配与集成收口](./v0.7.0-a6-actor-adapters-and-integration.md) | Planned | 切换 Alice、Passive、管理和 MTP 消费者，完成稳定装配、旧旁路退出与 shutdown 验收 |
| [v0.7.0 计划 B：外部记忆服务与 Actor 交互契约](./v0.7.0-external-memory-service-and-actor-interaction.md) | Planned | 建立被动对话与主动资源交互的外部协议，明确身份、来源、领域提交与结果查询；以无 Alice 的参考客户端验收，并用历史样例验证后续导入的契约边界 |
| [话题折叠、Actor 上下文与原始证据统一改造](./topic-folding-context-and-raw-evidence.md) | Planned / 占位 | 独立里程碑，统筹话题折叠算法重构、原始证据和长 turn 上下文两份 Idea；详细设计与发布版本待补齐 |

默认按 A1 → A2 → A3 → A4 → A5 → A6 推进；A2/A3 可在 A1 后并行，其余完整依赖见[协调入口第 3 节](./v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md#3-依赖图与执行顺序)。每份领域计划交付模型、公共路由和真实行为，A5 不再是前置 API 发布平台。A2/A5 的 Active 承接既有工作，不代表新增范围已经完成；旧编号迁移表见协调入口第 1 节。

2026-09-19 状态更新：A1 已完成实施、验收、代码审查与文档收口并归档为 [v0.7.0 A1 Workspace 访问边界与授权（归档）](../archive/plans/v0.7.0-a1-workspace-access-boundary.md)。当前事实入口：[Workspace 架构](../architecture/workspace.md)第 4 节（统一认证网关、两类登记、guard 签发生命周期与逐次行为授权）、[错误模型](../contracts/error-model.md)第 4.4 节、[子系统公共契约](../contracts/subsystem-contracts.md)第 3.5 节与 [ADR-0005](../architecture/decisions/0005-unified-actor-authentication-and-workspace-authorization.md)。真实生产入口切换、shutdown 关闭时机和兼容分支退出仍由 [A6](./v0.7.0-a6-actor-adapters-and-integration.md) 完成；附件上传的 scope 一致性缺陷单独追踪于 [Todo](../todo/workspace-asset-upload-access-scope-mismatch.md)。

A1–A5 可独立组合验收，A6 负责真实消费者与生产收口，A 系列不等待 B 外部客户端。三方调用示例集中在 [A5](./v0.7.0-a5-patchouli-unified-api.md)；Topic 生命周期只在 [A3 第 4 节](./v0.7.0-a3-conversation-session-and-topic-projection.md#4-topic-路由指令与交接)定义，完整折叠算法仍归[专项占位计划](./topic-folding-context-and-raw-evidence.md)。B 按 A1–A5 实际交付能力映射外部协议；A/B 共同构成 v0.7.0 发布范围。具体 harness connector、执行基座与完整历史导入仍按 ROADMAP 后续排期推进。

已完成计划与实施历史：

v0.7.0 的缓存所有权与 ADR-0004 的版本适用关系以 [A2 第 1.1 节](./v0.7.0-a2-workspace-resource-reads-and-caches.md)裁定为准；以下归档记录仍描述 v0.6.2 基线。完整引用解析见 [A4 第 4.1 节](./v0.7.0-a4-pending-memory-intents.md#41-共同引用读取与-alias-resolver-归属)，Profile 定义/执行分离见 [A2 第 1.4 节](./v0.7.0-a2-workspace-resource-reads-and-caches.md#14-profile-定义读取与执行配置应用)，正式 ADR 替代须在 A6 联合验收收尾后进行。

| Plan | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.7.0 A1 Workspace 访问边界与授权（归档）](../archive/plans/v0.7.0-a1-workspace-access-boundary.md) | Archived | 统一认证网关、两类登记、guard 签发生命周期与逐次行为授权已完成；当前事实见 [Workspace 架构](../architecture/workspace.md)第 4 节与 [ADR-0005](../architecture/decisions/0005-unified-actor-authentication-and-workspace-authorization.md) |
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
