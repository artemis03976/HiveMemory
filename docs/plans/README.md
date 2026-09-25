---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-09-20
---

# Plans

本目录只存放已经绑定明确版本或里程碑、能够独立实施和验收，但尚未完全成为当前事实的功能、重构与迁移计划。已完成的计划移入 [Archived Plans](../archive/plans/README.md)，不在此目录继续作为活动实施入口。

v0.6.2 W1 Chat Attachments 实现与验收已完成，当前行为以 [Workspace 架构](../architecture/workspace.md) 与 [Chat 附件链路](../system/attachments.md) 为事实入口，W1 Plan 已移入 archive/plans，保留实施历史。v0.7.0 计划 A 现作为协调入口，包含 A1–A6 与新增的 A2-P 前置计划（A1 已完成并归档）；计划 B 继续负责外部协议、Passive 升级和参考客户端。旧的 Workspace Runtime 聚合与缓存所有权迁移保留 v0.6.2 的已完成历史。其他候选工作与版本顺序见 [ROADMAP](../ROADMAP.md)，实施前分别建立正式 Plan。

| 当前计划 | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.7.0 计划 A：Workspace 资源平面重构协调计划](./v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md) | Active | 维护 A1–A6 及 A2-P 的共同边界、依赖和发布出口；不再重复维护子计划的完整实施细节 |
| [v0.7.0 计划 A 边界宪章：Workspace 与记忆库的归属与独立工作契约](./v0.7.0-plan-a-boundary-charter.md) | Active | 冻结 Workspace 与 Patchouli 的归属判据、独立工作契约、状态归属表与证伪条件；已于 2026-09-23 采纳并完成 §10 联动修订，生效为计划 A 家族边界裁决唯一理由源 |
| [A2-P 记忆内容版本与 Lifecycle 状态重构](../archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md) | 已完成（2026-09-24 归档）/ A2 前置 | 完整版本历史、meta.lifecycle 聚合、受控局部更新与 schema 2.1 迁移；维护不改内容版本、不整颗重写，无 cache/Alice 独立验收 |
| [全项目时间语义与可控时钟统一](../archive/plans/v0.7.0-time-semantics-and-controllable-clock.md) | 已完成（2026-09-24 归档，嵌入 A2-P 实施）/ 跨子系统 | Memory 域 UTC 业务时间、四时间字段职责、局部 now 注入与 TimeFormatter 契约；全项目收口转为 idea（见 ideas 索引） |
| [A2 Workspace 读取能力面与派生缓存](./v0.7.0-a2-workspace-resource-reads-and-caches.md) | Active / 2026-09-25 整份重写 | 能力面读取方法（resolver 原地授权）、双 cache、失效事件协作（D1–D3）、operation 授权迁移与 L2 backing 契约；A2-0 冻结遗留开放点 |
| [A3 Conversation Session 与 Topic 投影边界](./v0.7.0-a3-conversation-session-and-topic-projection.md) | Planned | 新增 Session，演进 InteractionPayload/TurnEvent；交付 Topic 生命周期、交互/资料公共路由与授权结果查询 |
| [A4 共享 Pending 与主动记忆写入](./v0.7.0-a4-pending-memory-intents.md) | Planned | 基于前置读取和 Topic 能力交付共享 Pending、主动提交、完整引用解析与结算 |
| [A5 能力面 API 收敛与库职责退出](./v0.7.0-a5-patchouli-unified-api.md) | Active | 消费 A1–A4 成果收敛能力面/backing 两层目录，补检索/使用报告差额，operation 检查全量退役核对，形成旧服务职责退出清单 |
| [A6 Actor 适配与集成收口](./v0.7.0-a6-actor-adapters-and-integration.md) | Planned | 切换 Alice、Passive、管理和 MTP 消费者，完成稳定装配、旧旁路退出与 shutdown 验收 |
| [v0.7.0 计划 B：外部记忆服务与 Actor 交互契约](./v0.7.0-external-memory-service-and-actor-interaction.md) | Planned | 建立被动对话与主动资源交互的外部协议，明确身份、来源、领域提交与结果查询；以无 Alice 的参考客户端验收，并用历史样例验证后续导入的契约边界 |
| [话题折叠、Actor 上下文与原始证据统一改造](./topic-folding-context-and-raw-evidence.md) | Planned / 占位 | 独立里程碑，统筹话题折叠算法重构、原始证据和长 turn 上下文两份 Idea；详细设计与发布版本待补齐 |

默认按 A1 → A2-P → A2 → A3 → A4 → A5 → A6 **严格线性**推进（2026-09-25 重排：能力层骨架由 A2 交付，A3/A4 的能力切片依赖该骨架，取消 A3 并行资格），完整依赖见[协调入口第 3 节](./v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md#3-依赖图与执行顺序)。A2-P 交付数据/持久化行为，A2 交付运行时与读取能力面，A3/A4 交付各自能力切片，A5 收敛两层目录与职责退出。A2/A5 的 Active 承接既有工作，不代表新增范围已经完成；A2-P 不重编号后续计划，旧编号迁移表见协调入口第 1 节。

全项目时间语义计划是可在 A1 后独立推进的跨子系统治理工作；A2-P 只依赖其中的 utils 时间工具和 UTC 字段契约，完整运行时迁移不改变 A 系列的业务依赖顺序。

2026-09-23 边界冻结：新增[计划 A 边界宪章](./v0.7.0-plan-a-boundary-charter.md)，以管护权/反转/用途三判据与"独立工作"契约重裁 Workspace 与 Patchouli 边界，cache/pending registry/resolver 归 workspace runtime，Patchouli 不增不减。同日完成宪章采纳与 §10 联动修订（协调入口、A2/A3/A4/A5/A6、AE2 措辞），宪章生效为计划 A 家族边界裁决唯一理由源；A1 返工项（operation 授权从 Patchouli application 层迁至 workspace 能力边界）的实施归属由 A2-0 裁定。canonical 变更事件契约在 A2-1 交付时登记入 routes-and-events。

2026-09-19 状态更新：A1 已完成实施、验收、代码审查与文档收口并归档为 [v0.7.0 A1 Workspace 访问边界与授权（归档）](../archive/plans/v0.7.0-a1-workspace-access-boundary.md)。当前事实入口：[Workspace 架构](../architecture/workspace.md)第 4 节（统一认证网关、两类登记、guard 签发生命周期与逐次行为授权）、[错误模型](../contracts/error-model.md)第 4.4 节、[子系统公共契约](../contracts/subsystem-contracts.md)第 3.5 节与 [ADR-0005](../architecture/decisions/0005-unified-actor-authentication-and-workspace-authorization.md)。真实生产入口切换、shutdown 关闭时机和兼容分支退出仍由 [A6](./v0.7.0-a6-actor-adapters-and-integration.md) 完成；附件上传的 scope 一致性缺陷单独追踪于 [Todo](../todo/workspace-asset-upload-access-scope-mismatch.md)。

A1–A5 可独立组合验收，A6 负责真实消费者与生产收口，A 系列不等待 B 外部客户端。三方调用示例集中在 [A5](./v0.7.0-a5-patchouli-unified-api.md)；Topic 生命周期只在 [A3 第 4 节](./v0.7.0-a3-conversation-session-and-topic-projection.md#4-topic-路由指令与交接)定义，完整折叠算法仍归[专项占位计划](./topic-folding-context-and-raw-evidence.md)。B 按 A1–A5 实际交付能力映射外部协议；A/B 共同构成 v0.7.0 发布范围。具体 harness connector、执行基座与完整历史导入仍按 ROADMAP 后续排期推进。

已完成计划与实施历史：

v0.7.0 的缓存所有权与 ADR-0004 的版本适用关系现以[计划 A 边界宪章](./v0.7.0-plan-a-boundary-charter.md)裁定为准；以下归档记录仍描述 v0.6.2 基线。完整引用解析见 [A4 第 4.1 节](./v0.7.0-a4-pending-memory-intents.md#41-共同引用读取与-alias-resolver-归属)，Profile 定义/执行分离见 [A2 第 2.3 节](./v0.7.0-a2-workspace-resource-reads-and-caches.md#23-profile-读取)，正式 ADR 替代须在 A6 联合验收收尾后进行。

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
