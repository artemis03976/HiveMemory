---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-09-05
---

# Plans

本目录只存放已经绑定明确版本或里程碑、能够独立实施和验收，但尚未完全成为当前事实的功能、重构与迁移计划。已完成的计划移入 [Archived Plans](../archive/plans/README.md)，不在此目录继续作为活动实施入口。

当前没有处于实施中的 Workspace W0 计划。W0 已完成并归档；当前行为以 [Workspace 架构](../architecture/workspace.md) 为事实入口，实施细节和阶段验收保留在历史归档中。当前已建立的后续重构计划如下：

| Plan | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.6.2 Identity 投影收敛](v0.6.2-identity-projection-cleanup.md) | Planned | 在 W0 身份契约上消除裸 `user_id` 兼容投影：应用服务入口统一 `IdentityScope`、`InteractionTurnSnapshot` actor 字段值对象化、读侧兼容属性收口；W0 裁定的存储 legacy 兼容保留至历史数据转换事项 |
| [v0.6.2 W0 Workspace MVP（归档）](../archive/plans/v0.6.2-workspace-mvp.md) | Archived | 已完成 `WorkspaceIdentity`、端到端 scope、双 Workspace 隔离、进程内 WorkspaceAssetStore、两级状态机和 SemanticBuffer binding；当前事实见 [Workspace 架构](../architecture/workspace.md) |

`v0.6.2` 的 Workspace 工作拆分为 W0/W1 两个独立交付切片。W0 已完成并归档；W1 Chat Attachments 尚未建立正式 Plan，必须把 W0 的稳定公共契约作为硬前置，再独立规划上传、解析、Context Compiler、可选 ContextAttachmentUse 明细和 Artifact promotion。历史数据批量转换也不在 W0 主链路中执行，需要未来另立脚本或 Plan。

[Workspace MVP 初步设计](../ideas/workspace-mvp-chat-attachments-design.md)继续保存 W1 Chat Attachments 的设计推导和开放问题；W0 的历史实施顺序、代码落点、兼容策略、删除/lease/settle 矩阵和测试出口见归档 Plan，当前事实不再由 Plan 或 Idea 承担。

## 准入规则

- 必须填写明确 `target` 或在标题/范围中绑定当前版本；
- 必须具有独立目标、非目标、迁移方式、测试和验收出口；
- 未排期治理主题进入 [Governance](../governance/README.md)；
- 尚需验证价值、样本或所有权的候选进入 [Ideas](../ideas/README.md)；
- 范围较小的缺陷和技术债进入 [Todo](../todo/README.md)；
- 已完成或被替代的实施稿进入 [Archived Plans](../archive/plans/README.md)。
