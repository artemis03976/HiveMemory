---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-08-18
---

# Plans

本目录只存放已经绑定明确版本或里程碑、能够独立实施和验收，但尚未完全成为当前事实的功能、重构与迁移计划。

当前尚无已经正式建立的实施 Plan。`v0.6.2` 将使用两份独立 Plan：首先实施 W0 Workspace MVP，完成 `WorkspaceIdentity`、端到端 scope、双 Workspace 隔离和进程内 WorkspaceAsset foundation；随后 W1 Chat Attachments 把 W0 已完成作为硬前置，再实现上传、解析、Context Compiler 和 Artifact promotion。

[Workspace MVP 初步设计](../ideas/workspace-mvp-chat-attachments-design.md)已经裁定 Workspace 使用 `WorkspaceIdentity(owner_user_id, workspace_key, workspace_id)` 统一持有身份，MVP 固定 `workspace_id == workspace_key`，默认值为 `main_workspace`，并且 WorkspaceAsset 只承诺进程内生命周期。W0 待 AccessContext 公共模型、现有资产迁移、WorkspaceAssetStore/lease、失败矩阵和测试出口冻结后进入本目录；DocumentArtifact promotion 失败语义不再阻塞 W0，而由 W1 的独立 Plan 裁定。

最近完成的 [v0.6.1 Local Work Queue Runtime](../archive/plans/v0.6.1-local-work-queue-runtime.md)
已归档；当前运行时事实由 [System 运行时与总线](../system/runtime-and-bus.md#3-local-work-queue-runtime)
承接，SQLite 后续由[持久化治理](../governance/reliability/durability-and-recovery.md#46-sqlite-workstore-持久化门槛与设计约束)维护。

## 准入规则

- 必须填写明确 `target` 或在标题/范围中绑定当前版本；
- 必须具有独立目标、非目标、迁移方式、测试和验收出口；
- 未排期治理主题进入 [Governance](../governance/README.md)；
- 尚需验证价值、样本或所有权的候选进入 [Ideas](../ideas/README.md)；
- 范围较小的缺陷和技术债进入 [Todo](../todo/README.md)；
- 已完成或被替代的实施稿进入 [Archived Plans](../archive/plans/README.md)。
