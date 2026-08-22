---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-08-22
---

# Plans

本目录只存放已经绑定明确版本或里程碑、能够独立实施和验收，但尚未完全成为当前事实的功能、重构与迁移计划。

当前正式实施入口为：

| Plan | 状态 | 目标结果 |
|:---|:---:|:---|
| [v0.6.2 W0 Workspace MVP](./v0.6.2-workspace-mvp.md) | Planned | 建立 `WorkspaceIdentity`、端到端 scope、双 Workspace 隔离、进程内 WorkspaceAssetStore、两级状态机和 SemanticBuffer binding |

`v0.6.2` 继续使用两份独立 Plan。当前只建立并实施 W0；W1 Chat Attachments 必须把 W0 的完成验收作为硬前置，再独立规划上传、解析、Context Compiler、可选 ContextAttachmentUse 明细和 Artifact promotion。历史数据批量转换也不在 W0 主链路中执行，而是在 W0 基本落地和双 Workspace 隔离验证通过后通过独立脚本完成。

[Workspace MVP 初步设计](../ideas/workspace-mvp-chat-attachments-design.md)继续保存 W0/W1 的设计推导和 W1 开放问题；W0 的实施顺序、代码落点、兼容策略、删除/lease/settle 矩阵和测试出口以正式 Plan 为准。

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
