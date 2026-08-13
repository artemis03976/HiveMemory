---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-08-13
---

# Plans

本目录只存放已经绑定明确版本或里程碑、能够独立实施和验收，但尚未完全成为当前事实的功能、重构与迁移计划。

当前计划：

- [v0.6.1 Local Work Queue Runtime](./v0.6.1-local-work-queue-runtime.md)：以多 lane 的本地工作队列收敛 interaction submission 与 memory generation 的机械生命周期，并完成 Active/Passive 共用 submission lane 的迁移。

## 准入规则

- 必须填写明确 `target` 或在标题/范围中绑定当前版本；
- 必须具有独立目标、非目标、迁移方式、测试和验收出口；
- 未排期治理主题进入 [Governance](../governance/README.md)；
- 尚需验证价值、样本或所有权的候选进入 [Ideas](../ideas/README.md)；
- 范围较小的缺陷和技术债进入 [Todo](../todo/README.md)；
- 已完成或被替代的实施稿进入 [Archived Plans](../archive/plans/README.md)。
