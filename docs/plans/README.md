---
title: Plans
status: current
owner: project
scope: implementation-plan-navigation-by-version
last_reviewed: 2026-11-10
---

# Plans

本目录存放已经绑定版本或里程碑、能够独立实施和验收，但尚未成为当前事实的计划。本索引是**版本内的计划导航**：按版本列出计划的状态与链接，不记录设计决定。设计讨论见 [Ideas](../ideas/README.md)，版本规划摘要见 [ROADMAP](../ROADMAP.md)，已完成的计划见 [Archived Plans](../archive/plans/README.md)。

## v0.7.3

版本内容已收尾（2026-11-04），规范版本调整为 0.7.3，发布标签 v0.7.3 在合并后创建；版本目标的逐条判定、内容核对、未纳入的事项与验证结果见[收尾审计](../archive/plans/v0.7.3-release-closeout-audit.md)。版本内的三份计划（话题内主动写入优先、生成任务失败带原因码、主动写入沿用 agent 的标题）与收尾审计登记在 [Archived Plans](../archive/plans/README.md#v073)。

v0.7.3 之后的版本目标尚未确定，当前没有绑定版本的计划。

## 已发布的版本

v0.7.2（Git tag `v0.7.2`，2026-11-01）、v0.7.1（Git tag `v0.7.1`，2026-10-30）、v0.7.0（Git tag `v0.7.0`，2026-10-20）与更早版本的计划都已归档，按版本登记在 [Archived Plans](../archive/plans/README.md)；版本目标与摘要见 [ROADMAP](../ROADMAP.md) 第 2、3 节，逐项核对见各版本的收尾审计。

## 未绑定版本

- [Electron 桌面客户端](./desktop-client.md)：Active（2026-11-10），v1.0 范围。

前端 UI 优化与其余空入口接线另写计划，在桌面计划之后（owner，2026-11-10）。身份与访问主线是版本之外的工作，后续方向见[身份与访问体系 Idea](../ideas/identity-and-access-model.md)。

退出时保存待生成的记忆工作（2026-11-10）、主配置的个人覆盖与热更新、访问登记的运行时管理（2026-11-09）、Claude Code 唤醒后的续作导入、dsh 插件并入统一插件目录、pi 插件接入与 Codex 插件接入（2026-11-08）、全局运行页、Runtime Event 体系完善（2026-11-07）、交互提交收束与话题按需创建、WorkspaceAsset 的归属拆分、Import Bus 的授权移入能力层与统一身份授权入口（2026-11-06）、HTTP 管理面的管理凭据（2026-11-05）、此前的 DeepSeek Harness 插件接入、模型选择统一（2026-11-01）与 dsh 子 Agent 会话不再独立导入（2026-11-02）已归档，见 [Archived Plans](../archive/plans/README.md)。观测与日志是版本之外的工作，其余工作包以治理主题跟踪，见[日志与运行时事件](../governance/observability/logs-and-events.md)。

## 准入规则

计划的准入条件、内容要求、计划之间的关系与归档去向见[文档治理规范](../DOCUMENTATION.md)第 8.3、10 节；本索引只做导航，不记录设计决定。
