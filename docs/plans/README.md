---
title: Plans
status: current
owner: project
scope: implementation-plan-navigation-by-version
last_reviewed: 2026-11-03
---

# Plans

本目录存放已经绑定版本或里程碑、能够独立实施和验收，但尚未成为当前事实的计划。本索引是**版本内的计划导航**：按版本列出计划的状态与链接，不记录设计决定。设计讨论见 [Ideas](../ideas/README.md)，版本规划摘要见 [ROADMAP](../ROADMAP.md)，已完成的计划见 [Archived Plans](../archive/plans/README.md)。

## v0.7.3

当前开发版本：主动写入与被动提取的收敛，作为过渡措施（owner，2026-11-03，[ROADMAP](../ROADMAP.md#43-v073主动写入与被动提取的收敛) 第 4.3 节）。之后的方向见候选 Idea [统一的交互记忆](../ideas/unified-interaction-memory.md)。设计讨论见[主动写入与被动提取的收敛 Idea](../ideas/generation-source-convergence.md)。

| 计划 | 状态 | 内容 |
|:---|:---|:---|
| [生成任务失败带原因码](./v0.7.3-generation-failure-reason-codes.md) | planned | 失败分类按异常给出原因码，进入工作记录、`work.failed`、`memory.task.failed` 与分类处的日志 |

话题内主动写入优先已于 2026-11-03 完成，见 [Archived Plans](../archive/plans/README.md#v073)。

尚未建立计划的方向：主动写入沿用 agent 的标题，设计见收敛 Idea 第 4.2 节。

## 已发布的版本

v0.7.2（Git tag `v0.7.2`，2026-11-01）、v0.7.1（Git tag `v0.7.1`，2026-10-30）、v0.7.0（Git tag `v0.7.0`，2026-10-20）与更早版本的计划都已归档，按版本登记在 [Archived Plans](../archive/plans/README.md)；版本目标与摘要见 [ROADMAP](../ROADMAP.md) 第 2、3 节，逐项核对见各版本的收尾审计。

## 未绑定版本

暂无。DeepSeek Harness 插件接入、模型选择统一（2026-11-01）与 dsh 子 Agent 会话不再独立导入（2026-11-02）已归档，见 [Archived Plans](../archive/plans/README.md)。观测与日志是版本之外的工作，目前以治理主题与 Todo 跟踪，见[日志与运行时事件](../governance/observability/logs-and-events.md)。

## 准入规则

计划的准入条件、内容要求、计划之间的关系与归档去向见[文档治理规范](../DOCUMENTATION.md)第 8.3、10 节；本索引只做导航，不记录设计决定。
