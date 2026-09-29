---
title: Plans
status: current
owner: project
scope: implementation-plan-navigation-by-version
last_reviewed: 2026-09-29
---

# Plans

本目录存放已经绑定版本或里程碑、能够独立实施和验收，但尚未成为当前事实的计划。本索引是**版本内的计划导航**：按版本列出计划的状态、顺序与链接，不记录设计决定。设计讨论见 [Ideas](../ideas/README.md)，版本规划摘要见 [ROADMAP](../ROADMAP.md)，已完成的计划见 [Archived Plans](../archive/plans/README.md)。

## v0.7.0

范围已决定（2026-09-27，[总 Idea](../ideas/workspace-network-task-process-architecture.md#61-已决定事项) M-5）：完全完成项目向新架构的演进，使 workspace 体系在项目架构中稳定存在；首条迁移流程为 Alice 的 chat 链路（M-3），以 Alice 在新架构下跑通、各流程协作无误作为验证；版本目标见 [ROADMAP](../ROADMAP.md) 第 4.3 节。v0.7.0 按小批量逐次实施：同一方向同时只有一份生效计划，完成一批、归档一批，再建立下一批的计划。

| 顺序 | 方向 | 状态 | 入口 |
|:---:|:---|:---|:---|
| 1 | 任务进程表与任务请求唯一注册入口 | 已完成归档：第一批[任务进程表：落位与进程标识](../archive/plans/v0.7.0-task-process-table.md)、第二批[prepare 拆分与 CPU 输入清单](../archive/plans/v0.7.0-task-process-prepare-split.md)；后续批次未立项 | 背景：[任务进程 Idea](../ideas/task-process-table-and-registration-entry.md) |
| 1 之后 | A1 访问边界返工 | 在任务进程表计划完成、已有稳定入口之后接入 | [Todo](../todo/a1-access-boundary-rework.md) |
| — | 外部会话消息的接收与 Topic 投影 | v0.7.0 内完成，Alice 为第一个使用者；包括前端回归 session 模型与新建、恢复两个会话操作（Alice 的压缩约在 v0.7.1）；与其他方向的先后未定 | [Idea](../ideas/external-session-and-topic-projection.md)（原 A3） |
| — | 写入意图（PendingAtom）体系的迁移 | v0.7.0 内完成，分两步；与外部会话与 Topic 投影的先后均可 | [Idea](../ideas/pending-intent-migration.md)（原 A4） |

- 不在 v0.7.0：外部 Actor 的真实接入（[Idea](../ideas/external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27)，原计划 B）分为两种接入模式，controller 模式在 v0.7.1（[ROADMAP](../ROADMAP.md) 第 4.4.2 节），plugin 模式在其后的 v0.7.x；Import Bus（现有 Passive Ingress 链路）逐步演进为独立功能，不在 v0.7.0 计划内；
- 已完成：[A1 访问边界](../archive/plans/v0.7.0-a1-workspace-access-boundary.md)、[A2-P 记忆内容版本与 Lifecycle](../archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md)、[全项目时间语义与可控时钟](../archive/plans/v0.7.0-time-semantics-and-controllable-clock.md)（均已归档）；
- 已删除：A2（未完成部分）、A5、A6、WRX-0 清单、计划 A 协调入口与边界宪章，删除前最后版本见 commit `dda9d9d`；处置记录见总 Idea 第 6.1 节。

## 未绑定版本

| 计划 | 状态 | 说明 |
|:---|:---:|:---|
| [话题折叠、Actor 上下文与原始证据统一改造](./topic-folding-context-and-raw-evidence.md) | Planned / 占位 | 统筹话题折叠算法重构、原始证据和长 turn 上下文两份 Idea；详细设计与发布版本待补齐 |

## 已完成的计划

| Plan | 状态 | 结果与事实入口 |
|:---|:---:|:---|
| [v0.7.0 任务进程：prepare 拆分与 CPU 输入清单](../archive/plans/v0.7.0-task-process-prepare-split.md) | Archived（2026-09-29） | Patchouli prepare 只做话题与检索；任务进程完成 CPU 分配（Profile 解析、附件租借与编译、记忆编译），输入清单 `CPUInputManifest` 位于 `workspace.contracts`；当前事实见 [System 应用服务](../system/application-services.md)与[子系统公共契约](../contracts/subsystem-contracts.md) |
| [v0.7.0 任务进程表：落位与进程标识](../archive/plans/v0.7.0-task-process-table.md) | Archived（2026-09-28） | 进程表与 chat 四阶段编排迁入 `workspace.process`、`process_id` 统一进程标识、取消只在 Gateway 与 Actor 执行响应；当前事实见 [System 应用服务](../system/application-services.md) |
| [v0.7.0 A2-P 记忆内容版本与 Lifecycle 状态重构](../archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md) | Archived（2026-09-24） | 完整版本历史、meta.lifecycle 聚合、受控局部更新与 schema 2.1 迁移 |
| [全项目时间语义与可控时钟统一](../archive/plans/v0.7.0-time-semantics-and-controllable-clock.md) | Archived（2026-09-24） | Memory 域 UTC 业务时间、四时间字段职责、局部 now 注入与 TimeFormatter 契约；全项目收口转为 Idea |
| [v0.7.0 A1 Workspace 访问边界与授权](../archive/plans/v0.7.0-a1-workspace-access-boundary.md) | Archived | 统一认证网关、两类登记、guard 签发生命周期与逐次行为授权；当前事实见 [Workspace 架构](../architecture/workspace.md)第 4 节 |
| [v0.6.2 W0 Workspace MVP](../archive/plans/v0.6.2-workspace-mvp.md) | Archived | `WorkspaceIdentity`、端到端 scope、双 Workspace 隔离、进程内 WorkspaceAssetStore、两级状态机和 SemanticBuffer binding；当前事实见 [Workspace 架构](../architecture/workspace.md) |
| [v0.6.2 Identity 投影收敛](../archive/plans/v0.6.2-identity-projection-cleanup.md) | Archived | 服务入口统一 `IdentityScope`、`InteractionTurnSnapshot` actor 值对象化、读侧兼容属性收口与 `system` 保留 actor 语义；当前事实见 [Workspace 架构](../architecture/workspace.md) 与 [System 应用服务](../system/application-services.md) |
| [v0.6.2 Workspace Runtime 聚合与缓存所有权迁移](../archive/plans/v0.6.2-workspace-runtime-cache-migration.md) | Archived | Workspace-aware cache key；派生缓存所有权随后归还 AliceRuntime；当前事实见 [System 组合根](../system/composition.md)与 [Alice](../alice/README.md) |
| [v0.6.2 W1 Chat Attachments](../archive/plans/v0.6.2-w1-chat-attachments.md) | Archived | 附件上传、确定性解析、Chat 选择与 lease、AttachmentCompiler、Topic binding 与按需 Artifact promotion；当前事实见 [Chat 附件链路](../system/attachments.md) |

其余已归档计划见 [Archived Plans](../archive/plans/README.md)。

## 准入规则

- 必须填写明确 `target` 或在标题/范围中绑定当前版本；
- 必须具有独立目标、非目标、迁移方式、测试和验收出口；
- 计划之间的关系满足[文档治理规范](../DOCUMENTATION.md)第 8.3 节的三条约束：同一目标方向只有一份生效计划、不设总分结构；设计依据只来自事实文档、代码、ADR、已归档计划与背景 Idea；计划之间只有整份粒度的顺序依赖；
- 本索引只做导航，不记录设计决定；
- 未排期治理主题进入 [Governance](../governance/README.md)；尚需验证价值、样本或所有权的候选进入 [Ideas](../ideas/README.md)；范围较小的缺陷和技术债进入 [Todo](../todo/README.md)；
- 已完成或被替代的实施稿进入 [Archived Plans](../archive/plans/README.md)；未完成实施即作废的计划直接删除，方向保留但前提改变的未实施计划退回 Idea，入链与遗留事项按文档治理规范第 8.4、10 节处理。
