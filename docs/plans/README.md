---
title: Plans
status: current
owner: project
scope: implementation-plan-navigation-by-version
last_reviewed: 2026-10-16
---

# Plans

本目录存放已经绑定版本或里程碑、能够独立实施和验收，但尚未成为当前事实的计划。本索引是**版本内的计划导航**：按版本列出计划的状态、顺序与链接，不记录设计决定。设计讨论见 [Ideas](../ideas/README.md)，版本规划摘要见 [ROADMAP](../ROADMAP.md)，已完成的计划见 [Archived Plans](../archive/plans/README.md)。

## v0.7.0

范围已决定（owner，2026-09-27）：完全完成项目向新架构的演进，使 workspace 体系在项目架构中稳定存在；首条迁移流程为 Alice 的 chat 链路（M-3），以 Alice 在新架构下跑通、各流程协作无误作为验证；版本目标见 [ROADMAP](../ROADMAP.md) 第 4.3 节。v0.7.0 按小批量逐次实施：同一方向同时只有一份生效计划，完成一批、归档一批，再建立下一批的计划。

| 顺序 | 方向 | 状态 | 入口 |
|:---:|:---|:---|:---|
| 1 | 任务进程表与任务请求唯一注册入口 | 已完成：第一至第五批均已实施归档（[落位与进程标识](../archive/plans/v0.7.0-task-process-table.md)、[prepare 拆分与 CPU 输入清单](../archive/plans/v0.7.0-task-process-prepare-split.md)、[结算阶段的中立输入](../archive/plans/v0.7.0-task-process-finalize-neutral-input.md)、[CPU 端口与测试 CPU](../archive/plans/v0.7.0-task-process-cpu-port.md)、[命令只解析不执行](../archive/plans/v0.7.0-task-process-command-parse-only.md)）；Topic 不再承担对话上下文与写入意图分别归外部会话与写入意图迁移两个方向（话题预建于 2026-10-13 决定保留） | 背景：[任务进程 Idea](../ideas/task-process-table-and-registration-entry.md) |
| 1 之后 | A1 访问边界返工 | 已完成：2026-10-04 实施归档；与 workspace 相关的 HTTP 请求都经认证网关，两阶段认证与两阶段授权按身份与访问体系 Idea 的第一批落地 | [归档计划](../archive/plans/v0.7.0-a1-access-boundary-rework.md)；当前事实见 [Workspace 架构](../architecture/workspace.md)第 4 节 |
| 与 A1 返工同步 | 身份与访问体系 | 已完成：第一批随 A1 返工完成；第二批于 2026-10-04 验收归档，实现在分支 `refactor/identity-access-batch-2` 提交（`b2c7aee`，审查后调整 `773fe4c`、`6af6e72`），同日经 PR #107 合并至 master（`5660fed`）；WorkspaceAssetStore 内部接口拆分明确暂缓，由 Todo 跟踪 | [第二批归档计划](../archive/plans/v0.7.0-identity-access-batch-2.md)；背景：[Idea](../ideas/identity-and-access-model.md)；[资产 Todo](../todo/workspace-asset-ownership-identity-split.md) |
| A1 返工之后 | 写入意图登记迁入 workspace 与读取缓存失效 | 已完成：2026-10-07 实施、验收与 code review 后归档，2026-10-09 经 PR #109 合并至 master（`0fcf118`）；workspace 唯一登记、统一引用解析、canonical 变更失效及进程操作通道落地 | [归档计划](../archive/plans/v0.7.0-intent-registry-and-read-cache.md)；当前事实：[Workspace 架构](../architecture/workspace.md) |
| 上一行之后 | 写入意图迁移·第 2 步 | 已完成（2026-10-13，三个阶段均经实现、验收与 code review 后归档）：写入意图在 actor 操作内经 `patchouli.public.memory_intent.submit` 实时派发，生成队列接纳后返回 ACK；收尾不再派发物化，提交后不受失败与取消影响；先于外部会话与 Topic 投影实施 | [归档计划](../archive/plans/v0.7.0-intent-inflight-dispatch.md)；后续方向：[Idea](../ideas/pending-intent-migration.md)（Candidate） |
| 写入意图登记与读取缓存失效之后 | Alice 的能力层调用迁移 | 已完成（2026-10-09）：三阶段实现、验收与 code review 均完成并归档；操作请求/执行凭据、SEARCH/引用记录/CALL Profile 迁移及 CPU 过渡身份删除已落地；子线程共用主线程凭据，独立身份留给执行线程层 | [归档计划](../archive/plans/v0.7.0-alice-capability-migration.md) |
| 线程 context 取代执行凭据之后 | 执行线程层 | 已完成（2026-10-11）：三阶段实现、验收与 code review 均完成并归档；子线程以 CALL 目标 agent 的身份运行，按它自己的白名单与可见性授权，写入记在它名下；派生请求 `DeriveCallThreadRequest` 取代 `ResolveCallTargetRequest`，进程以被调用方准入后派生并登记，CPU 驱动以通用的 `ThreadOperationSubmitter` 为子帧绑定提交函数，全部线程 context 在 CPU 执行结束时一起撤销 | [归档计划](../archive/plans/v0.7.0-execution-thread-layer.md)；当前事实：[Workspace 架构](../architecture/workspace.md#54-写入意图线程-context-与操作入口)第 4.3、5.4 节 |
| Alice 的能力层调用迁移之后 | 能力层分层：root API 与 actor API | 已完成（2026-10-10）：三阶段实现、验收与 code review 均完成并归档；capability 目录分为只供 `system` 的 root API 与 actor API，actor 操作对齐 READ、SEARCH、RUN、WRITE、UPDATE 与只有 Alice 使用的 CALL，进程阶段调用改为不检查 actor 操作的阶段授权 | [归档计划](../archive/plans/v0.7.0-capability-surface-split.md)；当前事实：[Workspace 架构](../architecture/workspace.md#45-三个调用面actor-apiroot-api-与任务进程的阶段调用)第 4.5 节 |
| 能力层分层之后 | Agent 操作白名单（原“Agent Profile 的权限并入 operation 控制”） | 已完成（2026-10-11）：三阶段实现、验收与 code review 均完成并归档；访问登记的白名单成为 Agent 权限的唯一来源，Profile 的两个 allow 字段删除，Alice 把输入清单的操作快照翻译为 MTP 动词，系统工具不设权限，子 agent 继承父 agent 的动词集合；运行时编辑并入配置热更新 | [归档计划](../archive/plans/v0.7.0-agent-operation-whitelist.md)；当前事实：[MTP 契约](../contracts/mtp.md#4-权限)第 4 节 |
| Agent 操作白名单之后 | 线程 context 取代执行凭据 | 已完成（2026-10-11）：三阶段实现、验收与 code review 均完成并归档；进程 context 服务于全部四个阶段，主线程的线程 context 在进入 CPU 执行阶段时从进程 context 派生、CPU 执行结束即撤销；执行凭据、凭据表与入口的拆参 handler 删除，入口只按请求类型分派到 actor API；子线程沿用主线程 context，独立 context 属于执行线程层 | [归档计划](../archive/plans/v0.7.0-thread-context.md)；当前事实：[Workspace 架构](../architecture/workspace.md#54-写入意图线程-context-与操作入口)第 4.3、5.4 节 |
| — | 总线路由类型化与 mypy 门禁 | 已完成：2026-10-12 三阶段实现、验收与 code review 后归档，分支 `fix/mypy-typing-baseline` 形成稳定基线；未发布 | [归档计划](../archive/plans/v0.7.0-typed-bus-routes.md)；当前事实见[运行时机制](../components/runtime-and-bus.md) |
| 写入意图实时派发之后 | 外部会话消息的接收与 Topic 投影 | 已完成（2026-10-15，四阶段均经实现、验收与 code review 后归档）：会话资源、Alice 全量历史、Gateway 最近 N 轮、部分输出保存与 finalize 保护、前端会话模型 | [归档计划](../archive/plans/v0.7.0-conversation-session.md)；背景：[Idea](../ideas/external-session-and-topic-projection.md)（原 A3） |
| 外部会话之后 | Alice 会话压缩 | v0.7.0 内单独一份计划（2026-10-15 由约 v0.7.1 提前）：会话历史的增长不再让 Alice 的请求超出模型窗口，较早轮次先做工具结果减重、再按需接力摘要；单轮执行由轮内预算守卫拦截，超窗以可区分的错误结束（减重与守卫随压缩一起实现，owner，2026-10-16）。尚未建立计划，待 AC-1（压缩结果的存放位置）与 AC-2（`/compact` 是否进入本计划）决定后编写 | 背景：[短期记忆与上下文压缩](../ideas/short-term-memory-and-context-compaction.md#5-alice-会话-compact-候选机制)第 5、8 节；[会话 `/compact` 指令 Todo](../todo/conversation-compact-command.md) |

- Alice 会话压缩的技术调研见 [Agent Harness 会话压缩技术调研](../ideas/agent-context-compaction-research.md)（2026-10-16；背景材料，尚未形成实施计划）。
- 不在 v0.7.0：执行环境的选择（[执行单元 Idea](../ideas/execution-unit-thread-and-environment.md#2-开放问题) T-7、T-8）随后续版本实施，不一定是 v0.7.1（2026-10-06）；外部 Actor 的真实接入（[Idea](../ideas/external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27)，原计划 B）分为两种接入模式，controller 模式在 v0.7.1（[ROADMAP](../ROADMAP.md) 第 4.4.2 节），plugin 模式在其后的 v0.7.x；Import Bus（现有 Passive Ingress 链路）逐步演进为独立功能，不在 v0.7.0 计划内；
- 已完成：[A1 访问边界](../archive/plans/v0.7.0-a1-workspace-access-boundary.md)、[A2-P 记忆内容版本与 Lifecycle](../archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md)、[全项目时间语义与可控时钟](../archive/plans/v0.7.0-time-semantics-and-controllable-clock.md)（均已归档）；
- 已删除：A2（未完成部分）、A5、A6、WRX-0 清单、计划 A 协调入口与边界宪章，删除前最后版本见 commit `dda9d9d`；处置记录见 commit `6add90f` 版本的《Workspace 网络与任务进程架构》Idea 第 5.5 节。

## 未绑定版本

| 计划 | 状态 | 说明 |
|:---|:---:|:---|
| 暂无 | — | 原占位计划“话题折叠、Actor 上下文与原始证据统一改造”于 2026-10-01 退回 Idea 后删除（删除前最后版本见 commit `74b5056`）：后台部分并入 [Page Folding Raw Evidence](../ideas/PatchouliPageFoldingRawEvidenceDesign.md)，前台部分并入[长时间运行 Agent 的 Turn 内上下文折叠](../ideas/long-running-agent-intra-turn-context-folding.md) |

## 已完成的计划

| Plan | 状态 | 结果与事实入口 |
|:---|:---:|:---|
| [v0.7.0 写入意图实时派发](../archive/plans/v0.7.0-intent-inflight-dispatch.md) | Archived（2026-10-13） | actor 操作内派发、队列接纳后 ACK；删除 completed 认领与运行失败撤回，交互和写入独立，任务经事件观察。当前事实见 [Workspace 架构](../architecture/workspace.md)、[MTP](../contracts/mtp.md)与 [生成](../patchouli/generation.md) |
| [v0.7.0 Agent 操作白名单](../archive/plans/v0.7.0-agent-operation-whitelist.md) | Archived（2026-10-11） | 访问登记的 actor 操作白名单成为 Agent 权限的唯一来源；输入清单携带操作快照，Alice 翻译为 MTP 动词驱动提示词与 frame 策略；Profile 不再承载权限，系统工具不设权限，子 agent 继承父 agent。当前事实见 [MTP 契约](../contracts/mtp.md)、[Workspace 架构](../architecture/workspace.md)与 [System 配置](../system/configuration.md) |
| [v0.7.0 能力层分层：root API 与 actor API](../archive/plans/v0.7.0-capability-surface-split.md) | Archived（2026-10-10） | root 授权只接受 `system`，取代 `management.*` 与 `task.observe`；六项 actor 操作与请求一一对应，撤回意图改为入口内部补偿；任务进程的阶段调用不检查 actor 操作。当前事实见 [Workspace 架构](../architecture/workspace.md)第 4.5 节、[子系统契约](../contracts/subsystem-contracts.md)与 [System 配置](../system/configuration.md) |
| [v0.7.0 Alice 的能力层调用迁移](../archive/plans/v0.7.0-alice-capability-migration.md) | Archived（2026-10-09） | 操作请求、执行凭据与单一入口；SEARCH、引用记录和 CALL Profile 经能力层；CPU/Alice 只携观测标签。当前事实见 [Workspace 架构](../architecture/workspace.md)、[子系统契约](../contracts/subsystem-contracts.md)与 [Alice](../alice/README.md) |
| [v0.7.0 写入意图登记迁入 workspace 与读取缓存失效](../archive/plans/v0.7.0-intent-registry-and-read-cache.md) | Archived（2026-10-07） | workspace 唯一登记与读取视图、进程绑定操作通道、按进程认领/取消、canonical 变更同步失效；当前事实见 [Workspace 架构](../architecture/workspace.md)、[MTP 契约](../contracts/mtp.md)与 [System 应用服务](../system/application-services.md) |
| [v0.7.0 身份与访问体系第二批](../archive/plans/v0.7.0-identity-access-batch-2.md) | Archived（2026-10-04） | Patchouli 内部与后台任务分开携带归属与发起者，SETTLE 四触发使用 system，finalize/cleanup 重新阶段授权，actor 移除 session_id；当前事实见 [Workspace 架构](../architecture/workspace.md)、[子系统公共契约](../contracts/subsystem-contracts.md)与 [Patchouli 生成](../patchouli/generation.md)；[WorkspaceAsset 拆分](../todo/workspace-asset-ownership-identity-split.md)暂缓 |
| [v0.7.0 A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md) | Archived（2026-10-04） | 与 workspace 相关的 HTTP 请求都经统一认证网关；访问 context 改为密封的运行时凭据，认证一侧与操作授权者互不依赖；注册入口先注册后运行，进程表登记任务进程，句柄按对象身份判定有效；Patchouli 与 Gateway 只接收 `IdentityScope`；两个登记文件与用户级记录；当前事实见 [Workspace 架构](../architecture/workspace.md)第 4 节、[错误模型](../contracts/error-model.md)、[System 应用服务](../system/application-services.md)与 [System 配置](../system/configuration.md) |
| [v0.7.0 任务进程：命令只解析不执行](../archive/plans/v0.7.0-task-process-command-parse-only.md) | Archived（2026-10-01） | Gateway 的命令结果只携带解析结果，删除命令分发与执行，任务进程产生“暂不可用”的命令终态；当前事实见 [Gateway 全局命令](../gateway/commands.md) |
| [v0.7.0 任务进程：CPU 端口与测试 CPU](../archive/plans/v0.7.0-task-process-cpu-port.md) | Archived（2026-10-01） | `workspace.contracts` 定义 `CPUPort` 与 `CPUExecutionResult`，任务进程只经注入的端口调用 CPU；Alice 统一流式与非流式入口并实现端口；测试 CPU 跑通任务进程；当前事实见[子系统公共契约](../contracts/subsystem-contracts.md)与 [System 应用服务](../system/application-services.md) |
| [v0.7.0 任务进程：结算阶段的中立输入](../archive/plans/v0.7.0-task-process-finalize-neutral-input.md) | Archived（2026-09-30） | 任务进程封口交互记录 `InteractionPayload`，Patchouli finalize 改为接收 `PreparedAgentRun` 与 `InteractionPayload` 并原样提交，不再接收 `AgentRunResult`；当前事实见[子系统公共契约](../contracts/subsystem-contracts.md)与 [System 应用服务](../system/application-services.md) |
| [v0.7.0 任务进程：prepare 拆分与 CPU 输入清单](../archive/plans/v0.7.0-task-process-prepare-split.md) | Archived（2026-09-29） | Patchouli prepare 只做话题与检索；任务进程完成 CPU 分配（Profile 解析、附件租借与编译、记忆编译），输入清单 `CPUInputManifest` 位于 `workspace.contracts`；当前事实见 [System 应用服务](../system/application-services.md)与[子系统公共契约](../contracts/subsystem-contracts.md) |
| [v0.7.0 任务进程表：落位与进程标识](../archive/plans/v0.7.0-task-process-table.md) | Archived（2026-09-28） | 进程表与 chat 四阶段编排迁入 `workspace.process`、`process_id` 统一进程标识、取消只在 Gateway 与 Actor 执行响应；当前事实见 [System 应用服务](../system/application-services.md) |
| [v0.7.0 A2-P 记忆内容版本与 Lifecycle 状态重构](../archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md) | Archived（2026-09-24） | 完整版本历史、meta.lifecycle 聚合、受控局部更新与 schema 2.1 迁移 |
| [全项目时间语义与可控时钟统一](../archive/plans/v0.7.0-time-semantics-and-controllable-clock.md) | Archived（2026-09-24） | Memory 域 UTC 业务时间、四时间字段职责、局部 now 注入与 TimeFormatter 契约；全项目收口转为 Idea |
| [v0.7.0 A1 Workspace 访问边界与授权](../archive/plans/v0.7.0-a1-workspace-access-boundary.md) | Archived | 统一认证网关、两类登记、guard 签发生命周期与逐次行为授权；当前事实见 [Workspace 架构](../architecture/workspace.md)第 4 节 |
| [v0.6.2 W0 Workspace MVP](../archive/plans/v0.6.2-workspace-mvp.md) | Archived | `WorkspaceIdentity`、端到端 scope、双 Workspace 隔离、进程内 WorkspaceAssetStore、两级状态机和 SemanticBuffer binding；当前事实见 [Workspace 架构](../architecture/workspace.md) |
| [v0.6.2 Identity 投影收敛](../archive/plans/v0.6.2-identity-projection-cleanup.md) | Archived | 服务入口统一 `IdentityScope`、`InteractionTurnSnapshot` actor 值对象化、读侧兼容属性收口与 `system` 保留 actor 语义；当前事实见 [Workspace 架构](../architecture/workspace.md) 与 [System 应用服务](../system/application-services.md) |
| [v0.6.2 Workspace Runtime 聚合与缓存所有权迁移](../archive/plans/v0.6.2-workspace-runtime-cache-migration.md) | Archived | Workspace-aware cache key；派生缓存所有权曾归还 AliceRuntime，v0.7.0 引用读取缓存已迁入 workspace；当前事实见 [System 组合根](../system/composition.md)与 [Alice](../alice/README.md) |
| [v0.6.2 W1 Chat Attachments](../archive/plans/v0.6.2-w1-chat-attachments.md) | Archived | 附件上传、确定性解析、Chat 选择与 lease、AttachmentCompiler、Topic binding 与按需 Artifact promotion；当前事实见 [Chat 附件链路](../system/attachments.md) |

其余已归档计划见 [Archived Plans](../archive/plans/README.md)。

## 准入规则

- 必须填写明确 `target` 或在标题/范围中绑定当前版本；
- 必须具有独立目标、非目标、迁移方式、测试和验收出口；
- 计划之间的关系满足[文档治理规范](../DOCUMENTATION.md)第 8.3 节的三条约束：同一目标方向只有一份生效计划、不设总分结构；设计依据只来自事实文档、代码、ADR、已归档计划与背景 Idea；计划之间只有整份粒度的顺序依赖；
- 本索引只做导航，不记录设计决定；
- 未排期治理主题进入 [Governance](../governance/README.md)；尚需验证价值、样本或所有权的候选进入 [Ideas](../ideas/README.md)；范围较小的缺陷和技术债进入 [Todo](../todo/README.md)；
- 已完成或被替代的实施稿进入 [Archived Plans](../archive/plans/README.md)；未完成实施即作废的计划直接删除，方向保留但前提改变的未实施计划退回 Idea，入链与遗留事项按文档治理规范第 8.4、10 节处理。
