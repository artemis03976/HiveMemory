---
title: Archived Plans
status: current
owner: project
scope: completed-or-superseded-plans
last_reviewed: 2026-11-07
---

# Archived Plans

本目录保存已经完成或被替代、且当前事实已经合并进规范文档的实施计划。每条登记一行：归档日期、实现所在的 PR（未合并时为提交）与替代它的当前文档；实施过程、验收与决定见各计划自身的状态行与实施记录。

## v0.7.3

- [v0.7.3 版本收尾审计](./v0.7.3-release-closeout-audit.md)：2026-11-04，PR #179；[PROJECT](../../PROJECT.md)
- [主动写入沿用 agent 的标题](./v0.7.3-write-title-kept.md)：2026-11-03，PR #178；[Patchouli 生成](../../patchouli/generation.md#32-mode-b主动-write)、[MCP 记忆访问](../../contracts/mcp.md#3-工具)
- [生成任务失败带原因码](./v0.7.3-generation-failure-reason-codes.md)：2026-11-03，PR #177；[Patchouli 生成](../../patchouli/generation.md#61-失败原因码)、[事件契约](../../contracts/routes-and-events.md#44-失败与降级观测)
- [话题内主动写入优先](./v0.7.3-topic-active-write-precedence.md)：2026-11-03，PR #176；[Patchouli 生成](../../patchouli/generation.md#42-本话题的主动写入)

## 版本之外

- [Runtime Event 体系完善](./runtime-event-system.md)：2026-11-07，PR #193；[运行时事件与可观测性](../../components/observability.md#11-一次运行)、[事件契约](../../contracts/routes-and-events.md#4-runtimeevent-观测契约)
- [交互提交收束与话题按需创建](./interaction-submission-and-topic-on-demand.md)：2026-11-06，PR #186；[任务进程](../../workspace/task-process.md)、[能力层](../../workspace/capability.md)、[Import Bus](../../system/import-bus.md)
- [WorkspaceAsset 的归属拆分](./workspace-asset-ownership-split.md)：2026-11-06，PR #185；[WorkspaceAsset](../../workspace/assets.md#3-身份与归属)、[身份与授权](../../workspace/identity-and-authorization.md)
- [Import Bus 的授权移入能力层](./import-bus-authorization-in-capability-layer.md)：2026-11-06，PR #184；[Import Bus](../../system/import-bus.md#41-认证与授权)、[能力层](../../workspace/capability.md)
- [统一身份授权入口](./unified-identity-authorization-entry.md)：2026-11-06，PR #184；[身份与授权](../../workspace/identity-and-authorization.md#4-操作授权)、[认证与访问登记](../../workspace/authentication.md#3-工作-actor)
- [HTTP 管理面的管理凭据](./http-management-credential.md)：2026-11-05，PR #182；[认证与访问登记](../../workspace/authentication.md#1-两阶段认证)、[安装与启动](../../help/setup.md#43-启动后端)
- [dsh 子 Agent 会话不再独立导入，交回的消息进入父会话的一轮](./dsh-subagent-sessions.md)：2026-11-02，提交 4a16b46a；[Import Bus](../../system/import-bus.md#93-轮次的切分与映射)、[数据模型](../../architecture/data-model.md#49-交互记录turnevent-与-interactionpayload)
- [模型选择统一为 ModelSelection，组件模型只经注册表解析](./llm-config-retirement.md)：2026-11-01，PR #167；[System 配置与注册表](../../system/configuration.md#3-registry-解析)、[System 组合根](../../system/composition.md#13-llm-bundle)
- [DeepSeek Harness 插件接入](./dsh-plugin-integration.md)：2026-11-01，PR #158；[Import Bus](../../system/import-bus.md#9-deepseek-harness-插件)、[接入 DeepSeek Harness](../../help/deepseek-harness-plugin.md)

## v0.7.2

- [v0.7.2 版本收尾审计](./v0.7.2-release-closeout-audit.md)：2026-11-01，PR #156；[PROJECT](../../PROJECT.md)
- [一个话题提取多条记忆](./v0.7.2-multi-memory-extraction.md)：2026-11-01，PR #155；[Patchouli 生成](../../patchouli/generation.md#31-mode-a被动观察--archive)
- [查重与更新：先落库再合并](./v0.7.2-dedup-merge.md)：2026-10-31，PR #154；[Patchouli 生成](../../patchouli/generation.md#4-查重与合并)、[读取视图与写入意图](../../workspace/read-view-and-write-intents.md)
- [plugin 模式开放 WRITE 与 UPDATE](./v0.7.2-plugin-write-update.md)：2026-10-31，PR #152；[MCP 记忆访问](../../contracts/mcp.md)、[Import Bus](../../system/import-bus.md#47-写入时确定话题)
- [生成材料修正与 Topic 之后的原始证据保存](./v0.7.2-generation-material-and-raw-evidence.md)：2026-10-31，PR #150；[感知与短期话题](../../patchouli/perception.md)、[Patchouli 生成](../../patchouli/generation.md)

## v0.7.1

- [v0.7.1 版本收尾审计](./v0.7.1-release-closeout-audit.md)：2026-10-30，PR #148；[PROJECT](../../PROJECT.md)
- [工具解析链路重写：`TurnEvent` 作为唯一的交互原子事件](./v0.7.1-tool-call-parsing-chain.md)：2026-10-30，PR #147；[数据模型](../../architecture/data-model.md)、[感知与短期话题](../../patchouli/perception.md)
- [推理模型的空闲卸载与按需加载](./v0.7.1-model-idle-unload.md)：2026-10-29，PR #144；[Patchouli](../../patchouli/README.md#41-模型驻留与空闲卸载)
- [记忆访问：Claude Code 经 MCP 工具检索与读取记忆](./v0.7.1-memory-access-mcp.md)：2026-10-29，PR #143；[MCP 记忆访问](../../contracts/mcp.md)、[认证与访问登记](../../workspace/authentication.md)
- [Import Bus 重设计：以 Claude Code 对话回流为第一条真实路径](./v0.7.1-import-bus-redesign.md)：2026-10-21，PR #137；[Import Bus](../../system/import-bus.md)

## v0.7.0

- [v0.7.0 版本收尾审计](./v0.7.0-release-closeout-audit.md)：2026-10-19，PR #135；[PROJECT](../../PROJECT.md)
- [执行线程的正式定义](./v0.7.0-execution-thread-definition.md)：2026-10-19，PR #133；[执行线程](../../workspace/execution-threads.md)
- [任务进程拥有执行流](./v0.7.0-task-process-owned-execution.md)：2026-10-19，PR #133；[任务进程](../../workspace/task-process.md)
- [LLM 访问收敛到 infrastructure 服务接口](./v0.7.0-llm-access-port.md)：2026-10-18，PR #129；[System 组合根](../../system/composition.md#13-llm-bundle)、[Agent Runtime](../../alice/agent-runtime.md#4-模型选择与-mtp-生成器)
- [异常体系与边界错误呈现收敛](./v0.7.0-error-model-convergence.md)：2026-10-18，PR #128；[错误模型](../../contracts/error-model.md)
- [提示词编译前移至任务进程](./v0.7.0-cpu-prompt-compilation.md)：2026-10-17，PR #127；[CPU 契约](../../contracts/subsystem-contracts.md#41-cpu-端口)、[任务进程](../../workspace/task-process.md#51-主动链路)
- [Alice 与 agent_runtime 结构重构](./v0.7.0-alice-agent-runtime-restructure.md)：2026-10-17，PR #127；[Alice](../../alice/README.md)、[Agent Runtime](../../alice/agent-runtime.md)
- [Alice 会话压缩](./v0.7.0-alice-session-compaction.md)：2026-10-17，PR #126；[Agent Runtime](../../alice/agent-runtime.md)、[会话资源](../../workspace/sessions.md)
- [会话与 Alice 迁移](./v0.7.0-conversation-session.md)：2026-10-15，PR #122；[会话资源](../../workspace/sessions.md)、[Chat 视图](../../frontend/chat-view.md)
- [写入意图实时派发](./v0.7.0-intent-inflight-dispatch.md)：2026-10-13，PR #120；[读取视图与写入意图](../../workspace/read-view-and-write-intents.md)、[MTP 契约](../../contracts/mtp.md)
- [总线路由类型化与 mypy 门禁](./v0.7.0-typed-bus-routes.md)：2026-10-12，PR #117；[运行时机制](../../components/runtime-and-bus.md)、[路由与事件](../../contracts/routes-and-events.md)
- [执行线程层](./v0.7.0-execution-thread-layer.md)：2026-10-11，PR #116；[执行线程](../../workspace/execution-threads.md)
- [线程 context 取代执行凭据](./v0.7.0-thread-context.md)：2026-10-11，PR #115；[执行线程](../../workspace/execution-threads.md)
- [Agent 操作白名单](./v0.7.0-agent-operation-whitelist.md)：2026-10-11，PR #114；[能力层](../../workspace/capability.md)、[MTP 契约](../../contracts/mtp.md)
- [能力层分层：root API 与 actor API](./v0.7.0-capability-surface-split.md)：2026-10-10，PR #113；[能力层](../../workspace/capability.md)
- [Alice 的能力层调用迁移](./v0.7.0-alice-capability-migration.md)：2026-10-09，PR #111；[能力层](../../workspace/capability.md)、[Alice](../../alice/README.md)
- [写入意图登记迁入 workspace 与读取缓存失效](./v0.7.0-intent-registry-and-read-cache.md)：2026-10-07，PR #109；[读取视图与写入意图](../../workspace/read-view-and-write-intents.md)
- [身份与访问体系第二批](./v0.7.0-identity-access-batch-2.md)：2026-10-04，PR #107；[身份与授权](../../workspace/identity-and-authorization.md)
- [A1 访问边界返工](./v0.7.0-a1-access-boundary-rework.md)：2026-10-04，PR #106；[身份与授权](../../workspace/identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)、[认证与访问登记](../../workspace/authentication.md)
- 任务进程方向的五批：[落位与进程标识](./v0.7.0-task-process-table.md)、[prepare 拆分与 CPU 输入清单](./v0.7.0-task-process-prepare-split.md)、[结算阶段的中立输入](./v0.7.0-task-process-finalize-neutral-input.md)、[CPU 端口与测试 CPU](./v0.7.0-task-process-cpu-port.md)、[命令只解析不执行](./v0.7.0-task-process-command-parse-only.md)：2026-09-28 至 10-01，PR #105；[任务进程](../../workspace/task-process.md)、[Gateway 全局命令](../../gateway/commands.md)
- [A2-P 记忆内容版本与 Lifecycle](./v0.7.0-a2-pre-memory-version-and-lifecycle.md)、[A2-P 时间语义与可控时钟](./v0.7.0-time-semantics-and-controllable-clock.md)：2026-09-24，PR #102；[数据模型](../../architecture/data-model.md)、[Lifecycle](../../patchouli/lifecycle.md)
- [A1 Workspace 访问边界与授权](./v0.7.0-a1-workspace-access-boundary.md)：2026-09-19，PR #100；[身份与授权](../../workspace/identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)

## v0.6.2

- [v0.6.2 版本收尾审计](./v0.6.2-release-closeout-audit.md)：2026-09-14，PR #99；[PROJECT](../../PROJECT.md)
- [W1 Chat Attachments](./v0.6.2-w1-chat-attachments.md)：2026-09-14，PR #97；[Chat 附件链路](../../system/attachments.md)
- [Workspace Runtime 聚合与缓存所有权迁移](./v0.6.2-workspace-runtime-cache-migration.md)：2026-09-13，PR #98；[读取视图与写入意图](../../workspace/read-view-and-write-intents.md)、[System 组合根](../../system/composition.md)
- [用户身份上下文与裸 user_id 兼容投影收敛](./v0.6.2-identity-projection-cleanup.md)：2026-09-06，PR #96；[身份与授权](../../workspace/identity-and-authorization.md)
- [V1 Memory 与 Artifact Legacy 数据迁移及 legacy 分支删除](./v0.6.2-v1-memory-legacy-migration.md)：2026-09-06，PR #96；[数据模型](../../architecture/data-model.md)、[MemoryLibrary](../../patchouli/memory-library.md)
- [Perception Topic Buffer 边界重组](./perception-topic-buffer-boundary-refactor.md)：2026-09-05，PR #95；[感知与短期话题](../../patchouli/perception.md)
- [ShortTermMemoryStore 边界收敛](./short-term-memory-store-boundary-cleanup.md)：2026-09-02，PR #95；[MemoryLibrary](../../patchouli/memory-library.md)
- [W0 Workspace MVP](./v0.6.2-workspace-mvp.md)：2026-09-01，PR #94；[Workspace](../../workspace/README.md)

## v0.6.1 及更早

- [Local Work Queue Runtime](./v0.6.1-local-work-queue-runtime.md)：2026-08-16，PR #88；[运行时机制](../../components/runtime-and-bus.md)
- [RuntimeEvent 生产端发布抽象重构](./runtime-event-publishing-refactor.md)：2026-08-13，PR #86，部分落地；[RuntimeEvent 观测契约](../../contracts/routes-and-events.md#4-runtimeevent-观测契约)、[生产端迁移 Todo](../../governance/observability/logs-and-events.md)
- [Chat Run 取消重构最小闭环](./chat-run-cancellation-unified.md)：2026-08-05，PR #82；[任务进程](../../workspace/task-process.md)
- [Alice 父子 Agent 进程调度流程收口](./alice-parent-child-run-scheduler.md)、[Alice Agent Runtime 控制流重构](./alice-agent-runtime-control-flow-refactor.md)：2026-08-03 至 08-05，PR #81；[Alice](../../alice/README.md)、[Agent Runtime](../../alice/agent-runtime.md)
- [文档体系迁移清单](./documentation-migration-inventory.md)：2026-07-30，PR #78；[文档治理规范](../../DOCUMENTATION.md)
- 文档迁移逐篇审计：[清单第 4～6 节](./documentation-migration-audit-sections-4-6.md)、[第 7 节](./documentation-migration-audit-section-7.md)、[第 8 节](./documentation-migration-audit-section-8.md)、[第 9～10 节](./documentation-migration-audit-sections-9-10.md)、[`docs/mod`](./documentation-migration-audit-docs-mod.md)、[最终收口](./documentation-migration-finalization-audit.md)：2026-07-29，PR #77
- [历史实施计划](./implementation/README.md)：原 `docs/mod/` 迁入的已完成或被替代的实施稿

## 删除的计划

未完成实施即作废的计划不进入本目录，直接删除，历史由版本控制保留：

- v0.7.0 期间删除的 A2 未完成部分、A5、A6、WRX-0 清单与计划 A 协调入口与边界宪章：删除前的最后版本见 commit `dda9d9d`，处置记录见 commit `6add90f` 版本的《Workspace 网络与任务进程架构》Idea 第 5.5 节；
- 占位计划“话题折叠、Actor 上下文与原始证据统一改造”：2026-10-01 退回 Idea 后删除，删除前的最后版本见 commit `74b5056`，后台部分并入 [Page Folding Raw Evidence](../../ideas/PatchouliPageFoldingRawEvidenceDesign.md)，前台部分并入[长时间运行 Agent 的 Turn 内上下文折叠](../../ideas/long-running-agent-intra-turn-context-folding.md)。
