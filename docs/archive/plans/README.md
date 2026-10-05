---
title: Archived Plans
status: current
owner: project
scope: completed-or-superseded-plans
last_reviewed: 2026-10-04
---

# Archived Plans

> 2026-09-27 链接维护：本文引用的 v0.7.0 A6 计划已作废删除，相关链接改为纯文本；删除前最后版本见 commit `dda9d9d`。

本目录保存已经完成或被替代、且当前事实已经合并进规范文档的实施计划。每篇归档计划必须标明归档日期、实现版本或 PR，以及替代它的当前文档。

当前记录：

- [v0.7.0 身份与访问体系第二批](./v0.7.0-identity-access-batch-2.md)：Patchouli 公开边界拆出归属与发起者，内部和后台记录不保存操作 scope；SETTLE 四种触发使用 system，查重只读取 PUBLIC；finalize/cleanup 使用阶段授权结果，actor 的 session_id 已移除。2026-10-04 经全量门槛、确定性 HTTP chat E2E 与审查后提交（commit `b2c7aee`，审查后调整见计划第 11.4 节），尚未合并，不表示版本已发布；[WorkspaceAsset 内部拆分](../../todo/workspace-asset-ownership-identity-split.md)按用户要求暂缓。当前事实见 [Workspace 架构](../../architecture/workspace.md)、[子系统公共契约](../../contracts/subsystem-contracts.md)与 [Patchouli 生成](../../patchouli/generation.md)。
- [v0.7.0 A1 访问边界返工](./v0.7.0-a1-access-boundary-rework.md)：A1 遗留的生产入口接线，按身份与访问体系 Idea 的第一批完成——与 workspace 相关的 HTTP 请求都经统一认证网关；四种身份数据形态与两阶段认证加两阶段授权落地；访问 context 改为密封的运行时凭据，guard 拆为 `WorkspaceAuthenticator` 与 `WorkspaceOperationAuthorizer` 且互不依赖；注册入口先注册后运行，进程表登记任务进程，句柄按对象身份判定有效，取消统一为一个方法；Patchouli 与 Gateway 只接收 `IdentityScope`；两个登记文件、用户级记录与默认登记；附件上传 scope 缺陷结构性修复；当前事实见 [Workspace 架构](../../architecture/workspace.md)第 4 节、[错误模型](../../contracts/error-model.md)第 4.4 节、[System 应用服务](../../system/application-services.md)第 4 节与 [System 配置](../../system/configuration.md)第 1.1 节。
- [v0.7.0 任务进程：命令只解析不执行](./v0.7.0-task-process-command-parse-only.md)：任务进程方向第五批实施已完成——Gateway 的命令结果只携带解析结果（解析模型移入 `core.protocol.gateway`），删除命令分发与执行，任务进程按解析状态产生命令终态，内置命令暂时不可用；被动请求按 Q-6 这一阶段不考虑，版本目标第 4 条在这一阶段收口；当前事实见 [Gateway 全局命令](../../gateway/commands.md)与[子系统公共契约](../../contracts/subsystem-contracts.md)。
- [v0.7.0 任务进程：CPU 端口与测试 CPU](./v0.7.0-task-process-cpu-port.md)：任务进程方向第四批实施已完成——`workspace.contracts` 定义对象端口 `CPUPort` 与 CPU 中立的 `CPUExecutionResult`（取代 `AgentRunResult`），任务进程只经组合根注入的端口调用 CPU；Alice 合并流式与非流式为以 `stream` 参数控制的统一入口与单一路由，并以 `AliceCPU` 实现端口；测试 CPU 在没有 Alice 路由的情况下跑通任务进程，收口 v0.7.0 版本目标第 2 条；当前事实见[子系统公共契约](../../contracts/subsystem-contracts.md)与 [System 应用服务](../../system/application-services.md)。
- [v0.7.0 任务进程：结算阶段的中立输入](./v0.7.0-task-process-finalize-neutral-input.md)：任务进程方向第三批实施已完成——任务进程在进入 finalize 前封口交互记录 `InteractionPayload`（`workspace/process/sealing.py`，MTP 轨迹经 core 归约器得到），Patchouli finalize 改为接收 `PreparedAgentRun` 与 `InteractionPayload` 并原样提交，不再接收 `AgentRunResult`，收口 v0.7.0 版本目标第 1 条；当前事实见[子系统公共契约](../../contracts/subsystem-contracts.md)与 [System 应用服务](../../system/application-services.md)。
- [v0.7.0 任务进程：prepare 拆分与 CPU 输入清单](./v0.7.0-task-process-prepare-split.md)：任务进程方向第二批实施已完成——Patchouli prepare 只做话题与检索，`PreparedAgentRun` 移入 `patchouli.contracts`；任务进程在进入 Alice 前完成 CPU 分配（Profile 解析、附件租借与编译、记忆编译），输入清单 `CPUInputManifest` 位于新建的 `workspace.contracts`，附件租借由进程工作集持有并在进程结束时释放；当前事实见 [System 应用服务](../../system/application-services.md)、[子系统公共契约](../../contracts/subsystem-contracts.md)与 [Chat 附件链路](../../system/attachments.md)。
- [v0.7.0 任务进程表：落位与进程标识](./v0.7.0-task-process-table.md)：任务进程方向第一批实施已完成——chat run 注册表与 chat 编排从 `alice.application` 迁入 `workspace.process` 并改用进程词汇（阶段 A），`process_id` 取代 `interaction_id`/`generation_id` 成为进程唯一标识、取消只在 Gateway 与 Actor 执行两个阶段响应（阶段 B）；当前事实见 [System 应用服务](../../system/application-services.md)、[系统边界与所有权](../../architecture/boundaries.md)与[公开路由与事件](../../contracts/routes-and-events.md)。
- [v0.7.0 A1 Workspace 访问边界与授权](./v0.7.0-a1-workspace-access-boundary.md)：统一 Actor 认证网关、System 接入登记与 Workspace Actor 访问注册表、guard 持有的最小准入 context 与签发生命周期、逐次行为授权及迁移兼容清单已完成并通过验收；当前事实见 [Workspace 架构](../../architecture/workspace.md)第 4 节、[错误模型](../../contracts/error-model.md)第 4.4 节、[子系统公共契约](../../contracts/subsystem-contracts.md)第 3.5 节、[System 应用服务](../../system/application-services.md)与 [ADR-0005](../../architecture/decisions/0005-unified-actor-authentication-and-workspace-authorization.md)。生产消费者切换与兼容退出原由 A6 计划收口；A6 作废后由 [A1 访问边界返工](./v0.7.0-a1-access-boundary-rework.md)完成。
- [v0.6.2 版本收尾审计](./v0.6.2-release-closeout-audit.md)：记录已合并范围、归档承接、版本声明、测试与构建结果；代码版本与本次发布标签为 0.6.2（标签在合并后创建）。
- [v0.6.2 W1 Chat Attachments](./v0.6.2-w1-chat-attachments.md)：W1-A 至 W1-F 已随 PR #97 合并并归档；当前事实见 [Chat 附件链路](../../system/attachments.md)、[Workspace 架构](../../architecture/workspace.md) 与 [Artifacts](../../patchouli/artifacts.md)。

- [v0.6.2 Workspace Runtime 聚合与缓存所有权迁移](./v0.6.2-workspace-runtime-cache-migration.md)：WRT-0～WRT-5 已完成——进程级唯一 `WorkspaceRuntime` 聚合（AssetStore + 两个派生 cache + 窄化端口）落地，KoakumaAtomCache 按 `(WorkspaceIdentity, alias)`、AgentProfileCache 按 `(WorkspaceIdentity, Actor 投影, alias)` 分区，PendingAtomRuntime 保持 Alice 所有；后续经 [ADR-0004](../../architecture/decisions/0004-execution-path-derived-caches.md) 将派生缓存所有权归还 AliceRuntime、聚合解体（分区键控保留）；当前事实见 [Workspace 架构](../../architecture/workspace.md)、[System 组合根](../../system/composition.md)、[Alice](../../alice/README.md)、[MTP 契约](../../contracts/mtp.md) 与两份治理文档。
- [Perception Topic Buffer 边界重组](./perception-topic-buffer-boundary-refactor.md)：TopicWorkingSet（驻留 + lease）、纯 CRUD 短期 Store、纯算法 MemoryPerceptionEngine 与 lease 编排的 PerceptionFamiliar 已落地；`TopicBufferService`/`SemanticBuffer`/`BufferState`/决策矩阵与感知层已删除；当前事实见 [Patchouli Perception](../../patchouli/perception.md) 与 [Patchouli MemoryLibrary](../../patchouli/memory-library.md)。
- [ShortTermMemoryStore 边界收敛](./short-term-memory-store-boundary-cleanup.md)：ShortTermMemoryStore 已收敛为 CRUD 与快照边界，WorkspaceTopicKey 已封装在短期 adapter；当前事实见 [Patchouli MemoryLibrary](../../patchouli/memory-library.md) 与 [Patchouli Perception](../../patchouli/perception.md)。
- [v0.6.2 用户身份上下文与裸 user_id 兼容投影收敛](./v0.6.2-identity-projection-cleanup.md)：B1–B4 已实施并通过守卫测试——应用服务入口统一 `IdentityScope`、server 唯一身份解析、`InteractionTurnSnapshot` actor 值对象化与旧 JSON 读升级、读侧兼容属性收口、`MemoryAccessPolicy` 禁用 `system` target、管理/检索可见性路径分离；当前事实见 [Workspace 架构](../../architecture/workspace.md)、[System 应用服务](../../system/application-services.md)、[公开路由与事件](../../contracts/routes-and-events.md)与[数据模型](../../architecture/data-model.md)；V1 存储数据迁移后续事项见 [v0.6.2 迁移 Plan](./v0.6.2-v1-memory-legacy-migration.md)（已完成并归档）。
- [v0.6.2 V1 Memory 与 Artifact Legacy 数据迁移及 legacy 分支删除](./v0.6.2-v1-memory-legacy-migration.md)：已实施并通过验收——存量 32 条 V1 Memory 与 6 个 legacy Artifact 迁移为 canonical v2 / canonical replacement（确定性迁移 ID、checkpoint 续跑、fail-closed 与修复清单见 `data/migration/`），codec/filter/快照读升级与 Artifact owner 解释路径删除，§7 当前文档已更新；当前事实见 [Workspace 架构](../../architecture/workspace.md)、[数据模型](../../architecture/data-model.md)、[Retrieval](../../patchouli/retrieval.md) 与 [MemoryLibrary](../../patchouli/memory-library.md)。
- [v0.6.2 W0 Workspace MVP](./v0.6.2-workspace-mvp.md)：P0–P6 实施、双 Workspace 隔离回归和 P7 文档收口已完成。当前 Workspace 事实见 [Workspace 架构](../../architecture/workspace.md)，System、Patchouli、Contracts、Alice、Gateway、Frontend 与治理文档承接各自边界；本计划仅保留实施历史、补充裁定、迁移边界和验收证据。
- [v0.6.1 Local Work Queue Runtime](./v0.6.1-local-work-queue-runtime.md)：Q0–Q4 已完成，Active/Passive Interaction Submission 与 Memory Generation 已接入进程内通用运行时；当前事实见 System Runtime、Passive Ingress 与 Patchouli Generation，SQLite 后续由持久化治理承接。
- [Chat Run 取消重构最小闭环](./chat-run-cancellation-unified.md)：已完成的 phase task 控制、Gateway/Alice 原生 task cancellation、prepare 延迟响应、finalize 门禁，以及 SSE/Worker/unwind 清理加固；当前事实见 `docs/system/application-services.md`、`docs/system/runtime-and-bus.md`、`docs/gateway/`、`docs/alice/` 与 `docs/contracts/`。
- [Alice 父子 Agent 进程调度流程收口](./alice-parent-child-run-scheduler.md)：已完成的 run-local RunScheduler、统一 root/callee 活动 frame 循环、CALL begin/complete、取消/异常收口与编排兼容层删除；当前事实见 `docs/alice/` 与 `docs/contracts/mtp.md`。
- [Alice Agent Runtime 控制流重构](./alice-agent-runtime-control-flow-refactor.md)：已完成的单 frame runtime、run-local 编排、CALL transaction 与 PendingAtom 生命周期收口；当前事实见 `docs/alice/`。
- [RuntimeEvent 生产端发布抽象重构](./runtime-event-publishing-refactor.md)：部分基础设施与领域 emitter 已落地，当前规范由 System/Contracts 承接，剩余生产端接驳已缩减为 Todo；原大重构稿不再作为当前 Plan。

- [文档体系迁移清单](./documentation-migration-inventory.md)：文档重构各批次的原始范围、分类、迁移动作与最终完成记录。
- [文档迁移逐篇审计：清单第 4～6 节](./documentation-migration-audit-sections-4-6.md)：顶层治理、Architecture、System/Contracts/i18n 的承接与物理迁移记录。
- [文档迁移逐篇审计：清单第 7 节](./documentation-migration-audit-section-7.md)：Patchouli 与 Engines 的逐篇承接、设计理念复核、拒绝继承项和物理迁移记录。
- [文档迁移逐篇审计：清单第 8 节](./documentation-migration-audit-section-8.md)：Alice 与 Agent Runtime 的逐篇承接、设计理念复核、拒绝继承项和物理迁移记录。
- [文档迁移逐篇审计：清单第 9～10 节](./documentation-migration-audit-sections-9-10.md)：Gateway、Applications 与 Frontend 的逐篇承接、产品边界、拒绝继承项和物理迁移记录。
- [`docs/mod` 逐篇迁移审计](./documentation-migration-audit-docs-mod.md)：18 篇混合设计/计划的当前承接、计划保留、拒绝项与最终物理路径。
- [文档迁移最终收口审计](./documentation-migration-finalization-audit.md)：Ideas、残余 README、旧 `archive/mod/` 分类、索引与全库门禁的最终结论。
- [历史实施计划索引](./implementation/README.md)：从原 `docs/mod/` 迁入的已完成或被替代实施稿。

原 `docs/mod/` 已完成迁移：Local Work Queue 在 v0.6.1 完成后已进入本目录；复合意图分解已降级为 Idea；RuntimeEvent 生产端大重构稿在部分落地并拆分当前规范/Todo 后进入本目录；其余十五篇进入 `implementation/`。归档稿只保留演化证据，当前事实仍从项目与子系统索引进入。
