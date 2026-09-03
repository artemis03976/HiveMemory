---
title: Archived Plans
status: current
owner: project
scope: completed-or-superseded-plans
last_reviewed: 2026-09-02
---

# Archived Plans

本目录保存已经完成或被替代、且当前事实已经合并进规范文档的实施计划。每篇归档计划必须标明归档日期、实现版本或 PR，以及替代它的当前文档。

当前记录：

- [ShortTermMemoryStore 边界收敛](./short-term-memory-store-boundary-cleanup.md)：ShortTermMemoryStore 已收敛为 CRUD 与快照边界，WorkspaceTopicKey 已封装在短期 adapter；当前事实见 [Patchouli MemoryLibrary](../../patchouli/memory-library.md) 与 [Patchouli Perception](../../patchouli/perception.md)。
- [v0.6.2 W0 Workspace MVP](./v0.6.2-workspace-mvp.md)：P0–P6 实施、双 Workspace 隔离回归和 P7 文档收口已完成。当前 Workspace 事实见 [Workspace 架构](../../architecture/workspace.md)，System、Patchouli、Contracts、Alice、Gateway、Frontend 与治理文档承接各自边界；本计划仅保留实施历史、补充裁定、迁移边界和验收证据。
- [Workspace 文档收口审计](./workspace-documentation-closeout-audit.md)：记录 D0–D5 对 Workspace Plan、Idea、Roadmap、Architecture、System、Patchouli、Contracts、治理、Alice、Gateway 与 Frontend 的逐项承接和最终链接/状态验证；当前事实见 [Workspace 架构](../../architecture/workspace.md)。
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
