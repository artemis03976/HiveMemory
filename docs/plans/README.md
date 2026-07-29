---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-07-29
---

# Plans

本目录存放已经形成明确目标、范围和验收条件，但尚未完全成为当前事实的功能、重构与迁移计划。

当前计划：

- [文档体系迁移清单](./documentation-migration-inventory.md)
- [数据模型可变性治理](./data-model-mutability-governance.md)：未排期的项目级模型角色、所有权与边界投影治理。
- [RuntimeEvent 生产端发布抽象重构](./runtime-event-publishing-refactor.md)：统一 Publisher、领域 emitter、payload 安全与生产端 best-effort 边界，当前未实现。
- [v0.6.0 复合意图分解](./v0.6.0-composite-intent-decomposition.md)：先建立样本门禁与 composite envelope，再讨论多分支执行。
- [v0.6.1 Local Work Queue Runtime](./v0.6.1-local-work-queue-runtime.md)：以多 lane 的本地工作队列收敛 interaction submission、memory generation 与未来 runtime job 的机械生命周期。

原 `docs/mod/` 混合目录已经完成逐篇审计与物理迁移。新计划只进入本目录；已完成或被替代的实施稿从 [Archived Plans](../archive/plans/README.md) 查阅。
