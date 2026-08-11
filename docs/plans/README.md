---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-08-10
---

# Plans

本目录存放已经形成明确目标、范围和验收条件，但尚未完全成为当前事实的功能、重构与迁移计划。

当前计划：

- [Chat Run 取消与生命周期后续设计](./chat-run-cancellation-future.md)：不阻塞最小闭环的独立候选集合，包括 SSE 所有权解耦、事件发布收敛、前端 `final_text` 契约等；`PreparedRunLease` 无限期延后，并保留删除该设计的可能。
- [数据模型可变性治理](./data-model-mutability-governance.md)：未排期的项目级模型角色、所有权与边界投影治理。
- [RuntimeEvent 生产端发布抽象重构](./runtime-event-publishing-refactor.md)：统一 Publisher、领域 emitter、payload 安全与生产端 best-effort 边界，当前未实现。
- [运行时状态持久化与故障恢复](./runtime-state-durability-and-recovery.md)：统一 Memory、Artifact、PendingAtom、Agent frame、工作项和恢复语义的耐久性分级；不替代 Local Work Queue 的机械设计。
- [跨子系统幂等性与重试语义](./cross-subsystem-idempotency-and-retry.md)：统一 interaction、generation、Artifact、MemoryLibrary、PendingAtom 与生命周期操作的稳定身份、重复结果和模糊失败边界。
- [身份隔离与执行安全](./identity-isolation-and-execution-safety.md)：收紧 Identity propagation、cache/frame 隔离、Profile fail-open 和 MTP RUN 的执行安全边界。
- [复合意图分解](./composite-intent-decomposition.md)：未排期；先建立样本门禁与 composite envelope，再讨论多分支执行。
- [v0.6.1 Local Work Queue Runtime](./v0.6.1-local-work-queue-runtime.md)：以多 lane 的本地工作队列收敛 interaction submission 与 memory generation 的机械生命周期，并完成 Active/Passive 共用 submission lane 的迁移。

已完成的 v0.6.1 前置清单：

- [Phase D0 持久化状态清单](./durability-d0-state-inventory.md)：冻结状态所有者、真相源、耐久性等级、恢复和保留语义现状。
- [Phase I0 业务操作清单](./idempotency-i0-operations-inventory.md)：冻结可重试入口、operation identity、重复结果、并发冲突和模糊失败现状。
- [Phase S0 身份与威胁模型清单](./identity-s0-threat-model-inventory.md)：冻结身份输入、授权所有者、继承关系和最小威胁复现样本。
- [数据模型 Phase I 清单](./data-model-phase-i-inventory.md)：冻结模型角色、冻结深度、聚合所有权、传播边界和复制性能基线。

原 `docs/mod/` 混合目录已经完成逐篇审计与物理迁移。新计划只进入本目录；已完成或被替代的实施稿从 [Archived Plans](../archive/plans/README.md) 查阅。
