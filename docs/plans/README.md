---
title: Plans
status: current
owner: project
scope: implementation-plans
last_reviewed: 2026-08-05
---

# Plans

本目录存放已经形成明确目标、范围和验收条件，但尚未完全成为当前事实的功能、重构与迁移计划。

当前计划：

- [Chat Run 取消重构最小闭环](./chat-run-cancellation-unified.md)：当前实施依据；仅以 `Task.cancel()` 跑通 `/chat/stop` 对 Gateway 与 Alice agent run 的取消，移除阶段内部私有取消协议，prepare 延迟响应 stop，finalize 进入后不再接受 stop。
- [Chat Run 取消与生命周期后续设计](./chat-run-cancellation-future.md)：不阻塞最小闭环的独立候选集合，包括 SSE 所有权解耦、事件发布收敛、前端 `final_text` 契约等；`PreparedRunLease` 无限期延后，并保留删除该设计的可能。
- [数据模型可变性治理](./data-model-mutability-governance.md)：未排期的项目级模型角色、所有权与边界投影治理。
- [RuntimeEvent 生产端发布抽象重构](./runtime-event-publishing-refactor.md)：统一 Publisher、领域 emitter、payload 安全与生产端 best-effort 边界，当前未实现。
- [运行时状态持久化与故障恢复](./runtime-state-durability-and-recovery.md)：统一 Memory、Artifact、PendingAtom、Agent frame、工作项和恢复语义的耐久性分级；不替代 Local Work Queue 的机械设计。
- [跨子系统幂等性与重试语义](./cross-subsystem-idempotency-and-retry.md)：统一 interaction、generation、Artifact、MemoryLibrary、PendingAtom 与生命周期操作的稳定身份、重复结果和模糊失败边界。
- [身份隔离与执行安全](./identity-isolation-and-execution-safety.md)：收紧 Identity propagation、cache/frame 隔离、Profile fail-open 和 MTP RUN 的执行安全边界。
- [复合意图分解](./composite-intent-decomposition.md)：未排期；先建立样本门禁与 composite envelope，再讨论多分支执行。
- [v0.6.1 Local Work Queue Runtime](./v0.6.1-local-work-queue-runtime.md)：以多 lane 的本地工作队列收敛 interaction submission、memory generation 与未来 runtime job 的机械生命周期。

原 `docs/mod/` 混合目录已经完成逐篇审计与物理迁移。新计划只进入本目录；已完成或被替代的实施稿从 [Archived Plans](../archive/plans/README.md) 查阅。
