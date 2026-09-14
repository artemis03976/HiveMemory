---
title: Architecture Decision Records
status: current
owner: project
scope: architecture-decisions
last_reviewed: 2026-09-13
---

# Architecture Decision Records

本目录保存具有长期解释价值的架构决策。ADR 使用 `NNNN-kebab-case-title.md` 命名，并包含 Context、Decision、Consequences、Alternatives、Status 和相关文档。

当前决策：

- [ADR-0001：按语义选择可变性，跨边界使用只读投影](./0001-data-model-mutability-and-boundary-projection.md)
- [ADR-0002：全局唯一身份与按需并发保护](./0002-unique-identities-and-minimal-concurrency.md)
- [ADR-0003：Memory-as-a-Tool 与 MTP RUN 的边界语义](./0003-memory-as-a-tool-and-mtp-run-boundary.md)
- [ADR-0004：Workspace 派生缓存的分区与所有权聚合](./0004-workspace-derived-cache-partitioning.md)（已被 ADR-0005 替代）
- [ADR-0005：执行路径派生缓存的所有权与键控](./0005-execution-path-derived-caches.md)

后续从历史设计中提炼 ADR 时，优先评估以下主题：

- Gateway 独立为顶层子系统；
- System / Service / Runtime 分层；
- Agent Runtime 与 MTP Runtime 解耦；
- MemoryCompiler 作为统一记忆表达边界；
- 纯异步总线与全局维护调度；
