---
title: Architecture Decision Records
status: current
owner: project
scope: architecture-decisions
last_reviewed: 2026-07-27
---

# Architecture Decision Records

本目录保存具有长期解释价值的架构决策。ADR 使用 `NNNN-kebab-case-title.md` 命名，并包含 Context、Decision、Consequences、Alternatives、Status 和相关文档。

当前尚未从历史设计中提炼正式 ADR。迁移时优先评估以下主题：

- Gateway 独立为顶层子系统；
- System / Service / Runtime 分层；
- Agent Runtime 与 MTP Runtime 解耦；
- MemoryCompiler 作为统一记忆表达边界；
- 纯异步总线与全局维护调度；
- 不可变公共模型与可变内部实体的边界。
