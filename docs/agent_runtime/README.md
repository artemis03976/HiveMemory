---
title: Legacy Agent Runtime Index
status: superseded
owner: alice
scope: retired-parallel-agent-runtime-index
archived_at: 2026-07-28
superseded_by:
  - docs/alice/agent-runtime.md
  - docs/alice/pending-atom.md
  - docs/alice/mtp-runtime.md
---

> 本目录仍保留执行层历史设计稿，但不再作为平级子系统文档入口。Agent Runtime 是 Alice 消费的共享执行层；当前设计以 [Agent Runtime](../alice/agent-runtime.md)、[PendingAtom](../alice/pending-atom.md)和 [MTP Runtime](../alice/mtp-runtime.md)为准。

# Agent Runtime

本目录用于存放单 Agent 执行运行时相关文档，包括 Agent loop、Koakuma ISA、PendingAtom 写缓冲与运行时生命周期设计。

## 当前目录

- [pending_atom](./pending_atom/README.md): PendingAtom 运行时、状态、缓存与 materialize task 设计。
