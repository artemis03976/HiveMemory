---
title: Memory alias 后续事项
status: todo
owner: patchouli
scope: alias-lookup-payload-index
priority: unscheduled
code_paths:
  - src/hivememory/engines/generation/alias.py
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/infrastructure/storage/vector_store.py
related_docs:
  - docs/archive/todo/memory-alias-uniqueness.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/generation.md
last_reviewed: 2026-11-07
---

# Memory alias 后续事项

## 状态

**未排期。** 两项均来自 alias 唯一性修复（[已归档](../archive/todo/memory-alias-uniqueness.md)）实现时记录的后续事项，原定在已作废删除的 A2 计划中决定，现由本 Todo 承接。第一项（无 alias 记忆的 fallback 别名作为 canonical alias 返回）已于 2026-11-07 按 owner 决定处理：生成时后缀与标题都无法构造候选、或手动创建没有给出 alias 时，以 `mem_<记忆 ID>` 命名并落库（Agent Profile 仍须显式给出 alias），结算不再返回未持久化的 fallback 别名（[Patchouli 生成](../patchouli/generation.md)第 4 节）。剩下第二项。

## 问题与证据

**alias 占用查询没有 payload index。** 一次完整写入会执行 2～3 次 alias 占用查询（生成侧、提交边界预检、`upsert` 兜底），`index.alias` 与 Workspace 字段在 Qdrant 中尚无 keyword payload index；数据量增长后查询成本随之上升。

## 完成条件

- [ ] 根据数据量决定是否为 `index.alias` 与 Workspace 字段建立 payload index（涉及集合结构，需单独安排迁移）。
