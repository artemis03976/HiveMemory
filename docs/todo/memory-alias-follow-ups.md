---
title: Memory alias 后续事项
status: todo
owner: patchouli
scope: aliasless-memory-addressing-and-alias-lookup-index
priority: unscheduled
code_paths:
  - src/hivememory/engines/generation/alias.py
  - src/hivememory/core/models/memory.py
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/infrastructure/storage/vector_store.py
related_docs:
  - docs/archive/todo/memory-alias-uniqueness.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/generation.md
last_reviewed: 2026-09-27
---

# Memory alias 后续事项

## 状态

**未排期。** 两项均来自 alias 唯一性修复（[已归档](../archive/todo/memory-alias-uniqueness.md)）实现时记录的后续事项，原定在已作废删除的 A2 计划中决定，现由本 Todo 承接。

## 问题与证据

1. **无 alias 记忆的 fallback 别名会作为 canonical alias 返回。** `AliasGenerator.build_candidate` 在 suffix 与 title 清洗后都为空（例如纯中文）时不生成 alias。此时 `MemoryAtom.get_alias()` 会按 memory type 与 title 临时构造 fallback 别名，该别名没有持久化，也不参与唯一性校验；但 `memory_generation.py` 在 settlement 中以 `atom.get_alias()` 填写 `canonical_alias`，调用方因此拿到一个无法可靠寻址的别名。
2. **alias 占用查询没有 payload index。** 一次完整写入会执行 2～3 次 alias 占用查询（生成侧、提交边界预检、`upsert` 兜底），`index.alias` 与 Workspace 字段在 Qdrant 中尚无 keyword payload index；数据量增长后查询成本随之上升。

## 完成条件

- [ ] 确定无 alias 记忆的寻址方式，settlement 不再返回未持久化的 fallback 别名；
- [ ] 根据数据量决定是否为 `index.alias` 与 Workspace 字段建立 payload index（涉及集合结构，需单独安排迁移）。
