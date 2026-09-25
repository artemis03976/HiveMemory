---
title: Memory Alias 重名缺陷
status: todo
owner: patchouli
scope: memory-alias-uniqueness-and-cache-key-precondition
related_docs:
  - docs/plans/v0.7.0-plan-a-boundary-charter.md
  - docs/plans/v0.7.0-a2-workspace-resource-reads-and-caches.md
  - docs/patchouli/generation.md
last_reviewed: 2026-09-23
---

# Memory Alias 重名缺陷

## 问题与证据

alias 由 `GenerationEngine._build_alias`（`src/hivememory/engines/generation/engine.py`）按 memory type 前缀 + extractor 给出的 suffix 构造（`code_*`/`fact_*` 等），全链路无唯一性校验：MidTerm store、`MemoryLibrary` 与生成/管理写入路径都不拒绝同一 Workspace 内的重复 alias。

alias 精确查询走 `QdrantMemoryStore.get_memory_by_alias`（scroll + `index.alias` MatchValue 精确匹配），重名时返回 scroll 顺序的第一个结果——解析结果多义，且不保证跨调用稳定。

影响：

1. 存储层 alias → 记忆的解析已是多义，精确读取结果不确定；
2. [计划 A 边界宪章](../plans/v0.7.0-plan-a-boundary-charter.md) §5 与 [A2](../plans/v0.7.0-a2-workspace-resource-reads-and-caches.md) §2 的 Atom cache 以 `(WorkspaceIdentity, normalized_alias)` 为资源 key，重名使该 key 无法定义唯一条目，缓存与存储可能各自解析到不同原子；
3. A2 §2.2 的 alias 替换/删除簿记（"同一资源多个 alias 的替换/删除不留下可返回的旧值"）以 alias 归属可追踪为前提，重名下不成立。

## 修复方向（待归属计划细化）

- **域规则落点**：同一 Workspace 内 canonical alias 唯一。候选做法二选一，须在修复计划中裁定：受控提交路径上做存在性检查、冲突显式拒绝（返回结构化错误）；或生成侧保证唯一（冲突时附加消歧后缀）+ 存储侧校验兜底。
- **存量数据**：已存在的重名 alias 需要一次盘点与消解（复用 A2-P 的受控 mutation 语义），不能只对新写入生效。
- **衔接时机**：修复应先于或伴随 A2-1 的 alias 索引交付，否则缓存键前提不成立；归属 A2-0 还是独立修复计划由 A2-0 裁定。

## 完成条件

- 同一 Workspace 内重复 alias 被结构化拒绝或生成侧不可产生，行为有测试覆盖；
- 存量重名数据有盘点与消解记录；
- alias 精确查询在任意存储顺序下解析结果唯一且稳定。
