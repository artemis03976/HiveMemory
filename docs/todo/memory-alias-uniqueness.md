---
title: Memory Alias 重名缺陷
status: todo
owner: patchouli
scope: memory-alias-uniqueness-and-cache-key-precondition
related_docs:
  - docs/plans/v0.7.0-plan-a-boundary-charter.md
  - docs/plans/v0.7.0-a2-workspace-resource-reads-and-caches.md
  - docs/patchouli/generation.md
  - docs/todo/agent-profile-model-evolution.md
last_reviewed: 2026-09-25
---

# Memory Alias 重名缺陷

## 问题与证据

alias 由 `GenerationEngine._build_alias`（`src/hivememory/engines/generation/engine.py`）按 memory type 前缀 + extractor 给出的 suffix 构造（`code_*`/`fact_*` 等），全链路无唯一性校验：MidTerm store、`MemoryLibrary` 与生成/管理写入路径都不拒绝同一 Workspace 内的重复 alias。

alias 精确查询走 `QdrantMemoryStore.get_memory_by_alias`（scroll + `index.alias` MatchValue 精确匹配），重名时返回 scroll 顺序的第一个结果——解析结果多义，且不保证跨调用稳定。

影响：

1. 存储层 alias → 记忆的解析已是多义，精确读取结果不确定；
2. [计划 A 边界宪章](../plans/v0.7.0-plan-a-boundary-charter.md) §5 与 [A2](../plans/v0.7.0-a2-workspace-resource-reads-and-caches.md) 的 Atom cache 与 Profile cache 均以 `(WorkspaceIdentity, alias)` 系资源 key 寻址（Profile 的 agent_id 即 alias），重名使 key 无法定义唯一条目，缓存与存储可能各自解析到不同原子；
3. A2 的 alias 替换/删除簿记（"不留可返回的旧值"）以 alias 归属可追踪为前提。

## 已裁定的分层修复设计（2026-09-25）

唯一性由两层共同保证，缺一不可：

**第一层（生成侧体验）：engine 层 `AliasGenerator` 独立组件**

- 替换 `GenerationEngine._build_alias`：按 memory type 前缀 + suffix 产出候选，查中期库验证唯一，冲突时消歧后缀重试；
- 注入式组件（依赖 MidTermMemoryStore 的查询口），保证抽取草稿的正常路径零摩擦、不触碰冲突错误；
- 注入式使其可被 Profile 管理等路径复用。

**第二层（不变量兜底）：`MidTermMemoryStore.upsert` 写前校验**

- 写入前检查同 Workspace 内 alias 是否被他者占用（排除自身 memory_id），冲突抛结构化错误；
- 这是**唯一能覆盖全部路径的检查点**——已核实的绕过路径（仅靠生成侧时全部失守）：
  1. Familiar 手工/外部 create+update：caller 直接传 `alias` 参数（`memory_generation.py:176→248` 写入 `index.alias`），不经生成器；
  2. Profile 管理创建：caller 传 `agent_alias`（`agent_profile_management_service.py`）；
  3. revive：归档期间 alias 已被新原子占用，revive 重新 upsert 时撞名；
  4. 未来任何新增 mutation。
- 触发时机 = 主后端写入前；校验失败不产生任何写入。

**语义裁定**：唯一性不变量 = "同一 Workspace 的中期库内唯一"。归档即释放 alias；revive 撞名显式失败（revive 本就是显式操作，报错可解释），不做归档期 alias 保留。`patch_payload` 白名单不含 `index.alias`（已核实），该路径无绕行；UPDATE 保留现有 alias、无 alias 原子经 update focus 获得 alias 的边角由第二层覆盖。

**与 AgentProfile 的关系**：C2 裁定 agent_id 即 alias、profile cache 按 `(Workspace, agent_alias)` 作 key（见 [AgentProfile 模型演进](./agent-profile-model-evolution.md)）——本 todo 的唯一性是 agent_id 唯一性的直接前提。

**衔接时机**：第一、二层均先于或伴随 [A2](../plans/v0.7.0-a2-workspace-resource-reads-and-caches.md) A2-1 的 alias 索引交付；归属计划由 A2-0 裁定。

## 完成条件

- [ ] `AliasGenerator` 组件落地并替换 `_build_alias`，生成路径冲突时消歧重试有测试；
- [ ] `MidTermMemoryStore.upsert` 写前唯一性校验落地，冲突抛结构化错误，覆盖 revive/手工/Profile 管理路径有测试；
- [ ] 存量重名数据有盘点与消解记录；
- [ ] alias 精确查询在任意存储顺序下解析结果唯一且稳定。
