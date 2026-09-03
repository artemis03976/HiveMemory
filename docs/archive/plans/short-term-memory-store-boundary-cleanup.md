---
title: ShortTermMemoryStore Boundary Cleanup Implementation Plan
status: archived
owner: patchouli
scope: short-term-memory-store-crud-boundary-and-storage-key-encapsulation
related_docs:
  - docs/todo/short-term-memory-store-boundary-cleanup.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/perception.md
last_reviewed: 2026-09-02
archived_at: 2026-09-02
superseded_by:
  - docs/patchouli/memory-library.md
  - docs/patchouli/perception.md
---

# ShortTermMemoryStore 边界收敛实施计划

本计划承接 [ShortTermMemoryStore Boundary Cleanup TODO](../../todo/short-term-memory-store-boundary-cleanup.md)，只记录本次分支的实施顺序、验收出口和 code review 记录，不复制 TODO 中的问题分析与长期设计理由。实现已完成；当前事实以 [Patchouli MemoryLibrary](../../patchouli/memory-library.md) 和 [Patchouli Perception](../../patchouli/perception.md) 为准。

## 实施阶段

1. **边界冻结**：完成调用清单和职责分类，确认迁移跨越 MemoryLibrary、Perception、事件模型与 adapter。
2. **存储键封装**：将 `WorkspaceTopicKey` 限制在短期 adapter/内部索引，领域模型、Perception 接口和事件使用 `IdentityScope + topic_id`。
3. **CRUD Store**：将 ShortTermMemoryStore 收敛为创建、读取、写回、删除、列表、计数和健康检查；状态机与生命周期编排迁回 Perception。
4. **并发归属**：Store/adapter 只保护存储容器和快照复制；Perception 负责状态转换的顺序与不变量。
5. **测试与文档收口**：补齐 CRUD、Workspace 隔离、快照隔离和无键泄漏契约测试，并同步当前事实文档。

## 验收出口

- Store 公共 API 不接收或返回 `WorkspaceTopicKey`，也不提供 `by_key` 入口。
- `SemanticBuffer` 与 `TopicData` 不暴露 `topic_key` 属性；复合键只由 adapter 构造和消费。
- Perception 的 Interaction、Compact、settle、evict、LRU 和状态预约行为保持可验证，且不依赖复合键事件参数。
- adapter 不向 Port/Store 泄漏可变内部 buffer；公开读取返回不可变快照或副本。
- 受影响测试按 testing governance 分类运行，静态契约检查确认键未泄漏到上层稳定接口。

## Code review 记录

- 阶段 1：调用边界盘点完成，确认需要跨子系统迁移；实现前建立本计划。
- 阶段 2：复合键收回 adapter，Perception/领域模型改用 `IdentityScope + topic_id`；静态搜索与 CRUD smoke review 通过。
- 阶段 3：ShortTermMemoryStore 收敛为 CRUD，状态机/生命周期迁回 TriggerManager 与 Perception；reservation/settlement smoke review 通过。
- 阶段 4：锁归属重画为 adapter map/index、Store snapshot write-back、Perception state transition 三层；快照隔离与状态转换契约测试通过。
- 阶段 5：完成 CRUD、Workspace 隔离、快照隔离和无键泄漏契约测试；复核文档事实、公开 API、锁归属与失败恢复路径，新增契约测试通过，计划与 TODO 收口。
