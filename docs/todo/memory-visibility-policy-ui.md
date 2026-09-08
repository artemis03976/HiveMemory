---
title: Memory Visibility Policy UI
status: todo
owner: frontend
scope: expose-memory-access-policy-in-create-and-management-views
code_paths:
  - frontend/src/types/memory.ts
  - frontend/src/components/memory/CreateMemoryModal.tsx
  - frontend/src/components/memory/MemoryDetailModal.tsx
  - frontend/src/services/memoryApi.ts
  - src/hivememory/server/models/memory.py
  - src/hivememory/server/routers/memories.py
related_docs:
  - ../frontend/management-views.md
  - ../architecture/workspace.md
  - ../archive/todo/memory-provenance-vs-authorship.md
last_reviewed: 2026-09-06
---

# Memory 可见性策略前端组件

## 现状

Memory Library 当前没有编辑 `MemoryAccessPolicy` 的前端组件。创建弹层 [CreateMemoryModal](../../frontend/src/components/memory/CreateMemoryModal.tsx) 只提交标题、摘要、正文、类型、标签和别名；详情编辑弹层 [MemoryDetailModal](../../frontend/src/components/memory/MemoryDetailModal.tsx) 也只允许修改内容字段。

前端的 [MemoryAtom 类型](../../frontend/src/types/memory.ts) 不包含 `access_policy`，[`memoryApi`](../../frontend/src/services/memoryApi.ts) 的创建和更新 payload 也没有策略字段。后端的 [MemoryCreateRequest](../../src/hivememory/server/models/memory.py) 与 [MemoryUpdateRequest](../../src/hivememory/server/models/memory.py) 同样未接受策略，MemoryResponse 未向前端投影策略；[memory service](../../src/hivememory/system/application/memory_service.py) 创建时固定使用 `MemoryAccessPolicy.public()`，更新路径不修改策略。

因此当前管理页面创建的 Memory 都是 PUBLIC，页面无法查看或修改已有记忆的 PRIVATE/TEAM 目标。该事实与管理读取“可以观测 Workspace 全部记忆”并不冲突：管理读取绕过 Agent 可见性过滤，但策略仍决定 Agent retrieval 的可见范围。

## 待办范围

增加一个可复用的可见性策略编辑组件，并接入创建弹层和详情编辑弹层：

- 提供 `PUBLIC`、`PRIVATE`、`TEAM` 选择；
- 选择 `PRIVATE` 时必须填写具体 `target_agent_id`；
- 选择 `TEAM` 时必须填写具体 `target_team_id`；
- `PUBLIC` 不显示或不提交任何 target；
- 前端拒绝空 target 和保留值 `system`，并在提交前显示校验错误；
- 详情页展示当前策略，编辑保存时只提交发生变化的策略字段；
- 创建成功和更新成功后的响应应回填策略，避免页面状态丢失；
- Agent/Team 选项应来自当前 Workspace 的可用资产，不能从 `source_agent_id` 或响应中的 owner 字段反推。

## 后端契约依赖

该 UI 需要先扩展 Memory CRUD 契约：

- `MemoryResponse` 返回 `access_policy`；
- `MemoryCreateRequest` 接受可选或显式的 `access_policy`，管理创建默认值是否继续为 PUBLIC 需要明确；
- `MemoryUpdateRequest` 接受策略更新，并由服务层校验 Workspace ownership 与策略 target；
- API 错误应把 PRIVATE/TEAM target 缺失、`system` target 和无效 Agent/Team 标识明确返回给前端。

后端仍以 `MemoryAccessPolicy` 为唯一授权模型。前端组件只能编辑和校验该契约，不得根据来源字段自行推导可见性，也不能把 `system` 当作可见性对象。

## 完成条件

- [ ] 前端 `MemoryAtom`、创建/更新 payload 和 API response 包含 `access_policy`；
- [ ] 创建和详情编辑页面共用策略组件，三种 visibility 的字段显隐和必填校验正确；
- [ ] `PRIVATE`/`TEAM` 不接受 `system` 或空 target，`PUBLIC` 不携带 target；
- [ ] Agent/Team 选择范围绑定当前 Workspace，切换 Workspace 后不会复用旧选项；
- [ ] 后端 Memory CRUD 契约支持策略读写并保持 `MemoryAccessPolicy` 校验；
- [ ] 覆盖创建、读取、更新以及无效 target 的前后端测试；
- [ ] 更新 [Frontend Management Views](../frontend/management-views.md)，记录策略编辑能力和失败语义。

## 非目标

- 不改变 Agent retrieval 的授权规则；
- 不让 `source_agent_id`、`source_team_id` 参与策略推导；
- 不允许 Memory 管理页面因为策略设置而失去对所属 Workspace 全部记忆的观测能力；
- 不在本事项中实现未来跨用户 actor/Workspace 访问。
