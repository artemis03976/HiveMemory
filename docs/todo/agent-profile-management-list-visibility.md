---
title: Profile 管理列表的可见性语义
status: todo
owner: patchouli
scope: agent-profile-management-list-owner-semantics
code_paths:
  - src/hivememory/patchouli/application/agent_profile_management_service.py
  - src/hivememory/workspace/capability/root/agent_profiles.py
related_docs:
  - docs/contracts/routes-and-events.md
  - docs/system/application-services.md
last_reviewed: 2026-10-16
---

# Profile 管理列表的可见性语义

## 问题与证据

[路由与事件契约](../contracts/routes-and-events.md)规定，Memory、Agent Profile 与 Topic 的管理读取按 owner-management 语义执行：在 Workspace 归属硬边界之内，读取该 Workspace 的 `PUBLIC/PRIVATE/TEAM` 全部记忆，不执行 Agent 可见性过滤。

Memory 管理读取显式遵守这一语义：`MemoryManagementService` 的 `list_memories` 与 `get_memory` 都传入 `enforce_actor_visibility=False`。Profile 管理列表 `AgentProfileManagementService.list_agent_profiles` 却没有传这个参数，本地路由 `MEMORY_LIST` 的默认值是 `True`。root API 的 Profile 列表以保留的 `system` 为可见性主体，因此只能看到 `PUBLIC` Profile。

2026-10-16 整理 application 薄转发测试时，用真实本地总线、`RetrievalFamiliar` 与 Qdrant 内存存储复现了这一差异：

- 同一 Workspace 内放入 PUBLIC Profile、PRIVATE Profile 与 PRIVATE 事实记忆；
- 以 `system` scope 调用 Memory 管理列表，三者都能列出；
- 以同一 scope 调用 Profile 管理列表，只返回 PUBLIC Profile。

原单测曾按精确参数断言这次转发，等于把缺少的参数固定成了预期；该断言已改为只检查 Profile 类型过滤。

## 影响范围

- root API 的 Profile 列表，以及经它展示的前端 Agent Management 页面：PRIVATE 与 TEAM Profile 不出现在管理列表中；
- Profile 的创建、读取与 CALL 解析不受影响：CALL 解析按调用方 Actor 的可见性执行，这是预期行为。

## 完成条件

- Profile 管理列表按 owner-management 语义列出 Workspace 内全部 Profile，或在契约中写明 Profile 列表不同于 Memory 管理读取的理由；
- `tests/unit/patchouli/application/test_agent_profile_management_service.py` 的列表用例断言 owner-management 可见性语义；经真实 handler 的回归覆盖 PRIVATE Profile 出现在管理列表中；
- [System 应用服务](../system/application-services.md)中 root Profile 管理的描述与实现一致。
