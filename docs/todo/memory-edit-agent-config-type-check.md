---
title: 记忆编辑可以给非 Profile 记忆写入 agent_config
status: todo
owner: patchouli
scope: memory-edit-agent-config-restricted-to-agent-profile
code_paths:
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/patchouli/contracts/memory_management.py
  - src/hivememory/server/models/memory.py
  - frontend/src/services/agentApi.ts
related_docs:
  - docs/patchouli/generation.md
  - docs/contracts/routes-and-events.md
  - docs/archive/todo/v0.7.0-code-convergence.md
last_reviewed: 2026-10-18
---

# 记忆编辑可以给非 Profile 记忆写入 agent_config

## 问题与证据

`payload.agent_config` 只对 Agent Profile 有意义：它的唯一读取方是 `AgentProfile.from_atom`（`core/models/agent.py`），只解析 `AGENT_PROFILE` 原子的模型参数与语言。创建链按入口区分类型：记忆创建材料 `MemoryCreation` 不带 `agent_config`，只有 Profile 创建入口会传入。

编辑链没有这样的区分。记忆与 Profile 共用一条编辑路由（`patchouli.public.memory.update`，材料 `MemoryEdit`），`MemoryGenerationFamiliar._apply_external_update` 对任何类型都会应用 `agent_config`，类型约束 `_require_content_invariants` 只检查 Profile 必须带 alias。HTTP 的 `PATCH /memories/{memory_id}` 请求模型同样接受 `agent_config`。

2026-10-18 复现：对一条 FACT 记忆调用 `update_external_memory(..., agent_config={"model": "x"})`，返回的原子保留了该配置，版本由 1 变为 2。

## 影响范围

- 普通记忆上写入的 `agent_config` 不会被任何读取方使用，却作为内容变化生成新的 MANUAL_EDIT 版本记录，并出现在该记忆的完整内容与历史中；
- 调用方得不到“该字段对此类型无效”的反馈。

处理时需要保留现有的 Profile 编辑：前端保存 Agent 走的就是 `PATCH /api/v1/memories/{id}`（`frontend/src/services/agentApi.ts` 的 `saveAgent`），没有单独的 Profile 编辑入口。可能的处理是在 Familiar 的类型约束中拒绝对非 `AGENT_PROFILE` 记忆设置 `agent_config`（`InvalidMemoryFieldError`，HTTP 422）；是否顺带拆出 Profile 专用编辑入口另行评估。已经写入普通记忆的 `agent_config` 是否清理，也需在处理时决定。

## 完成条件

- 对非 `AGENT_PROFILE` 记忆提交 `agent_config` 的编辑被拒绝且不产生版本记录，Profile 的 `agent_config` 编辑照常生效；
- 归属层测试覆盖两种类型；
- [Patchouli 生成](../patchouli/generation.md)中手工编辑的约束描述随实现更新。

## 追踪

- 2026-10-18：为记忆编辑引入 `MemoryEdit` 材料类型时发现（[v0.7.0 代码收敛清单](../archive/todo/v0.7.0-code-convergence.md)第 2.9 条），属于行为变更，不随该条处理；未绑定版本或 Issue。
