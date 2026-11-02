---
title: Agent Profile 以 ModelSelection 表达模型选择
status: todo
owner: alice
scope: agent-profile-model-selection-as-model-selection
code_paths:
  - src/hivememory/core/models/agent.py
  - src/hivememory/core/models/model_definition.py
  - src/hivememory/agent_runtime/model/resolution.py
  - src/hivememory/server/models/agent.py
  - frontend/src/types/agent.ts
related_docs:
  - docs/alice/agent-runtime.md
  - docs/archive/plans/llm-config-retirement.md
last_reviewed: 2026-11-01
---

# Agent Profile 以 ModelSelection 表达模型选择

## 问题与证据

模型选择在会话请求与 Gateway、Patchouli 的组件配置中已统一为 `ModelSelection`（`core.models.model_definition`；[归档计划](../archive/plans/llm-config-retirement.md)），Agent Profile 仍以平铺字段表达：

- [`AgentProfile`](../../src/hivememory/core/models/agent.py) 有 `model_name`、`temperature`、`top_p`，没有 `max_tokens`；
- Alice 的 [`resolve_frame_model`](../../src/hivememory/agent_runtime/model/resolution.py) 按会话优先于 Profile 的顺序，把这三项逐项拼进一份 `ModelSelection`；
- 同一个概念（注册表模型 ID）在 Profile 中叫 `model_name`，在 `ModelSelection` 中叫 `model_id`；
- Profile 持久化在 Profile 记忆原子的 `payload.agent_config`，经 Agent 管理接口（[`server/models/agent.py`](../../src/hivememory/server/models/agent.py)）与前端 Agent 编辑页读写（[`frontend/src/types/agent.ts`](../../frontend/src/types/agent.ts)）。

## 影响范围

- Agent Profile 不能覆盖 `max_tokens`，Agent 的输出上限只能由会话请求或模型定义决定；
- 使用方对模型的选择仍有两种形状，Alice 内部要逐项拼装；
- 改动涉及 `AgentProfile`、`agent_config` 的读写与已存 Profile 的读取、内置 Profile、Agent 管理接口与前端 Agent 编辑页。

可能的处理（未决定）：`AgentProfile` 以一个 `ModelSelection` 字段承载模型选择，`agent_config` 改为嵌套写法；读取时仍接受已存的平铺写法，或一次性改写已存的 Profile 原子。

## 完成条件

- `AgentProfile` 以一个 `ModelSelection` 表达模型选择（含 `max_tokens`），Alice 把会话的选择叠加在 Profile 的选择之上，不再逐项拼装；
- 已存 Profile 的平铺 `agent_config` 照常读取，或已迁移；
- Agent 管理接口与前端 Agent 编辑页随之更新；
- 在 Profile 模型与 Alice 帧模型解析的测试层覆盖上述行为，[Agent Runtime](../alice/agent-runtime.md#4-模型选择与-mtp-生成器)第 4 节按最终行为更新。

## 追踪

- 2026-11-01：模型选择统一计划实施时登记，owner 决定另行处理。未排期，未绑定 Issue。
