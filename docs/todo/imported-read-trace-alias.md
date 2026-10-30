---
title: 导入的 MCP 读取在轨迹中显示 alias
status: todo
owner: system
scope: imported-mcp-read-trace-target
code_paths:
  - src/hivememory/core/models/interaction.py
  - src/hivememory/agent_runtime/loop.py
  - src/hivememory/prompts/transcript/actions.py
  - integrations/claude-code/connector/claude_code/reader.py
related_docs:
  - docs/system/import-bus.md
  - docs/patchouli/generation.md
last_reviewed: 2026-10-29
---

# 导入的 MCP 读取在轨迹中显示 alias

## 问题与证据

共用封口从动作归约轨迹，READ 轨迹的目标取动作的 `tool_name`（[`interaction.py`](../../src/hivememory/core/models/interaction.py) 的 `_action_to_trace`），生成视图对带 `tool_kind` 的动作只渲染这一行轨迹：

- Alice 的 MTP READ 把命令目标（所读的 alias）写为 `tool_name`（[`loop.py`](../../src/hivememory/agent_runtime/loop.py)），轨迹显示读了哪些记忆；
- Claude Code 经 MCP 读取时，导入事件的 `tool_name` 是工具调用名 `mcp__plugin_hivememory-connector_hivememory__memory_read`，所读的 alias 在参数 `aliases` 中。连接器只设置 `tool_kind`，不改写工具名，生成视图因此只显示 `READ: mcp__plugin_hivememory-connector_hivememory__memory_read`，看不出读了哪些记忆。

SEARCH 不受影响：轨迹的查询取自参数 `query`，MCP 检索工具的参数同名。

2026-10-29 实施记忆访问（MCP）计划阶段四时发现。

## 影响范围

- 生成材料与话题折叠的接力摘要中，外部 harness 的读取轨迹缺少目标，生成模型不知道这一轮参考了哪些已有记忆；
- 不影响回声去重：读取结果照常不渲染；不丢失数据，交互 Artifact 保留原始事件。

可能的处理：

- 轨迹归约对 READ 在工具名之外读取参数中的 alias 列表。不改导入契约，需确认 MTP READ 的轨迹不变；
- 或由连接器把 alias 写入导入事件。导入事件目前拒绝 `target` 等字段，需要扩展导入契约。

## 完成条件

- 导入的 MCP 读取在生成视图的轨迹行中显示所读的 alias；
- MTP READ 的轨迹不变；
- 回归覆盖导入的 READ 动作归约出带 alias 的轨迹；
- [Import Bus](../system/import-bus.md) 与 [Patchouli 生成](../patchouli/generation.md)按最终行为更新。

## 追踪

- 2026-10-29：记忆访问计划阶段四审查时发现，owner 要求之后单独处理。未排期，未绑定 Issue。
