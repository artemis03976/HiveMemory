---
title: 导入轮次被落进另一个外部会话的话题
status: todo
owner: gateway
scope: import-topic-routing-across-external-conversations
code_paths:
  - src/hivememory/engines/gateway/topic_router.py
  - src/hivememory/gateway/context.py
  - src/hivememory/system/services/import_bus/
related_docs:
  - docs/gateway/analysis.md
  - docs/system/import-bus.md
  - docs/patchouli/generation.md
  - docs/ideas/generation-source-convergence.md
last_reviewed: 2026-11-03
---

# 导入轮次被落进另一个外部会话的话题

## 问题与证据（2026-11-03）

Import Bus 为导入轮次与写入时的在途轮次落位，经 Gateway 的话题路由在本 Workspace 的驻留话题中选择（[Gateway 分析](../gateway/analysis.md#3-topic-router只做选择)第 3 节，[Import Bus](../system/import-bus.md#47-写入时落位)第 4.7 节）。候选话题不区分来源与外部会话，外部会话的第一轮没有 `session_turns`，路由只凭这一轮的用户消息与候选话题的标题、摘要判断。

2026-11-02（UTC 11-03）的一次实例：

- dsh 新会话的第一轮是「请使用 repo-deep-dive 的 standard 模式，对 dsh-agent-soul 进行深度调研」。它在第一次 `memory_write` 时落位（UTC 02:24:53），被路由到已有话题 `45fe6125`（`new_topic=False`）。
- 这个话题来自同一时间的另一段 Claude Code 会话，标题是「dsh 调研记忆质量评估与 repo-deep-dive skill 改进」，与这一轮只在 dsh、repo-deep-dive 两个词上重合，内容无关。
- 话题结算（UTC 03:24）时，生成材料混合了 4 轮 Claude Code 与 1 轮 dsh，一次提取用了约 17.5 万 token。
- 这一轮 dsh 的 9 条主动写入，正式记忆的 `contributing_agent_ids` 都多了 `claude_code`，Claude Code 并没有参与这次调研。

证据：本机 `.hivememory/evidence/2026-11-03-dsh-research/`（不入库）中的后端日志摘录与记忆导出，以及该话题的来源 InteractionArtifact。

## 影响

- 两段无关对话合进一次提取，材料变长，提取的取舍受另一段对话影响；这次结算没有提取调研的结论，只提取了一条调研轮次里的环境事实（[收敛 Idea](../ideas/generation-source-convergence.md) 第 2 节）。
- 主动写入的贡献者被话题中的其他参与者污染，来源追踪失真。
- 同一个 Workspace 中有多个 harness 或多段会话同时工作时，话题名里的通用词（项目名、工具名）越多，这种误判越容易发生。

## 待决定

一个话题能否跨外部会话，是路由规则的问题，不只是模型判断失误。可能的方向：

- 外部会话的第一轮只有在用户消息明确延续某个已有话题时才续接，否则新建；
- 候选话题的快照带上参与过的来源与外部会话，由路由提示词区分；
- 维持现状，只改路由提示词。

跨会话延续同一件事（例如在两个 harness 之间切换做同一项工作）是正常用法，选项需要说明它是否仍然成立。

## 完成条件

- owner 选定规则；
- 外部会话第一轮的落位覆盖：内容无关但标题有通用词重合时新建话题，明确延续时续接；
- Gateway 分析与 Import Bus 文档写明候选话题与外部会话的关系。

## 非目标

- 话题内多条记忆的查重与合并，见[收敛 Idea](../ideas/generation-source-convergence.md)；
- 主动写入的贡献者计算方式本身。
