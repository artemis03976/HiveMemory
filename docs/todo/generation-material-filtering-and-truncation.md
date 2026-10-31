---
title: Generation Material Filtering And Truncation
status: todo
owner: patchouli
scope: settlement-material-filtering-generation-view-limits-and-worth-saving-removal
code_paths:
  - src/hivememory/engines/perception/models.py
  - src/hivememory/engines/perception/material.py
  - src/hivememory/config/patchouli.py
  - src/hivememory/system/services/import_bus/ingestor.py
  - src/hivememory/system/services/import_bus/events.py
  - src/hivememory/core/protocol/gateway.py
  - integrations/claude-code/connector/claude_code/reader.py
  - configs/config.yaml
related_docs:
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/gateway/analysis.md
  - docs/system/import-bus.md
  - docs/ROADMAP.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/archive/todo/import-bus-real-use-observation.md
last_reviewed: 2026-10-30
---

# 生成材料的筛选与截断

## 问题

Claude Code 单个会话的上下文常达数十万 token，而每个话题送到提取模型的材料只有 1～2 万 token。2026-10-29 对真实导入做了逐段对账：读取会话记录损失的对话内容很少，缩减来自服务端两处。

1. **入口价值预判整轮丢弃材料**：Gateway 在只看到用户消息的情况下判断价值，判为不值得保存的轮次在结算时被整轮过滤。
   - Owner 决定（2026-10-29）：`worth_saving` 不得再影响材料去留。结算已于同日改为不按 `worth_saving` 筛选，当前事实见[感知与短期话题](../patchouli/perception.md)第 2 节。
   - Owner 决定（2026-10-30）：价值体系由 owner 重新设计，不在现有信号上增补；`worth_saving` 没有保留与否的问题，只做移除。
2. **一轮交互的内容被上限数字截断**：保留下来的轮次里，思考与工具步骤大多被截断，多数工具调用只剩工具名与参数，看不到结果。
   - Owner 决定（2026-10-30）：一轮交互的内容不得被任何一个上限数字截断，最多只允许单次工具调用的结果受限。
   - 现行生成视图的单轮上限 `max_turn_action_chars`，以及思考、参数的单条上限 `max_thought_chars`、`max_args_chars`，都违反这条规则，属于实现漂移。owner 于 2026-10-21 只决定把思考与外部工具纳入生成视图；这些上限是 [Import Bus 重设计计划](../archive/plans/v0.7.1-import-bus-redesign.md)第 4.10 节记录该决定时加入的候选默认值，后由[工具解析链路重写](../archive/plans/v0.7.1-tool-call-parsing-chain.md)沿用。单条工具结果的上限 `max_result_chars` 符合规则。

## 证据（2026-10-29）

样本是 2026-10-27 至 10-28 的 3 个 Claude Code 会话：31 个导入轮次，5 个话题，生成 5 条记忆。会话记录用连接器自己的 reader 解析；生成视图读自各话题落盘的记忆创建 Artifact，也就是当时实际交给提取模型的材料。

| 阶段 | 量 |
|:---|:---|
| Claude Code 上下文峰值 | 三个会话分别约 46 万、63 万、34 万 token；会话开头已有约 5 万 token 的系统提示、工具定义与项目指令 |
| 连接器导入的原文 | 179 万字符（项目估算器约 51 万 token）：工具结果 71%，工具参数 25%，助手正文 3%，思考与用户消息不到 1% |
| 按 `worth_saving` 结算过滤后 | 剩 17/31 轮、109 万字符；被丢 14 轮、69 万字符（39%） |
| 生成视图（各项上限截断后） | 共 14.6 万字符，为导入原文的 8% |
| 提取模型实际用量 | 5 个话题共约 7 万 token（含提示词模板与输出），每个话题 0.6 万～2.1 万 |

- **价值预判看不到本轮过程**：Import Bus 调用 Gateway 时只给本轮用户消息与最近几轮的正文，本轮的回答、工具调用与结果都不在其中。被丢的 14 轮里有工作量最大的几轮：
  - 「请单独提交，之后开始实现」：78 次工具调用，23.6 万字符，是话题折叠的整个实现过程；
  - 「请提交，并开始推理移出事件循环的 todo」：63 次，18.0 万字符；
  - 「请提交改动，并继续收尾」：30 次，9.3 万字符；
  - 「请先完成第一部分的内容」：30 次，8.0 万字符。

  因此这些话题生成的记忆只有设计讨论，没有实现过程。交互 Artifact 只保存进入结算的轮次，这 14 轮的原文在 HiveMemory 中没有留存，只剩 Claude Code 的本地会话记录；连接器不回填已确认的轮次。
- **截断**：进入生成的 17 轮中，12 轮顶到 `max_turn_action_chars`（8000 字符）。
  - 257 次工具调用中只有 68 次保留了结果，另有 38 个条目被整条省略；
  - 单条结果按 `max_result_chars`（800 字符）截断；
  - 提取模型的主要依据是助手正文与工具名、参数各一行的摘要。
- **读取会话记录本身的取舍**：连接器跳过的内容如下。
  - Claude Code 自己的指令材料（系统提示快照、工具清单、项目指令、技能清单），约 80 万字符；
  - 子代理的内部过程，其中一个会话约 1.6 MB；主线收到的子代理最终报告已作为工具结果导入；
  - 少量 IDE 诊断与引用文件，约 4 万字符。

  另外，思考正文基本不可得：模型多只记录签名，31 轮中可用的思考正文只有 6400 字符。这些取舍属于连接器的既定边界，不在本 Todo 内改动。

## 影响

- 停止筛选后，被判为不值得保存的轮次也进入生成，提取 token 与折叠频率会上升；删除单轮、思考与参数上限后，生成视图还会继续变大，单个轮次也可能超过提取模型的窗口。材料超窗时的消费方式是 v0.7.2 的待决问题（[Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md#03-待决问题) R-1）。
- 上限不删除时，进入生成的实现轮次主要贡献助手正文，测试失败、报错与改动内容等工具结果大多看不到。
- `worth_saving` 停用后不再参与任何决定，2026-10-30 起也已移出交互记录与话题 block；但 Gateway 仍为每一轮计算 `memory_write_signal`，由它得到的 `worth_saving` 仍随 `import.turn.accepted` 事件传递，只供观测。

## 完成条件

随 v0.7.2 的生成材料修正计划完成（设计讨论见 [Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md#0-v072生成材料修正与-topic-之后的原始证据保存)第 0 节），计划建立时本 Todo 并入计划：

- 生成视图只保留单次工具调用结果的上限，删除 `max_turn_action_chars`、`max_thought_chars`、`max_args_chars` 及对应的降级与省略规则；
- 移除 `worth_saving` 与只为它计算的传递链。
