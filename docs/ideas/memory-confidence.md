---
title: 记忆置信度：起始值与浮动
status: idea
horizon: candidate
owner: patchouli
scope: memory-confidence-initial-value-and-event-driven-updates
code_paths:
  - src/hivememory/core/models/memory.py
  - src/hivememory/engines/generation/engine.py
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/engines/lifecycle/reinforcement.py
  - src/hivememory/engines/retrieval/
  - src/hivememory/engines/memory_compiler/handlers/prompt.py
  - src/hivememory/i18n/prompts.py
related_docs:
  - docs/ideas/memory-type-system.md
  - docs/ideas/VitalityScoringLongTermEvolutionIdeas.md
  - docs/patchouli/lifecycle.md
  - docs/patchouli/generation.md
last_reviewed: 2026-11-22
---

# 记忆置信度：起始值与浮动

本文是候选方向：让置信度重新成为一个有信号的浮动值。置信度回答“这条记忆凭什么可信”，与生命力回答的“还值不值得留”分开；生命力的演进见[生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md)，当前的强化事件与提交规则以[记忆生命周期](../patchouli/lifecycle.md)和[记忆生成](../patchouli/generation.md)为准。

## 1. 观察到的问题与证据

**最初的设计**：置信度是一个浮动值。创建时按来源给出预设的起始值，之后随验证与使用浮动；低置信的草稿不得覆盖高置信的旧记忆（[类型体系](./memory-type-system.md)第 2 节）。浮动依赖的生命周期维护一直没有开发，置信度因此停在一个尴尬的状态：有使用方，没有信号。

**现状**（以代码为准）：

- 起始值由模型给出，不按来源预设。被动生成与馆长的提示词只有一句判断标准（用户陈述或工具结果证实的高，AI 推理的低）；主动写入的提示词要求一律写 1.0；WRITE 解析失败时的兜底草稿写死为 1.0（`_build_fallback_draft`）。
- 只能往上走：任何一次 UPDATE 提交（馆长修订、MCP 更新、合并）都把置信度重置为 1.0，与修订依据的来源无关。往下只有负反馈（×0.5），而负反馈没有发出方。
- 使用方：检索排序中的置信度加成、`min_confidence` 过滤，以及注入给 agent 的索引。索引在 0.9 及以上时显示“高”，低于 0.7 且没有验证状态时显示“未验证”。

日常实例的分布（2026-11-22，只读统计中期库）：

| 置信度 | 条数 | 其中修订过（version > 1） |
|:---|---:|---:|
| 1.0 | 268 | 170 |
| 0.9–0.99 | 420 | 39 |
| 0.7–0.89 | 63 | 2 |
| 低于 0.7 | 0 | 0 |

**影响**：

- 751 条中 92% 向 agent 显示为“高”，没有一条显示为“未验证”。展示出去的置信度不能帮读者区分该信多少，反而是噪声；解析失败后退化的兜底记忆同样显示“100%（高）”。
- 修订一律重置为 1.0，抹掉了最初设计里两条规则的前提：用户显式输入不可被 agent 覆盖；低置信不覆盖高置信。“被改过”并不等于“被验证过”：馆长修订的依据常常就是 agent 的推理材料。

## 2. 假设与潜在价值

- **起始值由来源推出，而不是由模型临场打分。** 来源与[类型体系](./memory-type-system.md)第 7.3 节“说过的 / 归纳的”来源标记、第 7.2 节的“真相所在”是同一件事。
- **浮动由带外部证据的事件推动。** 没有发出事件的一方，置信度要么卡住不动（现状），要么自己漂移。
- **价值**：
  - 给 agent 一个能区分的采信信号；
  - 给维护一个依据，例如冲突时谁让步、合并时以哪条为底、低置信的草稿能不能覆盖旧记忆；
  - 与验证状态配合，让“已确认”“已过时”有写入方。

## 3. 可能的路径（候选）

按从简到繁排列，可以只做前几项：

1. **在有信号之前不展示**：置信度暂不注入给 agent，也不参与检索排序，只保留存储。改动最小，但不解决问题本身。
2. **修订不再一律重置为 1.0**：修订保留原值，或取原值与修订依据来源二者中较低的一个；用户本人的陈述不被 agent 的修订降级。
3. **起始值按系统可知的来源预设**：先只用不靠模型判断的来源，即主动写入还是被动结算、内容是否出自用户轮次的原话；其余仍由模型给出。
4. **补上事件的发出方**：
   - 馆长 `revise_memory` 带上修订原因：“当初记错了”对原记忆是负信号，“后来情况变了”不是（[类型体系](./memory-type-system.md)第 4 节第 3 点）。不增加模型调用。
   - 资源记忆的机械检查，即路径、符号还在不在，网址能不能打开（[类型体系](./memory-type-system.md)第 7.1 节候选），作为验证状态的写入方。
   - 用户在后续对话中的确认与纠正。

## 4. 主要未知项与验证方式

- 来源标记怎样判：“说过的”可以核对原话是否出现在用户轮次中，“归纳的”仍要靠模型；类型体系中的来源标记尚未实施。
- 置信度按条还是按段给出：[类型体系](./memory-type-system.md)第 7.2、8 节倾向于逐段，但存储与展示现在都是整条。
- 存量记忆怎样处理：按新规则重算起始值，还是保留现值、只对新写入生效。
- 验证：
  - 分布不再塌缩，可以复用上面的只读统计；
  - 来源判定在回放中的稳定性，用 `scripts/memory_replay.py`；
  - 展示之后 agent 的采信行为是否改变，目前测不了。

## 5. 升级为 Plan 的条件

- 确定起始值的来源规则，至少覆盖系统可知的那部分（第 3 节第 3 项）。
- 至少有一个下行事件的发出方。
- 有存量记忆的处理方案，以及对检索排序与 `min_confidence` 过滤的影响评估。
