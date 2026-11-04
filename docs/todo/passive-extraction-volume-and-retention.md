---
title: 多条提取后被动提取的记忆膨胀与低留存
status: todo
owner: patchouli
scope: mode-a-memory-volume-and-retention-after-multi-extraction-under-manual-maintenance
code_paths:
  - src/hivememory/engines/generation/engine.py
  - src/hivememory/engines/generation/extractor.py
  - src/hivememory/i18n/prompts.py
related_docs:
  - docs/patchouli/generation.md
  - docs/archive/plans/v0.7.2-multi-memory-extraction.md
  - docs/ideas/unified-interaction-memory.md
  - docs/ideas/generation-source-convergence.md
  - docs/todo/generation-dedup-decision-signal.md
last_reviewed: 2026-11-03
---

# 多条提取后被动提取的记忆膨胀与低留存

## 问题与证据（2026-11-03）

owner 的最终目标是全自主记忆维护：记忆的新建、合并、删除与整理尽量不需要人介入（owner，2026-11-03）。现状离这个目标很远：多条提取（v0.7.2，[归档计划](../archive/plans/v0.7.2-multi-memory-extraction.md)，2026-11-01 合并）之后，记忆库膨胀明显，owner 已手动删除多批被动提取（Mode A）产出的记忆，2026-11-03 一天就删了 30 条。

按 memory_creation Artifact 统计（共 256 次新建，日期为 UTC）：

| | 多条提取之前 | 多条提取之后（4 天） |
|:---|:---:|:---:|
| 每次产出记忆的结算，Mode A 平均新建条数 | 1.00 | 5.20 |
| Mode A 新建总数 | 28 | 182 |
| 其中仍在库 | 26（93%） | 94（52%） |
| 其中被合并 | 0 | 31（17%） |
| 其中被删除 | 2 | 57（31%） |
| Mode A 每条正文平均长度 | 1648 字 | 1016 字 |
| 主动写入新建，其中仍在库 | 5，1 | 38，29（76%） |

- **提取的草稿数**：日志中 41 次 Mode A 提取的草稿数，0 条 7 次，1～4 条 10 次，5～6 条 13 次，7～10 条 11 次，中位数 5。
- **两个因素叠加**：每次结算的产出从约 1.6 千字变为约 5.3 千字。使用量同时上升，dsh 调研期间每天约 15 次结算，之前约 4 次。
- **最近的样本**：11-04（UTC）Claude Code 开发会话的 3 次结算新建 18 条 Mode A 记忆，当天被 owner 删掉 12 条。
- **对照**：主动写入的留存（76%）明显高于被动提取（52%），被动提取多出的部分大多是 owner 判断不值得保留的。

统计方法：

- 新建按 memory_creation Artifact 计数，删除与合并后 Artifact 仍保留；
- "被合并"指某次 MERGE 版本的来源中有它的 creation Artifact；
- "仍在库"指中期库中现有的记忆；
- 其余为"被删除"。这段时间没有记忆被生命周期整理归档，系统也不会自动删除，"被删除"基本等于 owner 判断不值得保留。

统计脚本是临时脚本，与结果保存在本机证据目录 `.hivememory/evidence/2026-11-03-dsh-research/`（不入库）。

## 影响

- **人工维护**：膨胀靠 owner 定期手动清理，与全自主维护的目标相反，记忆越多越难维持。
- **检索与查重**：低价值的被动记忆进入检索候选；同项目的记忆越多，dense top-1 查重越难分开同一件事与相关的其他事（[查重 Todo](./generation-dedup-decision-signal.md)）。
- **合并**：被动记忆的 17% 最终被合并，合并越多，保留的那条正文越长、摘要越难覆盖要点。

## 可能的方向（未决定）

- **提取规则**：收紧拆分，例如只为之后会被单独查找的决定、事实与经验成条，过程与讨论不成条；或要求每条草稿说明之后什么时候会用到它。
- **输出数量**：限制一次结算的草稿条数。这限制的是输出，与"生成材料不按轮次截断"无关。
- **价值反馈**：等记忆价值体系重新设计（[ROADMAP](../ROADMAP.md) 第 4.7 节），把留存信号（被删、被合并、被检索命中）作为价值反馈，替代人工清理。
- **断言整理**：在[统一的交互记忆](../ideas/unified-interaction-memory.md)中，推断的断言默认地位低于第一当事人的断言，被检索或被确认后再提升；第一当事人"没写"是否也是一种判断（该 Idea 第 5 节）直接决定被动提取补回多少内容。

这些方向可以合进任一 Idea 或计划；本 Todo 先保留证据与度量。

## 完成条件

- 每批调研或较长的开发会话之后，可以用同一方法重跑统计，把 Mode A 每次结算的新建条数与留存率追加到上表，作为对照基线；
- 选定并实施降低膨胀的做法后，用同一方法复测，写明变化；
- 或者本 Todo 被某份 Idea 或计划吸收，届时删除本文，并在[归档索引](../archive/todo/README.md)登记去向。

## 非目标

- 已有记忆的清理与拆分，由 owner 经管理入口执行；
- 查重信号本身（见查重 Todo）；
- 主动写入的生成方式。
