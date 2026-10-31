---
title: Generation Dedup Decision Signal
status: todo
owner: patchouli
scope: generation-dedup-candidate-scoring-thresholds-and-update-semantics
code_paths:
  - src/hivememory/engines/generation/deduplicator.py
  - src/hivememory/engines/generation/engine.py
  - src/hivememory/engines/memory_compiler/handlers/embedding.py
  - src/hivememory/infrastructure/embedding/bge_m3.py
  - configs/config.yaml
related_docs:
  - docs/patchouli/generation.md
  - docs/patchouli/retrieval.md
  - docs/archive/todo/import-bus-real-use-observation.md
  - docs/todo/reranker-int8-and-onnx-memory.md
last_reviewed: 2026-10-29
---

# 生成查重的决策信号

## 问题

Mode A/B 生成的草稿以 `title + summary` 对中期库做 dense top-1 检索，再按原始余弦与固定阈值决策：大于 0.95 时按内容是否一致判为 TOUCH 或 UPDATE，0.75～0.95 判为 UPDATE，其余判为 CREATE（[Patchouli 生成](../patchouli/generation.md)第 4 节）。

Claude Code 接入后的真实使用中，查重从未做出 CREATE 以外的决策，其中包括 owner 有意制造的一次同一工作的跨话题拆分。离线复现表明，用当前的嵌入模型做"记忆对记忆"的比较时，dense 余弦区分不了"同一件事"与"同一项目里的其他事"，固定阈值也够不到。

## 证据（2026-10-29）

真实使用：2026-10-27 至 10-28，Claude Code 共导入 31 轮，形成 5 个话题，生成 5 条记忆，查重全部判为 CREATE，Top-1 相似度在 0.493～0.653 之间。

离线复现：使用与服务端相同的嵌入配置（`Xenova/bge-m3` 的 `onnx/model_int8.onnx`，CLS 池化并归一化），对查重当时的中期库重算。当时的库包含现存 30 条和已归档 10 条，其中多数是早期测试数据。5 次查重的 Top-1 与日志一致，分数相差不超过 0.012。

- **噪声底**：40 条记忆按入库文本两两比较，余弦的 p50 为 0.451、p90 为 0.542、p99 为 0.655。日志中的 0.493 与 0.567 就是"库里没有相关记忆"时的 Top-1。
- **有意拆分的一组**：同一项工作（话题折叠移出交互应用）的前后两段相隔超过空闲时限，分别生成「话题折叠耗时不计入交互提交时限的设计决定」与「HiveMemory 感知层折叠与 journal 重构」。
  - 后者查重时前者已经在库中，但相关对只得 0.572，排第 2；Top-1 是不相关的「HiveMemory 推理模型常驻内存分析与空闲卸载方案」（0.653）。
  - 相关对的得分落在无关对的 p90 与 p99 之间。
  - 查询也套用入库模板（0.605）或两边都去掉模板（0.560），排名都不变。
- **自相似够不到 TOUCH**：查询文本 `title summary` 与入库文本 `Title/Type/Tags/Summary` 模板形态不同，同一条记忆拿自己来比只有 0.925～0.953，内容完全相同也难以超过 0.95。
- **交叉编码器**：已常驻的 bge-reranker-base 对同一组候选给出的分数（sigmoid）为：相关对 0.850，其余最高 0.485，其余中位数 0.002。但在 5 个问题式短查询上，它只把 3 个目标排在第 1，「InteractionApplyJournal 为什么删掉了」排到第 5。记忆对的样本只有一组，只能说明方向。
- **对照**：同样 5 个问题式短查询，dense 检索在这 30 条记忆中都把目标排在第 1（0.556～0.744）。问题集中在记忆对记忆的比较，不是检索整体失效；但库很小，结论不能外推到同项目记忆增多之后。
- **未测**：int8 与 fp32 嵌入模型的差异。

## 影响

- 同一工作被空闲结算或会话切换拆到不同话题时，后续阶段总是新建记忆，同一件事在库中分散成多条。Claude Code 的长任务常有较长间隔，这种拆分会是常态。
- 单靠调低阈值无法解决：
  - 0.75～0.95 区间的 UPDATE 会用草稿内容整体覆盖已有记忆。按上面那组样本，若判为 UPDATE，「设计决定」会被 code review 的内容替换掉。
  - dense 分数在相关对与无关对之间没有间隔，任何阈值在放进相关对的同时也会放进无关记忆。
- `MemoryDeduplicator` 的 docstring 把 0.75～0.95 区间写成"内容合并"，与实际的覆盖行为不符。

## 完成条件

- 从真实导入的记忆中整理一组标注样本，覆盖四类关系：重复、同一工作的不同阶段、同项目但无关、完全无关。
- 在这组样本上比较候选决策信号，按结果选定信号并重新标定阈值。候选例如：
  - dense Top-k 召回后用交叉编码器打分；
  - dense 宽门槛召回后，由 LLM 判定新建、更新、合并或只建立关联。
- 若用交叉编码器分数做阈值判定，需要与[重排模型 int8](./reranker-int8-and-onnx-memory.md)的评估一起考虑：int8 会改变原始分数。
- 放宽 UPDATE 判定之前，先让 UPDATE 保留已有记忆中仍然有效的内容（例如经 extractor 合并），不再整体覆盖。
- 相同内容能够判为 TOUCH：让查询与入库的嵌入文本形态一致，或改用不依赖绝对余弦的判定。
- 如果需要新的 extractor 提示词或改动跨子系统契约，升级为 Plan。
