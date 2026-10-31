---
title: 话题结算的多条提取、查重与更新
status: idea
horizon: current
serves_version: v0.7.2
owner: patchouli
scope: settlement-multi-memory-extraction-dedup-signal-and-update-mechanism
related_current:
  - docs/patchouli/generation.md
  - docs/patchouli/retrieval.md
  - docs/patchouli/memory-compiler.md
related_todos:
  - docs/todo/generation-dedup-decision-signal.md
related_ideas:
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
last_reviewed: 2026-10-30
---

# 话题结算的多条提取、查重与更新

## 0. 文档性质

本文是 v0.7.2（生成质量）的设计讨论，为两份计划供料：一个话题提取多条记忆，以及查重与更新机制。两份计划各自在阻塞问题决定、能够立即执行时建立，彼此不预设先后。当前行为以 [Patchouli 生成](../patchouli/generation.md)为准；查重失灵的证据与完成条件见[生成查重的决策信号 Todo](../todo/generation-dedup-decision-signal.md)。

同一版本的另外两个方向分别在其他 Idea 中讨论：生成材料修正与 Topic 之后的原始证据保存见 [Raw Evidence Idea](./PatchouliPageFoldingRawEvidenceDesign.md#0-v072生成材料修正与-topic-之后的原始证据保存) 第 0 节；plugin 模式开放 WRITE 与 UPDATE 见[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#v072plugin-模式开放-write-与-update)。

## 1. owner 决定（2026-10-30）

- v0.7.2 的版本目标是生成质量，版本内要完成一个话题提取多条记忆的能力。
- 查重的新设计尚未决定，包括 UPDATE 的正确机制；第 4 节只列选项。
- owner 判断：Claude Code 真实使用中暴露的生成问题，约有一半是因为 plugin 模式没有接回 agent 对记忆的写入能力。该模式下 WRITE 与 UPDATE 从未发生，而主动链路中显式的 UPDATE 能正常更新记忆。v0.7.2 同步开放这两项能力，见外部 Actor Idea。

## 2. 问题

### 2.1 一个话题至多一条记忆

Mode A 的 extractor 对整段话题材料只返回一份草稿，同一话题中讨论过的多项决定、事实或偏好只能压进一条记忆，次要内容往往只留在正文里。检索的稠密与稀疏向量只由标题、类型、标签与摘要编译，只写在正文中的内容因此检索不到（[Patchouli 生成](../patchouli/generation.md#8-当前限制)第 8 节）。MCP 记忆访问的第一个检索落空样本，原因之一就是这一点（[已归档的观察 Todo](../archive/todo/import-bus-real-use-observation.md)）。

### 2.2 查重分不开相关记忆，命中后整体覆盖

Todo 记录了 2026-10-29 的离线复现：用当前的嵌入模型做“记忆对记忆”的比较时，dense top-1 余弦区分不了“同一件事”与“同一项目里的其他事”，相关对与无关对的分数之间没有间隔；同一条记忆拿自己来比也够不到 TOUCH 的阈值。

Todo 登记时查重只做过 CREATE。2026-10-29 晚至 10-30，同一项目的记忆增多后，UPDATE 开始发生。Claude Code 插件接入（10-27）以来共 19 次查重决策：15 次 CREATE、4 次 UPDATE、没有 TOUCH。4 次 UPDATE 的相似度在 0.81～0.85 之间，构成两条链，每条都被连续覆盖了两次：

| 最初的记忆 | 第一次覆盖后 | 第二次覆盖后（当前版本） | 当前版本丢掉的内容 |
|:---|:---|:---|:---|
| v0.7.1 记忆访问设计决定：P-1c/P-9c/E-2/E-3b/E-7 与 harness principal 复核 | v0.7.1 记忆访问 MCP：阶段 1-2 实现与操作入口统一出口设计 | v0.7.1 记忆访问：操作入口单一出口与 actor 授权合并 | E-2、E-3b、E-7 等设计决定；当前正文只剩阶段二的返工 |
| v0.7.1 版本剩余工作梳理与收尾建议 | v0.7.1 收尾审计：Import Bus 归档、int8 移出与文档收口 | v0.7.1 版本收尾：提交、推送并创建 PR #148 | 剩余工作的分析与建议；当前正文中“合并后待办”在生成时即已过时 |

两条链都出自同一会话中先后进行的几段工作：空闲结算把它们拆成不同话题，后一段的草稿与前一段的记忆相似度落入 UPDATE 区间，按现行规则整体覆盖。被覆盖的版本保留在 MemoryVersionArtifact 中，检索与读取只看到最新版本。同一项目的记忆越多，这种情况越常见。

## 3. 现状事实（代码核对，2026-10-30）

- **提取**：`BaseMemoryExtractor.extract` 返回 `ExtractedMemoryDraft | None`（[生成接口](../../src/hivememory/engines/generation/interfaces.py)）；Mode B 以 `WriteFocus` 为核心，Mode C 以 `UpdateFocus` 为核心，各自只产出一份草稿。
- **查重**：Mode A/B 以 `title + summary` 对中期库做 dense top-1 检索；大于 0.95 时按内容是否一致判为 TOUCH 或 UPDATE，0.75～0.95 判为 UPDATE，其余判为 CREATE（[Deduplicator](../../src/hivememory/engines/generation/deduplicator.py)，阈值见 `configs/config.yaml` 的 `patchouli.generation.deduplicator`）。
- **两种 UPDATE**：查重判出的 UPDATE 用草稿内容整体覆盖已有记忆的 head；显式的 UPDATE（Mode C）用 extractor merge 生成新正文与 changelog，merge 失败时有追加或保留旧正文的回退（[Patchouli 生成](../patchouli/generation.md#33-mode-c主动-update)第 3.3、4 节）。
- **嵌入文本**：入库按 `Title/Type/Tags/Summary` 模板编译，查询文本是 `title summary`，两者形态不同（[嵌入处理器](../../src/hivememory/engines/memory_compiler/handlers/embedding.py)）。
- **来源**：一次结算捕获一份 InteractionArtifact，按生成结果挂载来源引用（[Artifacts](../patchouli/artifacts.md)）。

## 4. 待决问题

每个问题只列选项及其影响，不作选择；选项顺序不代表倾向。

### G-1 一个话题的多条记忆怎样提取

阻塞多条提取的计划。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | extractor 一次调用输出草稿列表 | 每个话题仍是一次模型调用；草稿之间的边界完全由模型决定；输出变长，解析失败影响全部草稿 |
| B | 先由模型列出话题中的议题，再逐个议题提取 | 每个话题多次模型调用；议题清单可以检查与记录，单条失败只影响该条 |
| C | 按材料的稳定范围（例如连续的轮次段）分片，逐片提取 | 与原始证据的分片读取一致，可以承接超出模型窗口的材料；同一议题跨片时会拆成多条，依赖查重合并 |
| D | 其他 | —— |

**owner（2026-10-30）**：选项 C 不予采纳。话题本来就是用来聚合同一语义的片段，再按范围分片生成相当于人为打散这一结构；理由与 [Raw Evidence Idea](./PatchouliPageFoldingRawEvidenceDesign.md#01-owner-决定2026-10-30) 第 0.1 节的 R-1 相同。G-1 仍在其余选项之间待决。

### G-2 查重的判定信号

阻塞查重与更新的计划。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 保留 dense top-1 与固定阈值，对齐查询与入库的嵌入文本后重新标定 | 不增加模型调用；离线复现显示相关对与无关对没有分数间隔，对齐文本只解决自相似够不到 TOUCH |
| B | dense top-k 召回后由已常驻的交叉编码器打分，按分数判定 | 不增加 LLM 调用；唯一一组记忆对样本中分开了相关与无关，问题式短查询上排序不稳定；原始分数会随重排模型量化改变（int8 暂不做） |
| C | dense 宽门槛召回 top-k，由 LLM 判定新建、更新、合并或只建立关联 | 每条草稿多一次模型调用；判定理由可以记录；需要输出契约与提供商失败时的回退 |
| D | 其他 | —— |

### G-3 查重命中已有记忆后的写入方式

阻塞查重与更新的计划。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 整体覆盖（现状） | 实现最简单；后一段工作的草稿会冲掉前一段仍然有效的内容（第 2.2 节） |
| B | 与显式 UPDATE（Mode C）共用 extractor merge | 被动与主动的更新是同一套逻辑；每次命中多一次 merge 调用，结果取决于 merge 质量 |
| C | 不改写已有记忆，新建一条并与之建立关联或替代关系 | 不丢内容；同一件事分散为多条，需要关联模型或读取侧聚合 |
| D | 按记忆类型区分写入方式（例如进行中的工作替换进度，事实与决定合并） | 规则变多；类型判错时退化为其他选项的问题 |
| E | 其他 | —— |

## 5. 候选设计（分析，不是 owner 决定，可在计划评审时调整）

- **两份计划的关系**：多条提取的计划不以 G-2、G-3 为前提，每条草稿逐条经过实施时的查重与写入；G-2、G-3 决定后由查重计划更换机制。若多条提取先实施，草稿变多会让查重命中更频繁，在 G-3 决定之前整体覆盖的影响也随之放大，计划需写明这一风险。
- **来源**：同一次结算产出的多条记忆共享这次结算冻结的来源 Artifact，各自挂载来源引用；一份来源可以支撑多条 Memory（[Raw Evidence Idea](./PatchouliPageFoldingRawEvidenceDesign.md#53-memory-的来源关联)第 5.3 节）。
- **评估样本**：从真实导入的记忆中整理一组标注样本，覆盖四类关系：重复、同一工作的不同阶段、同项目但无关、完全无关；第 2.2 节的两条覆盖链列入样本。G-2 的信号与 G-3 的写入方式在这组样本上比较。
- **docstring**：`MemoryDeduplicator` 把 0.75～0.95 区间写成“内容合并”，与整体覆盖的行为不符，随查重计划修正。

## 6. 不在本文

- 正文进入检索索引：只写在正文中的内容召回不到的另一条处理路径，搁置。
- 价值体系：由 owner 重新设计，见 [ROADMAP](../ROADMAP.md#47-记忆价值体系重新设计候选)。
- 重排模型 int8：未排期，见 [Todo](../todo/reranker-int8-and-onnx-memory.md)。

## 7. 升级为 Plan 的条件

- 多条提取的计划：G-1 已决定。
- 查重与更新的计划：G-2、G-3 已决定，标注样本已整理。

两份计划都以 `target: v0.7.2` 建立，在 [Plans 导航](../plans/README.md#v072)登记；实施、验收与代码审查完成后，按最终代码更新 [Patchouli 生成](../patchouli/generation.md)与相关事实文档。
