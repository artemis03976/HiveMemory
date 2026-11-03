---
title: 主动写入与被动提取的收敛
status: idea
horizon: candidate
owner: patchouli
scope: active-write-and-passive-extraction-overlap-merge-precedence-and-merge-conflicts
code_paths:
  - src/hivememory/engines/perception/material.py
  - src/hivememory/engines/generation/
  - src/hivememory/i18n/prompts.py
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/patchouli/control/memory_generation/
related_docs:
  - docs/patchouli/generation.md
  - docs/patchouli/memory-library.md
  - docs/todo/generation-dedup-decision-signal.md
  - docs/ideas/pending-intent-migration.md
  - docs/archive/ideas/settlement-multi-memory-extraction-and-dedup.md
  - docs/research/deepseek-harness-plugin-survey.md
last_reviewed: 2026-11-02
---

# 主动写入与被动提取的收敛

## 0. 文档性质

本文是候选方向，不描述当前行为，也不代表排期。主动写入、话题结算的被动提取、查重与合并的当前设计见 [Patchouli 生成](../patchouli/generation.md)第 2.1、3、4、7、8 节。查重的判定信号由[查重决策信号 Todo](../todo/generation-dedup-decision-signal.md)跟踪，本文不讨论信号本身；写入意图的登记、派发与结算见[写入意图 Idea](./pending-intent-migration.md)。

版本归属未定。owner 2026-11-02 提到，如果较为紧迫，可以让 v0.7.3 或 v0.8 回到记忆主线；v0.7.3 的版本目标仍未决定（[ROADMAP](../ROADMAP.md) 第 4.3 节）。

## 1. 背景

owner 对记忆来源的定位与判断：

- （2026-10-31）记忆主要由执行者经记忆工具主动写入，因为只有执行者知道什么是一手且重要的；话题结算的被动提取（Mode A）作为兜底（[已归档的多条提取与查重 Idea](../archive/ideas/settlement-multi-memory-extraction-and-dedup.md#1-owner-决定)第 1 节）。
- （2026-11-01）多条提取定案时确认：主动写入的内容被 Mode A 再提取一遍是早已接受的代价，由查重与合并完成去重与迭代更新（同上）。
- （2026-11-02）复盘 dsh 的调研记忆后：被动提取对主动写入再做一轮合并是已知缺陷，确实会导致记忆生成的质量降级；合并失败后留下重复本是已知行为，但同样让记忆库变乱；这些改动都涉及记忆库的底层机制，适合写成 Idea 而不是 Todo。

现行设计接受这份代价，依赖三条前提：

1. 显式写入与更新不渲染进生成材料，Mode A 就不会把它们的内容再提取一遍（[生成](../patchouli/generation.md#21-材料与渲染)第 2.1 节）；
2. 查重命中的两条记忆地位对等，合并时模型按内容更新时间得知哪条较新，状态冲突时以较新的为准（第 4、4.1 节；提示词见 `i18n/prompts.py` 的合并提示）；
3. 先落库再合并，关闭、队列拒绝或合并失败最多留下两条相似的记忆，不会丢失内容（第 4 节）；并发合并同一条记忆时后写回的失败，已列为当前限制（第 8 节）。

此前的真实使用以 Claude Code 开发 HiveMemory 为主，plugin 模式的主动写入在 v0.7.2 才开放，样本里主动写入很少。2026-11-02 的 dsh 调研是第一批以主动写入为主的样本：agent 按 repo-deep-dive 与 HiveMemory 使用手册 skill 在调研结束时写入结论，子 Agent 只读。三条前提在这批样本上都出现了偏差（第 2 节）。

## 2. 证据（2026-11-02，dsh 调研）

### 2.1 样本与来源

- **调研对象**：`jack66g/spatial-memory-engine`（SME）调研两轮，两轮之间 owner 改过 skill，第二轮为 standard 档，另有两轮追问；`doiito/gliding_horse`（Gliding Horse）为 deep 档，另有一轮追问。
- **统计范围**：SME 第一轮与第二轮在 PR #168（[已归档的子 Agent 会话计划](../archive/plans/dsh-subagent-sessions.md)）部署之前进行，子 Agent 会话被独立导入，生成了以子任务命名的话题与记忆。下文的统计只取主会话话题（SME 第二轮与 Gliding Horse），子 Agent 话题的记忆只在附录中处置。Gliding Horse 在修复部署之后。
- **数据来源**：
  - 中期库的当前原子；
  - `memory_creation` 与 `memory_version` Artifact：来源意图、每个版本的来源与被吸收的记忆，以及 Mode A 的完整生成视图；
  - dsh 会话日志中 agent 的 `memory_write` 原始参数；
  - 后端日志的提取、查重、存储与合并记录。日志用 WSL 本地时间（UTC−6），下文统一换算为 UTC。
- **证据包**：本机 `.hivememory/evidence/2026-11-02-dsh-research/`，不入库。删除记忆只移除中期库原子，创建与版本记录仍在 `.hivememory/artifacts/` 中，可以继续回放。

### 2.2 Mode A 复现了主动写入的全集

| 话题 | agent 主动写入 | 结算时 Mode A 的草稿 | 对应关系 |
|:---|:---|:---|:---|
| SME 第二轮主话题（UTC 18:34 结算） | 6 条：架构、空间分区、零漂移兼容、排序层、持久化、记忆生成 | 6 条 | 一一对应，相似度 0.786～0.872，全部判为 SIMILAR |
| SME 记忆动力学追问（UTC 18:54 结算） | 1 条 | 1 条 | 0.843，SIMILAR |
| Gliding Horse 主话题（UTC 23:36 结算） | 4 条：架构、证据收据门、收尾门、工作区租约 | 4 条 | 一一对应；3 条 SIMILAR（0.751～0.841），收尾门与已合并的混合记忆只有 0.683，判为 DISTINCT 另立一条 |

第 1 条前提在字面上成立：Gliding Horse 的生成视图有 210 个条目、约 34.5 万字符，`memory_write` 调用确实没有渲染。但 Mode A 用的是同一份材料：

- agent 写记忆之前的 THOUGHT 列出了计划写的标题，4 条 Mode A 草稿中有 3 条的标题与之相同或只差一两个字；
- 报告正文作为 `write` 工具的参数完整在材料里（约 2.8 万字符）；
- 6 个子 Agent 交回的输出约 8.9 万字符。

不渲染写入只防住了原文回声，防不住同一批结论被重新推导一遍。agent 认真调研、把结论写成记忆时，话题材料里本来就包含每一条结论，Mode A 因此产出同一组记忆。

这三个话题上，Mode A 没有贡献主动写入之外的新记忆（唯一另立的收尾门与已有内容重复）。它的实际作用是触发 10 次合并（1 次失败，见第 2.5 节），改写了 9 条 agent 写入的记忆（第 2.3 节）。

### 2.3 合并按新旧定主次，被动草稿成了骨架

同一话题的被动草稿总在空闲结算之后产生，必然比主动写入新。被动草稿并入主动写入、且成功写回的 10 次合并中（第 2.2 节的 9 次，加 1 次来自修复前的子 Agent 话题），合并记录的 changelog 都以"较新的 N"为准、为主干或为骨架。示例：

- 持久化：「以较新的风险清单记忆为准补充了带行号的风险与未决问题章节」，又一次「以较新的 N 为骨架补入 M 的完整风险清单」；
- 排序层：「以较新的 N 为主干补入 M 的定位、设计取舍、权重表」；
- 证据收据门：「以较新的 N 的收据门表述……替换并细化 M 中对应小节」。

篇幅随之增长（字符数，v1 是 agent 写入生成的首版）：

| 记忆 | v1 | 当前 |
|:---|---:|---:|
| SME 持久化协议 | 2358 | 4812 |
| SME 记忆动力学 | 3138 | 4438 |
| SME 记忆生成 | 2534 | 3581 |
| SME 零漂移兼容 | 1854 | 2575 |
| SME 空间分区 | 2299 | 2783 |
| SME 排序层 | 2136 | 2479 |
| SME 架构 | 3336 | 3675 |
| Gliding Horse 工作区租约 | 2244 | 2630 |
| Gliding Horse 证据收据门 | 1732 | 6069（含第 2.4 节的两次主动写入合并） |

第 2 条前提把"较新"当作信息更新的信号；对同一份材料上的主动写入与被动草稿，较新只是流水线时序的结果。合并因此系统性地以被动草稿为主体，把子 Agent 输出里的风险清单、行号与未决问题带进了 agent 有意精简过的记忆。

### 2.4 同一项目的主动写入在写入时互相合并

Gliding Horse 的 agent 有意写成 1 条架构记忆加 3 条机制记忆，实际结果：

- 架构记忆与证据收据门以 0.768 判为 SIMILAR 并合并，收尾门又以 0.772 并入；
- 工作区租约为 0.671，保持独立；
- 结果是一条 6069 字符、同时包含架构、收据门、收尾门与未接线能力的记忆；
- owner 之后明确要求单独写一条 Agent OS 记忆，它以 0.767 命中第 2.5 节的合并失败残留，被并了进去。

SME 第二轮的主动写入之间为 0.714～0.724，都判为 DISTINCT。同项目的不同记忆正落在阈值两侧，这是查重 Todo 所说的"同项目但无关"一类；新增的样本已登记在该 Todo。它与本文相关是因为第 3 节的选项会决定同一轮内的主动写入之间是否合并。

### 2.5 合并冲突

2026-11-01 起两份后端日志共有 147 次查重决策：29 次 SIMILAR、118 次 DISTINCT。合并失败 2 次，模式相同：同一次结算的两条草稿命中同一条记忆 M，前一个合并写回 M 1～2 秒后，后一个合并失败。

| 结算 | 命中 M 的两条草稿 | 前一个合并写回 | 失败 | 残留 |
|:---|:---|:---|:---|:---|
| SME 持久化子 Agent 话题（UTC 17:56） | 0.845、0.760 | 17:56:54 | 17:56:55 | 独立的"存储与持久化调研结论" |
| Gliding Horse 主话题（UTC 23:36） | 0.771、0.751 | 23:36:49 | 23:36:51 | 独立的"Agent OS 架构与定位"，之后吸收了 owner 要求单独写的 Agent OS 记忆 |

一次结算提取多条草稿之后，多条草稿命中同一条已有记忆不再罕见，第 8 节"并发合并同一条 M"的限制会成为常态。失败原因无法从日志直接确认：控制器只记录通用文本"记忆生成失败，请检查服务状态"，`MemoryMergeAbortedError` 的 `reason` 没有写进日志或任务快照。时序与 `version_changed` 相符，但这一点是推断。

## 3. 开放问题：查重与合并怎样对待主动写入

现在来源（主动写入或被动提取）不参与查重与合并：被动草稿与主动写入在查重中等价，合并时地位对等、以较新者为准。第 2 节显示，以主动写入为主的使用方式下，这让被动提取从"兜底"变成了"改写主动写入"。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 维持现状，只改进查重的判定信号（查重 Todo） | 不改合并语义。被动草稿与主动写入确实是同一件事，判定信号再准也会判为 SIMILAR，第 2.2、2.3 节的改写照常发生；改进的只是第 2.4 节那类误合并 |
| B | 来源参与合并：主动写入的一方作为主体，被动草稿只补入主动写入没有的事实；同一轮内的主动写入之间不合并 | 合并提名与合并输入要携带双方来源，合并提示词不再只按新旧定主次；被动提取发现的新事实仍能进入记忆；同一轮内两条主动写入是同一件事时会各自保留。改动 Patchouli 生成第 4 节的"地位对等" |
| C | 来源参与提取：生成材料把本轮已写入的记忆渲染为一行（标题或 alias，与 `READ`、`SEARCH` 的一行摘要同形），提示词要求不重复提取已写入的事 | 从源头减少重复草稿，合并语义不变；能否稳定压住重新推导要回放验证。渲染函数同时服务接力摘要与 token 统计，需要说明与第 7 节"不让 Perception 感知写入焦点或待生成状态"的边界：渲染的是对话中发生过的调用，不是待生成状态 |
| D | 话题中有主动写入时，Mode A 不提取这个话题 | 最符合"兜底"的字面含义，实现最简单；agent 没写下的事实（环境、顺带的决定）随之丢失，被动提取对写得不全的 agent 不再兜底 |

B 与 C 可以组合：C 减少重复草稿，B 处理漏网的重复与真正带来新事实的草稿。

## 4. 候选设计（分析，不是 owner 决定，可在计划评审时调整）

以下两项与第 3 节的选择无关，都有明显的默认做法。

### 4.1 合并中止原因进入日志与任务快照

合并以 `MemoryMergeAbortedError` 失败时，失败日志带上 `reason` 与 `memory_id`。现在只能凭时序推断原因（第 2.5 节），而四种原因的处置不同：

- `version_changed` 是可重做的冲突；
- `model_output_invalid` 指向提示词或模型；
- `memory_missing` 与 `not_mergeable` 是正常的放弃。

### 4.2 合并遇到版本冲突时重做一次

`version_changed` 时重新读取 M 与 N，以最新的 M 重做一次合并；再次冲突才失败，两条记忆照常保留。理由：

- 一次结算的多条草稿命中同一条 M，是多条提取之后的正常情况，不是故障；
- 重做沿用现有的合并路径与"先写 M 再删 N"的顺序，不增加新的写入者，也不引入按 M 排队的新结构。

代价是每次冲突多一次模型调用。

## 5. 搁置项

- 生成会重写主动写入的标题：SME 第二轮 6 条全被改回项目前缀，例如「在线增量聚类分区与密度阈值陷阱：……」变成「SME 空间分区召回层：……」。标题由生成重写是当前事实，使用手册 skill 也是这样告诉 agent 的；是否让主动写入的标题优先，待需要时展开。
- 同一个发现写进多条主动写入：属于 agent 写法，已由 DeeperResearch 工作区的 repo-deep-dive skill 改写处理（2026-11-02，不在本仓库）。

## 6. 验证方式

- **离线回放**：两个主话题的 Mode A 生成视图、agent 的写入原文与合并前后的版本记录都已保存。候选方案可以在同一份材料上重放，比较草稿与合并结果。
- **指标（初稿）**：

| 指标 | 本批样本的现状 |
|:---|:---|
| 有主动写入的话题中，Mode A 草稿与主动写入重复的比例 | 11/11 |
| 合并后以被动草稿为主体的比例 | 10/10 |
| 合并失败次数 | 29 次 SIMILAR 中 2 次 |
| Mode A 在这些话题中贡献的新事实 | 本批为 0，需要人工标注确认 |

- **样本扩充**：owner 将用改写后的 repo-deep-dive 再调研几个仓库（2026-11-02）。新样本同时检验 skill 改写后，同项目主动写入之间的相似度是否下降。

## 7. 升级为 Plan 的条件

- owner 对第 3 节作出选择；
- 新的调研样本完成离线回放，给出第 6 节指标在当前实现下的基线；
- 选 B 或 C 时，写清合并输入或生成材料的契约变化，以及受影响的当前文档（[生成](../patchouli/generation.md)第 2.1、4、7、8 节，[MemoryLibrary](../patchouli/memory-library.md)）；
- 版本归属确定。

## 附录：2026-11-02 样本现状与处置建议

处置由 owner 经管理入口执行。删除只移除中期库原子，证据仍可从 Artifact 回放。本附录只记录取证时的状态，处置之后不再更新。

**SME**

| 记忆 | 来源与经历 | 建议 |
|:---|:---|:---|
| `fact_sme_architecture` | 第二轮主动写入，并入 1 条被动草稿 | 保留 |
| `ref_sme_spatial_partition` | 同上 | 保留 |
| `ref_sme_zero_drift_compat` | 同上 | 保留 |
| `fact_sme_ranking_layer` | 同上 | 保留 |
| `fact_sme_memory_generation` | 同上 | 保留 |
| `fact_sme_memory_dynamics` | 追问轮主动写入，并入 1 条被动草稿 | 保留 |
| `fact_sme_persistence_protocol` | 主动写入，先后并入子 Agent 话题的风险清单与 1 条被动草稿 | 保留；"风险清单""未决问题"两节可以删去 |
| `ref_lazy_decay_pattern` | 第一轮主动写入，第二轮修正 | 保留（可迁移经验），与记忆动力学一条部分重叠 |
| `ref_sme_dynamics_risks` | 第一轮子 Agent 话题，合并检索风险后又被追加复核段 | 删除：审计清单，其中设计层面的结论已在架构、空间分区、零漂移、排序层各条中 |
| `fact_sme_retrieval_ranking` | 第二轮子 Agent 话题 | 删除；检索管线（区域召回、全局补充、PPR 图扩展）目前没有讲解式的记忆，下次调研 SME 时补写 |
| `ref_sme_dead_code_list` | 第二轮子 Agent 话题 | 删除 |
| `fact_sme_storage_persistence` | 第二轮子 Agent 话题，合并失败残留 | 删除：内容已在持久化协议一条 |
| `fact_rerank_topn_topk` | 第一轮子 Agent 话题，第二轮并入同类草稿 | 删除：精排不回写分数已在排序层一条 |
| `fact_query_vector_cache` | 第一轮子 Agent 话题 | 删除 |
| `wip_retrieval_open_questions` | 第一轮子 Agent 话题 | 删除：未解问题留在报告，WORK_IN_PROGRESS 类型会误导 |

**Gliding Horse**

| 记忆 | 来源与经历 | 建议 |
|:---|:---|:---|
| `ref_evidence_receipt_audit` | 主动写入（证据收据门），写入时并入架构与收尾门两条主动写入，再并入 1 条被动草稿 | 保留并拆分：收据门留在这里；架构部分与 `fact_gliding_horse_arch` 重叠，收尾门与 `ref_close_gate_pattern` 重叠，可以人工删去这两节，或下次调研时重写 |
| `ref_parallel_agent_file_isolation` | 主动写入，并入 1 条被动草稿 | 保留 |
| `fact_gliding_horse_arch` | 合并失败残留，吸收了 owner 要求单独写的 Agent OS 记忆 | 保留：Gliding Horse 的架构与 Agent OS 定位以它为准 |
| `ref_close_gate_pattern` | 被动草稿（与混合记忆 0.683，判为 DISTINCT） | 保留，作为收尾门的独立记忆 |

**调研流程**

| 记忆 | 现状 | 建议 |
|:---|:---|:---|
| `code_repo_deep_dive_skill` | 记录 2026-11-01 版 skill | 已于 2026-11-02 按改写后的 skill 修正 |
| `code_repo_deep_dive_fallback` | 记录一次 skill 修改 | 删除：规则已在 skill 第 1 步，skill 文件是事实来源 |
| `wip_repo_deep_dive_writeback` | 进行中状态，"待选下一步"已经过时 | 删除 |

另：simple3DRender（2026-11-01）从标题看有两对重复，`code_simple3drender_arch` 与 `fact_simple3drender_architecture`、`ref_hodgman_vertex_bound` 与 `ref_fixed_plane_clip_buffer`，未逐条核对正文。
