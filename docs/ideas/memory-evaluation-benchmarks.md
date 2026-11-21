---
title: 记忆核心：评测集
status: idea
horizon: current
serves_version: v0.8
owner: patchouli
scope: v0.8-memory-evaluation-public-benchmarks-agent-workflow-datasets-and-harness
code_paths:
  - scripts/memory_replay.py
  - scripts/halumem_eval.py
  - scripts/trial_instance.py
  - src/hivememory/system/services/import_bus/models.py
  - src/hivememory/mcp_server/server.py
related_docs:
  - docs/ideas/memory-flow-short-to-mid-term.md
  - docs/ideas/memory-type-system.md
  - docs/ideas/lightweight-footprint.md
last_reviewed: 2026-11-20
---

# 记忆核心：评测集

## 0. 文档性质

本文是 v0.8 记忆核心的子文档（[总纲](./memory-flow-short-to-mid-term.md)），用公开评测集与 agent 工作流数据检验记忆核心的设计：候选评测集、接入 HiveMemory 要解决的问题，以及评测在评判分层中的位置。正文是已定的设计，候选设计标“候选”。第 3 节的外部评测集只依据论文摘要与数据页整理（检索于 2026-11-16），没有读全文，也没有复现其中的任何数字。

## 1. 出发点

- **公开评测集与 agent 工作流数据集属于 v0.8 的内容，要尽早接入**：LoCoMo、LongMemEval 这类对话评测之外，还要找专门建模 agent 工作流的数据集。
- **理由：手上没有带标准答案的数据**。类型体系的回放（[类型体系](./memory-type-system.md)第 9 节）只能比较同一份材料多次运行是否一致：预期标签由 Claude 起草，不一定对；删除信号几乎没有用，owner 只顺手删过一部分，没有筛过全库；“类别能否引导记忆按预期用法被使用”哪一边都测不了。
- **评测结果用于验收与对比，不用来推出设计**，与“样本统计不作为设计依据”不冲突。

## 2. 现状（以代码为准）

- **回放**：[回放脚本](../../scripts/memory_replay.py)从一个 Workspace 的 Artifact 目录只读导出生成材料与建成的记忆，能重放被动提取、按提示词分类、统计多次运行是否一致；它没有标准答案，也不经过话题、查重与检索。
- **导入**：外部会话经 Import Bus 按轮导入。一轮带来源、外部会话与轮次的标识、用户消息、此后的事件与结局，没有时间字段（[Import Bus 模型](../../src/hivememory/system/services/import_bus/models.py)），记忆记下的是导入的时刻。
- **读取**：外部 harness 经 MCP 的 `memory_search` 只传一句查询（[MCP 入口](../../src/hivememory/mcp_server/server.py)）；HiveMemory 只交出记忆，不负责作答。
- **长程**：owner 每天实际使用 HiveMemory，这是现成的长程试验场，但只能验收已建成的机制（[总纲](./memory-flow-short-to-mid-term.md)第 4.2 节）。
- **HaluMem**：[`halumem_eval.py`](../../scripts/halumem_eval.py) 在一个独立的实例（[`trial_instance.py`](../../scripts/trial_instance.py)，自己的数据根与端口、每个用户一个 Workspace）上逐次会话导入 HaluMem 的对话、结算并取出写出的记忆，按官方协议判抽取与更新，汇总成报告；用法见[在 HaluMem 上评测记忆生成](../help/halumem-evaluation.md)。问答与依赖日期的题没有接入。

## 3. 候选评测集

### 3.1 对话记忆

| 评测集 | 测什么 | 规模与发布 | 已知问题 |
|:---|:---|:---|:---|
| [LongMemEval](https://arxiv.org/abs/2410.10813)（ICLR 2025） | 信息抽取、跨会话推理、时间推理、知识更新、拒答五种能力 | 500 题；每题一份独立的历史，S 版约 40 次会话、约 115k token，M 版约 500 次会话；数据与代码公开 | 论文中长上下文模型在 S 版上下降 30%–60%，“全部历史直接交给模型”是现成的对照 |
| [LoCoMo](https://arxiv.org/abs/2402.17753) | 长对话问答：单跳、多跳、时间、开放域与对抗 | 10 段对话，每段约 300 轮、最多 35 次会话；公开 | 一份独立[审计](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer)称 1,540 道非对抗题中 6.4% 的答案有误，LLM 判分偏宽；各家记忆厂商对彼此的分数有[争议](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) |
| [HaluMem](https://arxiv.org/abs/2511.03506) | 逐个环节打分：抽取（完整度、准确度）、更新（幻觉率）、问答 | Medium 与 Long 两版，各 20 个用户、14,948 个标注的记忆要点、3,467 题，Long 版每个用户约 1M token；[Hugging Face](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem) 发布，英文，许可 CC-BY-NC-ND | 记忆要点的粒度与 HiveMemory 的记忆不同，对照要靠模型判断 |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257)（ICLR 2026） | 准确检索、部署中学习、长程理解、冲突消解（FactConsolidation） | Hugging Face 发布 | — |

### 3.2 agent 工作流

| 评测集 | 测什么 | 发布 |
|:---|:---|:---|
| [MemoryArena](https://arxiv.org/abs/2602.16313)（ICML 2026） | 多次会话、子任务互相依赖：组合网购、渐进搜索、团体旅行规划、形式推理；按任务完成度打分，不按答题；在 LoCoMo 上接近满分的系统在这里成功率很低 | 数据在 Hugging Face（CC-BY-4.0），代码在 GitHub（[项目页](https://memoryarena.github.io)） |
| [MEMTRACK](https://arxiv.org/abs/2510.01353)（Patronus，NeurIPS 2025 workshop） | 模拟软件团队：Slack、Linear、Git 之间按时间线交错发生的事件，含噪声、冲突与相互引用；按正确率、效率与冗余打分，GPT-5 的正确率约 60% | 47 个实例（据二手摘要）；数据经 Google Drive 发布，需要搭它的环境 |
| [DreamBench-SWE](https://arxiv.org/abs/2608.20664)（2026-08） | 多次编码会话，后面的任务依赖前面会话中推不出来的细节，并检查是否用了过时的记忆；由可执行的隐藏测试判分 | 代码在 GitHub 发布；单作者 |
| [VibeMemBench](https://arxiv.org/abs/2609.23570)（2026-09） | 记忆系统能否帮编码 agent 在真实仓库上完成任务；4 个记忆系统与 3 个求解器的 12 种组合中，11 种不如不用记忆 | 没有找到发布的数据 |
| [SWE-ContextBench](https://arxiv.org/abs/2602.08316)（2026-02） | 相关的 SWE-Bench 任务组成序列，看能否复用此前轨迹中的经验 | 论文 |
| [MERIT](https://arxiv.org/abs/2609.05441)（2026-09） | 带工具的 agent 按世界的最终状态判成败，并计量每次记忆操作的成本；检索到正确值之后，agent 只有约 55% 的情况据此行动 | 论文称发布了评测框架与全部轨迹 |

**没有标注的轨迹数据**：例如 [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)（约 20.7 万条 OpenHands 与 SWE-agent 轨迹）。它们没有题目，可以作为回放脚本的输入，检验被动生成在 owner 自己的会话之外产出什么；但都是自动运行的单个任务，没有真人参与，也不跨会话。

## 4. 接入 HiveMemory 要解决的（候选）

1. **原始时间**：评测集的会话都带日期，Import Bus 的一轮没有时间字段（第 2 节）。按会话顺序导入时，“哪个是更新的说法”仍由到达顺序保留；“那件事发生在哪天”会丢，时间推理类的题因此失真。日常使用时导入紧跟在对话之后，差别很小，只有批量导入历史时才是问题。
2. **评测壳**：HiveMemory 只交出记忆，评测需要“检索 → 由一个模型作答 → 判分”这一层；检索要用到 Qdrant 与向量，要跑在一个实例上，不能像回放那样完全离线。判分若用 LLM，要先校准它的宽严（LoCoMo 判分偏宽的问题）。
3. **规模与成本**：被动生成要经过话题与结算。LongMemEval 每题一份历史，全量要导入 500 份、每份约 115k token，宜先按题型分层抽样；LoCoMo 只有 10 段对话；HaluMem 是 20 个用户、每人一份长历史。
4. **隔离**：每份历史一个独立的 Workspace 或数据目录，不碰日常实例的数据根（与回放相同）。
5. **许可**：HaluMem 是 CC-BY-NC-ND，只能做内部评测，不能改写后再分发；其余发布前逐个核对。

## 5. 评测在 v0.8 中的位置（候选）

按[总纲](./memory-flow-short-to-mid-term.md)第 4.2 节的评判分层：

| 层 | 评测集 | 检验什么 |
|:---|:---|:---|
| 生成质量，逐个环节 | HaluMem | 抽取得全不全、对不对，更新时有没有编造；对应生成核心与被动结算，也补上类型体系回放测不了的“写得对不对” |
| 单轮底线：不交记忆税 | LongMemEval S 版，分层抽样 | 与“全部历史直接交给模型”对照；知识更新与拒答对应修订与“不知道就说不知道”；LoCoMo 只用来和已发表的系统横向比较 |
| 记忆是否改变行动 | MemoryArena 或 DreamBench-SWE | 按任务成败判分，对应统一对话与长程 agent 工作的方向；DreamBench-SWE 的场景最接近 owner 日常的编码会话 |
| 长程 | owner 的日常使用 | 评测集代替不了 |

- 候选顺序：HaluMem → LongMemEval S 版（分层抽样）→ MemoryArena 或 DreamBench-SWE → LoCoMo。
- [轻量化 Idea](./lightweight-footprint.md)需要的检索评测集（更轻的嵌入与重排方案损失多少质量）也可以从这里取。

## 6. 首个评测切片

**设计**：

- 首个评测切片是 HaluMem 的 Medium 版。v0.8 改的是写入一侧，HaluMem 是唯一逐个环节打分抽取、准确与更新的评测集，测的正是写得全不全、写出的有没有编造、更新时有没有改对；问答类的评测集还掺进了检索与作答两层。
- 首轮只测抽取与更新，不测问答与依赖日期的题；原始时间暂不带入，Import Bus 的一轮不加时间字段，日期也不写进用户消息。时间推理类的题留给之后：届时在“给 Import Bus 的一轮加上原始时间”（公开契约的变更，日常的批量补录也会受益）与“日期写进用户消息”（不改契约，但改变生成材料）之间选择。

**现状：第一次评测**（仓库 `c0a9fcc5`；Medium 按对话量分层取 5 个用户，346 次会话、7,499 轮；生成 deepseek-flash、英文提示词；评判 GPT-6 Luna，温度 0、推理关闭；每种结算方式只跑了一次）：

| 指标 | 一次性生成 | 馆长 | 馆长更高的用户 |
|:---|---:|---:|---:|
| 完整度（召回 / 按重要度加权） | 63.7% / 81.0% | 66.1% / 76.3% | 3/5、0/5 |
| 准确度（落在要点内的记忆 / 全部记忆） | 92.3% / 66.3% | 87.9% / 75.1% | 0/5、5/5 |
| 不记干扰 | 29.0% | 38.4% | 5/5 |
| 更新（正确 / 幻觉 / 遗漏） | 44.6% / 5.6% / 49.6% | 48.1% / 3.9% / 47.4% | 3/5（正确） |
| 每次会话写出的记忆 / 正文平均字数 | 3.51 / 1,782 | 2.73 / 718 | — |
| 失败的生成任务 | 82 / 1,385，全部是合并中止 | 12 / 594，提供商失败与超时 | — |

- 评判的 10% 复判一致率为 92–94%，几个百分点以内的差距只看逐用户方向一致的指标：馆长写出的记忆整体更准、更少记干扰、每次写得更少；落在要点内的那部分准确度反而略低；更新与完整度两者相当。
- HaluMem 论文（评判 GPT-4o、全部 20 个用户）中各系统在同一组指标上的数字只作参考：两种结算方式的完整度与 F1 仅次于 MemOS，全部记忆的准确度高于所有系统；不记干扰低于所有系统；更新的幻觉率高于所有系统（它们都不超过 1.2%）。HiveMemory 的记忆是长文，条数与粒度都与它们不同。
- 被判为更新幻觉的样本中，生成模型编出事实的只占少数，主要是两类：被推翻的旧事实没有撤下或标为过时（一次性生成的合并只融合相似的记忆，馆长不修订旧记忆），以及新值没有写进库、检索到的旧值被判为幻觉。数据本身也有噪声：数值类的更新要点中约三分之一的新值在该次会话的对话里没有出现。
- 运行目录、逐项数字、预注册与结论的核对、以及上面的归因在开发 checkout 的 `.hivememory/lab/`（运行目录 `runs/eval1-*` 与账本 `ledger.md`），不进仓库。

## 7. 升级为计划的条件

- 选定的评测集已核对：确认数据格式、发布状态与许可；需要横向比较时，作为一份调研需求放进 [Research](../research/README.md)。
- 评测壳的形态已确定。
