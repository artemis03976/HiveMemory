---
title: 记忆核心：短期到中期的流转
status: idea
horizon: current
serves_version: v0.8
owner: patchouli
scope: redefine-the-short-to-mid-term-memory-flow-the-two-write-paths-and-the-shape-of-the-mid-term-library
code_paths:
  - src/hivememory/engines/perception/material.py
  - src/hivememory/engines/generation/
  - src/hivememory/engines/memory_compiler/
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/services/retrieval.py
  - src/hivememory/core/models/memory.py
  - src/hivememory/i18n/prompts.py
  - src/hivememory/engines/lifecycle/garbage_collector.py
  - src/hivememory/engines/lifecycle/reinforcement.py
  - src/hivememory/engines/retrieval/fusion.py
  - src/hivememory/workspace/capability/actor/memory.py
  - src/hivememory/mcp_server/server.py
  - integrations/hivememory/
  - src/hivememory/memory_library/
  - src/hivememory/infrastructure/storage/vector_store.py
  - src/hivememory/agent_runtime/
  - src/hivememory/engines/artifacts/
  - src/hivememory/patchouli/services/memory_generation.py
related_docs:
  - docs/ideas/memory-flow-write-path.md
  - docs/ideas/memory-library-shape-and-index.md
  - docs/ideas/memory-type-system.md
  - docs/ideas/memory-evaluation-benchmarks.md
  - docs/research/memory-core-external-references.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/generation.md
  - docs/patchouli/perception.md
  - docs/ideas/unified-interaction-memory.md
  - docs/ideas/short-term-memory-and-context-compaction.md
  - docs/ideas/memory-as-tool-skill-plugin-boundary.md
  - docs/governance/reliability/durability-and-recovery.md
  - docs/patchouli/artifacts.md
  - docs/todo/generation-dedup-decision-signal.md
  - docs/todo/passive-extraction-volume-and-retention.md
  - docs/todo/cross-conversation-topic-placement.md
last_reviewed: 2026-11-18
---

# 记忆核心：短期到中期的流转

## 0. 文档性质与阅读路径

本文是 v0.8 设计讨论的总纲，为该版本的计划供料，不描述当前行为，也不代表排期。三层存储的当前设计见 [MemoryLibrary 与存储层](../patchouli/memory-library.md)，生成、查重与合并见 [Patchouli 生成](../patchouli/generation.md)，话题与结算见[感知与短期话题](../patchouli/perception.md)。

[统一的交互记忆](./unified-interaction-memory.md)是 owner 认为可能的出发点，但 owner 还没有逐项过，认为它仍有缺陷；其中的轨迹、断言、整理与消费水位设想不在这里复述，它第 5 节的两项与本文第 4.1 节、[写入链路](./memory-flow-write-path.md)第 5 节一并考虑。

2026-11-16 按主题拆成一份总纲、三份子文档与一份外部参考，内容未删减：

| 文档 | 内容 |
|:---|:---|
| 本文（总纲） | owner 的出发点与总体说明（第 1.1、1.2 节）；全部决定的一览（第 1.3 节）；与讨论有关的现状，以代码为准（第 2 节）；外部调研对设计的启发（第 3 节）；v0.8 的顺序、辅线与评判方式（第 4 节）；搁置项、与其他文档的关系和升级为计划的条件（第 5 至 7 节） |
| [写入链路](./memory-flow-write-path.md) | 对照已有记忆复核、主动写入的背景与短期记忆、时间线、什么值得写；Patchouli 的角色；整理写入与生命周期维护的边界；两条写入路径在哪里汇合 |
| [库的形状、图索引与保存形态](./memory-library-shape-and-index.md) | 平铺的库、时空图索引、关系层；保存形态与"索引不是真相"；中期库的形状与保存形态两个问题 |
| [类型体系](./memory-type-system.md) | 项目最初设计的回顾与类型体系的全部讨论 |
| [评测集](./memory-evaluation-benchmarks.md) | 用公开评测集与 agent 工作流数据检验记忆核心：候选评测集、接入要解决的问题与在评判分层中的位置（2026-11-16 新增） |
| [外部参考](../research/memory-core-external-references.md) | 八个项目、VCPToolBox 的 Rust 运行时与 OneRing、Anthropic、Perplexity、Obelisk 与 Nowledge Mem 的外部事实 |

各文档中，除注明为 owner 决定的内容外，都是讨论中的分析或候选设计；决定只在本文第 1.3 节列一览，细节在各子文档。

## 1. 出发点、说明与决定（owner）

### 1.1 进入 v0.8（2026-11-13）

- v0.8 重新设计记忆的核心机制：重新定义并设计短期到中期的流转，而不是为现有的症状打补丁。
- 被动生成的质量问题是现象，不是根因（2026-11-16 修正最初"根因在上游，是被动生成的质量"的表述）。现象是：一次结算可以产出多条记忆（[多记忆提取](../archive/ideas/settlement-multi-memory-extraction-and-dedup.md)，v0.7.2）之后，模型倾向于保留每个细节，较长的对话产出三四条甚至更多；碎片多了，系统就依赖查重大量合并，数量一大，查重的失衡就暴露出来。现在过滤掉被动生成的重复记忆，已经暂时缓解了碎片；但只修好被动生成并没有解决碎片化，一旦要解决"流水账""查重"等问题，就已经不是修复被动生成质量这么简单。根因是讨论中提出的那些问题：记忆系统在最初的原型完成后，主体设计就没什么变化，又等到项目基建完善才被验证，所以现在才暴露出来；即使没有这些分析，被动生成的质量也不能是根因，v0.8 不是来打补丁的。查重的判定信号（[查重 Todo](../todo/generation-dedup-decision-signal.md)）重要，但只是下游策略。
- v0.8 的顺序（2026-11-16）：先完成生成核心的升级，包括 Patchouli 馆长 agent 与类型的更迭，再处理被动生成，最后是主动写入。被动结算具体产出什么，实际上完全取决于提示词怎样写（第 4.1 节）。
- 原型证明了能保留重要证据并反哺 agent，但质量与自主程度离目标还很远。目标是全自主地维护记忆库；维护固然重要，但写入从一开始就决定了维护的难度（2026-11-14）。
- 三层的含义不变：短期是话题工作台，中期是 Qdrant 中可检索的记忆，长期是归档记忆的冷藏库。长期层在项目开始时就已定下，计划改用 sqlite 一类数据库存储，有自己的维护规则，不在这次重新设计的范围内。
- owner 对讨论中第一份提议的转述：记忆素材的来源与最终的生成队列之间，缺少一个归一化的中间表示（[写入链路](./memory-flow-write-path.md)第 2.1 节）。

### 1.2 讨论中的说明与纠正（2026-11-13～16，按主题）

**方法与诊断的补充**

- 查重的相似度只看索引层，正文不参与，正文的长短影响不了相似度（第 2.2 节）。
- 被删除的被动记忆，大多出于 owner 的偏好：记忆不该像流水账，不为 PR 单独写工作记录。这与两种交互形式有关：chatbot 式的交互天然像流水账，倾向于记住"用户说的"；agent 工作式的交互要记的是工作要点。真实情况比这复杂得多，不能只凭这几点概括设计。
- 样本分析对重新设计基本没有作用，设计要从底层重新开始。样本以 owner 自己的 HiveMemory 开发会话为主，从中得到的特征（例如 agent 工作式的会话里用户消息多是短指令）只对 owner 现在的使用成立，不能推广。

**记忆观**

- 两条写入路径都保留。被动提取与 agent 的主动写入是大多数 agent 记忆项目会采用的两种做法（[外部参考](../research/memory-core-external-references.md)）；把 chatbot 式与 agent 工作式同时纳入考虑时，两侧的信息同样重要。从人的角度看，本来就有"潜意识记忆"与"主动记忆"两种行为，因此这个方向是合理的，也是必须解决的。
- 调研过的项目都没有覆盖一点：通过强化学习让 agent 学会使用记忆工具。如果 agent 从根本行为上就愿意一直使用记忆工具，而不是依赖外部的 skill，留给被动提取的信息就不多了；但对闭源模型，这条路不可行（第 5 节）。
- 记忆即知识（2026-11-14）。项目一贯的理念是：持久化、可寻址、可追溯、可演化的记忆资产，不只是一段可读文本的"回忆"，而是 agent 所有知识的外部积累。流行的 skill 文档、犯错经验的反思、知识库，对无状态、无感知的 LLM 来说都是记忆，正如人从出生起的所有经历既是记忆，也是先验知识。关键在于把这些材料统一内化，成为真正属于 agent 的知识，这是记忆生成链路要做到的事（[写入链路](./memory-flow-write-path.md)第 2.4 节）。
- 价值由写入者判定（2026-11-15）。记忆价值体系原本通过分析用户输入来标记对话价值，后来被 owner 停用：不只是误判率太高，更因为只有终端的 Patchouli 才有资格决定"看到了什么但不写成记忆"。如果 Patchouli 完全明白什么该写、什么不用写，记忆价值在这一刻就已经判定，不再需要外部评判来决定材料的去向。关键在于她需要知道发生过什么、写过什么、更新过什么；作为记忆库唯一的管理者，她有权知道全部历史记忆记录，尽管现在所有写入路径都已经经过她（本文第 2.1 节、[写入链路](./memory-flow-write-path.md)第 3.1 节）。
- 写入与维护分属（2026-11-15）。Anthropic 文章中的 dreaming 更偏向记忆的生命周期维护，原则上与记忆的整理和写入分属两件事；只是在强调查重与复核的当下，两者确实有重叠。未来的"记忆的分裂与合并""记忆冲突检测"等，都属于落库之后怎样保证记忆的演化质量（[写入链路](./memory-flow-write-path.md)第 4 节；维护一侧的内容记在[生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md)方向 6）。
- 知识不等于工具（2026-11-14）。如果记忆即知识，那么插件就是工具，是 agent 与外界交互必不可少的一环：正如人空有理论知识也不能行动，知道自行车哪里坏了、怎么修，也得有一把扳手来修。所以 [MaaT Idea](./memory-as-tool-skill-plugin-boundary.md) 的对象需要重新考虑，核心命题是知识与工具不能划等号；"记忆作为工具"要如何体现、目标是什么，或许要从头考虑（第 2.4 节、[写入链路](./memory-flow-write-path.md)第 3.3 节）。
- 记忆的价值在长时间段内（2026-11-15）。agent memory 的重点一直在可演化与动态维护，即一部分人所说的 runtime adaptation。如果只是一轮召回，那就是 RAG，与老技术没有区别；agent memory 应当是一个极长时间段内的行为设计：让一个系统连续跑一周、甚至一个月，它积累的内容能否有效地用在未来的运行中，以及需要多少人类的介入，才是这个领域真正的价值（第 4.2 节）。

其余几组说明按主题放在子文档中：库的形状、访问与保存见[库的形状](./memory-library-shape-and-index.md)第 1 节，写入链路与 Patchouli 的角色见[写入链路](./memory-flow-write-path.md)第 1 节，类型体系见[类型体系](./memory-type-system.md)第 1 节。

### 1.3 已作出的决定

| 日期 | 决定 | 详见 |
|:---|:---|:---|
| 2026-11-13 | v0.8 重新定义并设计短期到中期的流转，不为现有症状打补丁；被动生成的质量是现象而不是根因（2026-11-16 修正），查重只是下游策略 | 第 1.1 节 |
| 2026-11-16 | v0.8 的顺序：先完成生成核心的升级（Patchouli 馆长 agent 与类型的更迭），再处理被动生成，最后是主动写入；被动结算产出什么主要取决于提示词怎样写 | 第 1.1、4.1 节 |
| 2026-11-13 | 三层含义不变；长期层（归档记忆，计划改用 sqlite 一类数据库）不在这次范围内 | 第 1.1 节 |
| 2026-11-13 | 话题跨会话按内容归并是有意的设计，不是开放问题；归并由 LLM 判定、不能保证合理，由 Todo 跟踪 | 第 1.2、5 节 |
| 2026-11-14 | 样本统计不作为设计依据 | 第 1.2 节 |
| 2026-11-14 | 两条写入路径（被动提取与主动写入）都保留 | 第 1.2 节 |
| 2026-11-14 | 强化学习让 agent 自发使用记忆工具不在 v0.8 范围；"记忆作为工具"的对象与目标要从头考虑，不阻塞 v0.8 | [写入链路](./memory-flow-write-path.md)第 3.3 节、本文第 5 节 |
| 2026-11-14 | 动态检索加载记忆是有意的选择，保留从索引到正文的渐进式披露 | 本文第 1.2 节、[写入链路](./memory-flow-write-path.md)第 2.4 节 |
| 2026-11-15 | 整理写入与生命周期维护分属；记忆的分裂与合并、冲突检测属于落库之后的演化质量 | [写入链路](./memory-flow-write-path.md)第 4 节 |
| 2026-11-15 | 中期库的存储保持平铺，分层由派生、可以重建的时空图索引承担，读写两侧都可以使用 | [库的形状](./memory-library-shape-and-index.md)第 4.1 节 |
| 2026-11-15 | 两条写入路径在中期记忆上汇合：主动写入写下即生效、修订前有效；被动结算是话题的终态收尾者；生成事件带上轮次与事件的信息 | [写入链路](./memory-flow-write-path.md)第 5 节 |
| 2026-11-15 | 时间线保留为顺序与身份的骨架，"每个事件只消费一次"的水位语义退场 | [写入链路](./memory-flow-write-path.md)第 2.3 节 |
| 2026-11-15 | 记忆的价值在长时间段内衡量：连续运行一周、一个月之后，积累的内容能否有效用于未来的运行，以及需要多少人工介入；单轮召回只是 RAG | 第 1.2、4.2 节 |
| 2026-11-15 | 去掉 WORK_IN_PROGRESS 类型，中期库也不再收"当前进度"；"做到哪了"留在交互记录上，即时间轴上，以后由短期记忆或经历节点上的状态承担 | [类型体系](./memory-type-system.md)第 6.1 节 |
| 2026-11-15 | 可运行是一条独立的轴，不绑在类型上，只适用于有可执行主体的记忆，不延伸到 AGENT_PROFILE；分环境声明（知识）与宿主执行（MaaT）两层；可执行物是从记忆编译出的派生物，可重建、随版本失效，生成链不产出；v0.8 不新增机制，CODE_SNIPPET 并入资源记忆后 v0.8 关闭 RUN 的用户工具路径（2026-11-16 改写，原为"RUN 的类型检查暂时保留"） | [类型体系](./memory-type-system.md)第 6.2 节 |
| 2026-11-15 | AGENT_PROFILE 移出中期记忆库，单独成一个 store（AgentProfileStore）：与 ArtifactStore 一样属于 MemoryLibrary，但不在三层记忆存储之中；可用 agent 的列表固定载入提示词，不再靠检索发现 | [类型体系](./memory-type-system.md)第 6.3 节 |
| 2026-11-16 | 一条记忆值得赋予类型，是因为下一位使用者对它有可以预期的用法：类型首先告诉读者能拿它做什么，其次告诉馆长怎样才算写完整；修订时只提示该找哪种依据，改不改由那一段的内容、出处与新证据决定（取代同日较早的"以维护契约为主"） | [类型体系](./memory-type-system.md)第 4 节 |
| 2026-11-16 | CODE_SNIPPET 与 URL_RESOURCE 同质，都是对外部资源的记忆；资源记忆的正文说明是什么、为什么、在哪里，真相留在源头；资源记忆不叫导览，导览是所有记忆都可以具备的性质 | [类型体系](./memory-type-system.md)第 7.1 节 |
| 2026-11-16 | FACT 不作为类型保留，按真相所在拆开：真相在可以查看的源头的并入资源记忆；由有权的人决定的成为"决定与约定"；从推理与多方证据得出的留作"知识"；从做过的事中得出的是经验（REFLECTION） | [类型体系](./memory-type-system.md)第 7.2 节 |
| 2026-11-16 | 用户一侧按描述的对象分为三类：USER_PROFILE（用户本人）、Preference（用户希望事情怎样做，不再冠以 USER）、Decision（对某件事的决定）；"说过一次"与"逐渐了解"作为每条记忆的来源标记，不另立类型。"决定与约定"由此细分为 Preference 与 Decision | [类型体系](./memory-type-system.md)第 7.3 节 |
| 2026-11-16 | REFLECTION 改为"经历"（情景记忆）：回答"你记不记得那一次"，重在发生了什么，教训可选；判据是下一位 agent 需要回到这一次具体经历才能判断是否有用；写入保住情境、经过（含失败的尝试）、结果与出处；一条经历对应一次发生，按出处识别；读别人项目得到的模式、例行记录、面向未来的事项与用户稳定的特征不算经历 | [类型体系](./memory-type-system.md)第 7.4 节 |
| 2026-11-16 | USER_PROFILE 与 Preference 的全局部分怎样送达外部 harness 是插件的问题，归入外部 actor 接入 Idea 的 E-7；先走主动读取，仍在选项 A 之内 | [类型体系](./memory-type-system.md)第 7.3 节；[外部 actor 接入 Idea](./external-actor-registration-and-runtime-access.md) E-7 |
| 2026-11-16 | 只存在于记忆中的代码不另设类型，按角色并入经历或知识；代码部分原样保住，只能整段替换；能运行时附环境声明 | [类型体系](./memory-type-system.md)第 7.5 节 |
| 2026-11-16 | 新的类型体系先在已有的对话数据上回放，检验 LLM 能否稳定把握它的边界，再写进计划 | 本文第 4.2 节、[类型体系](./memory-type-system.md)第 9 节、本文第 7 节 |
| 2026-11-16 | 首份计划落在类型体系；先在已有的对话数据上完成回放检验（补一小组 chatbot 式样本），项目持久的回放需求之后由一套回放工具承担 | [类型体系](./memory-type-system.md)第 1、9.1 节 |
| 2026-11-16 | 公开评测集（LoCoMo、LongMemEval 等）与建模 agent 工作流的数据集属于 v0.8 的内容，需要尽早提上日程，单独成一份 Idea | [评测集](./memory-evaluation-benchmarks.md)第 1 节 |
| 2026-11-18 | 馆长的第一步之前，先让生成出来的记忆指向具体的交互轮次：被动结算下的复核与复核要用的判定格式都依赖它 | [写入链路](./memory-flow-write-path.md)第 5 节 |
| 2026-11-18 | 馆长的第一步（固定流程的复核）先做：它本来就是新的主被动复核需要实现的，被动结算要结合主动写入动过的记忆来生成，不等于 agent 化；agent 化排在它之后 | [写入链路](./memory-flow-write-path.md)第 3.1 节 |
| 2026-11-18 | Agent Profile 移出中期库后不再作为记忆原子看待，也不再遵循记忆原子的结构；手动编辑的历史不保留；直接修改本地文件与经 Agent 页修改等同，需要热重载 | [类型体系](./memory-type-system.md)第 6.3 节 |
| 2026-11-17 | 类型收敛建立计划前的两项：库中已有的 WIP、CODE_SNIPPET 与 URL_RESOURCE 逐条列出，由 owner 决定改类型还是删除；AgentProfileStore 用文件存储，一个 Agent Profile 一个文件 | [类型体系](./memory-type-system.md)第 6.3 节、第 7 节开头 |
| 2026-11-17 | 记忆类型的收敛只作为 v0.8.0 的第一个方向实现。v0.8 参考 v0.7 的版本规划重新考虑：v0.8.0 的版本目标为生成核心的升级（按 2026-11-16 的顺序取第 1 步，Claude 整理，待 owner 确认）；一个版本围绕一个功能，由几份能各自验收的计划完成，计划只在能立即执行时建立 | [类型体系](./memory-type-system.md)第 7 节开头；[ROADMAP](../ROADMAP.md) 第 4.2 节 |
| 2026-11-16 | 回放与对照组之后，新旧类别在稳定性上没有区别、类别依旧不平衡，能否引导按预期用法使用两边都测不了。首份计划只实施已移出类型的三处（WIP、AGENT_PROFILE、可运行）与 CODE_SNIPPET、URL_RESOURCE 合并为统一的资源记忆类型；FACT、用户一侧、REFLECTION 与只存在于记忆中的代码的重新划分不进入首份计划，FACT、REFLECTION、USER_PROFILE 维持现状 | [类型体系](./memory-type-system.md)第 1 节、第 7 节开头、第 9.2 节 |

## 2. 现状（以代码为准）

### 2.1 写入链路：两条路径、生成材料、判断与修订

**两条写入路径互相看不到**

- 被动提取的输入只有话题材料。交互材料有意不渲染 WRITE、UPDATE 与 CALL（[交互材料](../../src/hivememory/engines/perception/material.py)），提取时也不带中期库中的任何记忆。
- 两条路径只在查重时碰面：控制面记下本话题的主动写入已经结算到的正式记忆（`GenerationRequest.topic_write_memory_ids`）；被动提取的草稿查重命中其中一条时丢弃，主动写入的草稿相似命中时照常新建、不提名合并（[生成引擎](../../src/hivememory/engines/generation/engine.py)）。也就是说，两边的关系在各自生成之后才按相似度判断。
- 生成提示词按模式分开（被动、写入、更新、合并）。被动提取用同一份提示词处理所有来源（Alice 的对话与外部 harness 导入的会话），其中已经有针对 agent 工作过程的规则：思考不作为事实依据，工具调用与结果可以作为事实依据，子 Agent 交回的消息按结论对待（[生成提示词](../../src/hivememory/i18n/prompts.py)）。

**主动写入生成时的材料**

- 主动写入生成时的材料是话题的状态摘要与最近若干个已完成的轮次；话题不存在时按空材料生成（[生成协调](../../src/hivememory/patchouli/control/memory_generation/coordinator.py)）。
- 外部 harness 的交互在轮次结束后才经 Import Bus 导入，所以写入的那一刻，当前这一轮还不在话题中；"一句话直出"时话题是新的、空的，生成侧只有 agent 在 WRITE 中写下的内容、标题与理由。
- 写入前的 hook 会读取本地的会话记录，但只为工具参数补上当前一轮的用户消息与之前几轮的文本，用于确定话题，不带本轮的工具事件（[Claude Code hooks](../../integrations/hivememory/connector/claude_code/hooks.py)）。

**判断的位置、修订的形态与证据引用**

- **外部的价值标记已经移除**：代码中已没有 `worth_saving`。
- **信息在系统里，但不在做判断的地方**：控制面知道本话题的主动写入落到了哪些记忆（第 2.1 节），但这份信息只用于查重；真正做判断的那次 LLM 调用既看不到这些写入，也看不到库。
- **修订是整篇重写**：Mode C 的输出是整篇新正文加一句变更说明（`RevisionResult`），合并也是整体重写。
- **修订不改索引层**：Mode C 只覆盖正文，标题、摘要、标签与类型保持不变，检索与查重用的向量也随之不变；一条记忆被修订后，检索结果显示的仍是修订前的标题与摘要（[生成引擎](../../src/hivememory/engines/generation/engine.py)）。管理页的手动编辑可以改标题、摘要、标签与正文，不能改类型（[记忆生成服务](../../src/hivememory/patchouli/services/memory_generation.py)）。能改类型的只有合并：类型由模型重选。
- **证据引用只到整个 Artifact**：记忆中的 `ArtifactRef` 指向整个 Artifact（话题快照），没有轮次或事件的位置。原始记录有坐标：`TurnRecord` 有 `turn_id`，每个 `TurnEvent` 有 `sequence`，工具调用另有 `action_id`，交互快照逐轮保留它们。坐标丢在中间：生成材料的 `GenerationTurn` 没有 `turn_id`，渲染为序数 `[Turn 1]`、`[Turn 2]`，事件没有编号；模型输出的草稿（`ExtractedMemoryDraft`）没有来源字段。被动结算的引用指向整个话题（含已折叠的轮次），主动写入的指向最近几轮；合并带上被吸收那条的引用，手动创建没有引用。生产代码只写入这些引用、在合并时带上，没有读取方（[记忆生成服务](../../src/hivememory/patchouli/services/memory_generation.py)）。

### 2.2 中期库：平铺、查重与生成形式

**查重只看索引层**

草稿按 `Title / Type / Tags / Summary` 四行编译成稠密向量的输入（[embedding handler](../../src/hivememory/engines/memory_compiler/handlers/embedding.py)），在整个 Workspace 中取最近的一条，按阈值判为相同、相似（先新建，再由合并任务融合）或不同（[Patchouli 生成](../patchouli/generation.md#4-查重与合并)第 4 节）。正文不参与。

**中期库是平铺的**

- `MemoryAtom` 只有 Workspace 归属、来源、读取策略与生命周期，索引层（标题、摘要、标签、类型、alias），正文与 Artifact 引用。关系层的 `relates_to`、`supersedes`、`depends_on` 是预留字段，生产代码不写入（[记忆模型](../../src/hivememory/core/models/memory.py)）。
- 检索的业务过滤只有类型、来源 agent、时间范围、标签与最低置信度（`QueryFilters`），没有"属于哪一组"。
- 话题是系统中唯一按内容归组的结构，但只存在于短期：结算时材料一经生成队列接纳，话题就被删除（`PerceptionFamiliar._settle_topic`）。之后，一条记忆与它的话题只剩间接的联系：它引用的来源 InteractionArtifact 记着 `topic_id`。

**生成的形式为平铺检索而定**

生成提示词（[生成提示词](../../src/hivememory/i18n/prompts.py)）把这一点写得很明确：

- 被动提取要求"将内容拆解为独立、自包含的知识点"，每件事写成一条记忆，"标题与摘要要让人能单独检索到它"；
- 主动写入沿用 agent 拟定的标题，但标题缺少"让人检索到它的关键名词（项目、组件、工具名）"时要修改；
- 合并要求标题与摘要覆盖合并后的全部要点，因为"检索只使用标题、类型、标签与摘要，不读取正文"。

**关系层**

- `RelationLayer` 有 `relates_to`、`supersedes`、`depends_on` 三个记忆 ID 列表，标注为预留（[记忆模型](../../src/hivememory/core/models/memory.py)）。
- 后端与前端都没有任何读取方或写入方。
- 它被序列化进每一个 Qdrant point；记忆版本快照也把 `relations` 列为必备字段。
- `patch_payload` 的白名单只有生命周期字段与读取策略，不含关系层；改一次关系就要整条记忆重新提交，并生成新的版本记录。
- 合并已经以另一种形式记录了"取代"：被吸收的记忆被删除，旧 alias 改指到保留的记忆，版本记录以 MERGE 来源记下这次合并。

### 2.3 存储：真相、一致性与记忆原子中的非文本部分

**存储层的真相与跨存储的一致性**

- 中期层的真相就是 Qdrant：向量、索引层与正文在同一个 point 上，它是 Retrieval、Generation 与 Lifecycle 共同依赖的当前记忆事实库（[MemoryLibrary 与存储层](../patchouli/memory-library.md#12-中期当前可检索书库)第 1.2 节）。
- 跨存储的写入都不在一个事务里（[耐久性与故障恢复](../governance/reliability/durability-and-recovery.md#1-当前状态与问题证据)第 1 节）：MemoryAtom 与 Artifact 的写入失败时可能留下未被引用的 Artifact；archive 与 revive 在中期库与冷藏库之间搬运，失败时可能留下重复副本或中间态。治理文档写的处理方向是对账与可重试的 saga。
- 读取视图的失效经进程内的变更事件，尽力而为、不重放；读取视图本身不跨重启，所以这样成立。
- 启动时，记忆 payload 只接受 schema 2.1，其他版本拒绝解码；已有的 Qdrant 集合则不核对向量配置，存在即直接使用（[向量存储](../../src/hivememory/infrastructure/storage/vector_store.py)）。

**记忆原子中哪些部分与文本等价**

- **与文本等价**：索引层（标题、摘要、标签、类型、alias）相当于 frontmatter；`payload.content` 本身就是 Markdown 正文；Artifact 引用可以用链接表达。
- **不是文本，并且有生产使用方**：
  - 使用状态：检索命中时只刷新最近访问时间；actor 经 READ 或 RUN 读取时记录一次引用，访问次数、生命力与衰减基准随之变化（[actor 记忆操作](../../src/hivememory/workspace/capability/actor/memory.py)、[强化](../../src/hivememory/engines/lifecycle/reinforcement.py)）。生命力参与检索排序（[融合排序](../../src/hivememory/engines/retrieval/fusion.py)）与归档。
  - 置信度：生成时给出，注入时显示，也参与检索排序。
  - 读取策略：读取时按 agent 执行（[MemoryLibrary 与存储层](../patchouli/memory-library.md#12-中期当前可检索书库)第 1.2 节）。
  - 版本：每次内容提交都必须带版本记录，变更事件让 workspace 的读取视图失效（同上）。
  - 可运行的定义：AGENT_PROFILE 的 `agent_config` 可以被 CALL，CODE_SNIPPET 可以被 RUN。
- **字段存在，但没有任何写入方**：
  - `verification_status` 永远是 UNVERIFIED。注入时会按它显示"已验证""已过时""幻觉"，但生产代码中没有任何地方设置这三种状态（[注入渲染](../../src/hivememory/engines/memory_compiler/handlers/prompt.py)）。
  - 关系层从未被写入（第 2.2 节）。
  - 负反馈在强化中有处理逻辑，但没有任何地方发出这个事件。

### 2.4 读取与交付

**两类 Profile**

- **USER_PROFILE** 是普通类型：由被动提取与写入产出，只能经相同的相似度检索找到，系统中没有任何常驻注入。它不按生命力归档，理由是"长时间未被检索不代表已经过时"（[垃圾回收器](../../src/hivememory/engines/lifecycle/garbage_collector.py)）。合并不排除它，合并结果的类型由模型给出，一条偏好可能被并进其他类型的记忆。
- **AGENT_PROFILE** 按 alias 精确解析：执行者自己的 Profile 与 CALL 的被调用方都不经检索（`RetrievalFamiliar.get_agent_profile`）。但运行时发现子代理只能靠检索命中：检索上下文中的"可用子代理"一节，只由这一轮检索结果中恰好出现的 Profile 组成（`MemoryCompiler` 的检索上下文分节）；MTP 提示词也只让 agent 在"记忆上下文中列出了可用子代理"时委托。运行时没有列出全部可用 agent 的途径，完整列表只供 HTTP 管理入口使用。
- AGENT_PROFILE 还以一组分散的类型判断，被排除在记忆原子的大部分行为之外：只能经 Profile 创建入口创建；查重不提名它合并，合并任务拒绝它，合并结果也不能变成它；不按生命力归档；不出现在管理页的记忆列表中；不在生成提示词的类型之内，所以交互经验永远不会产出或修订一个 Profile，它只能手动编辑。生命力照常计算和展示，但不据此做任何处理。

**记忆库中像工具的成员与外部入口**

- 外部 harness 经 MCP 只有检索、读取、写入、更新四个工具（[MCP 入口](../../src/hivememory/mcp_server/server.py)）；RUN 与 CALL 只在 Alice 的 MTP 中存在。
- 记忆库中像工具的成员都走特殊路径：AGENT_PROFILE 被 CALL 解析，并被一组类型判断排除在记忆原子的大部分行为之外（第 2.4 节）；CODE_SNIPPET 被 RUN，由本地执行器运行，面向执行的编译目标 `RUNNABLE_TOOL` 仍是保留项，编译时直接报错（[编译目标](../../src/hivememory/engines/memory_compiler/handlers/targets.py)）。
- RUN 把整段正文当作 Python，在子进程中执行：内置函数走白名单，禁用 import、open、exec、eval，参数经 `params` 字典传入（用户工具（`agent_runtime/tools/user.py`，2026-11-17 删除）、[沙箱执行](../../src/hivememory/agent_runtime/tools/repl.py)）。生成链写出的正文是 Markdown（带说明文字与代码围栏，代码可能不是 Python，也可能是配置文件），写入提示词还要求补全 import 语句，所以被动提取与 WRITE 产出的 CODE_SNIPPET 基本运行不了；能运行的只有专门写成"纯 Python、不用 import、读取 `params`"的记忆。运行结果不回流到验证状态。
- 被动提取把出错的工具结果当作提炼错误原因与修复经验的依据（[生成提示词](../../src/hivememory/i18n/prompts.py)的过程条目规则）。

**预检索只在 Alice 中存在**

- Alice 的任务进程在 PREPARE 阶段以主线程做预检索（[任务进程](../workspace/task-process.md)）。
- 外部 harness 上没有预检索。Claude Code 与 Codex 插件的 hook 只在会话开始、结束与每轮结束时登记会话记录，并在写入与更新前为工具参数补上当前一轮的信息（[hooks 配置](../../integrations/hivememory/hooks/hooks.json)）；dsh 与 pi 插件向系统提示词注入的是 MCP 的使用说明，不是记忆。记忆能否进入上下文，取决于 agent 依照这份说明与 skill 自己调用检索。

### 2.5 外部来源与 agent 循环

- **agent 循环属于 Alice**：`agent_runtime` 是走 MTP 的单帧执行内核，分层测试把它与 Alice 视为同一个子系统，Patchouli 不能导入它。
- **附件只留下了元数据**：Alice 对话中的附件在结算产出新建或更新时被提升为 DocumentArtifact（`MemoryGenerationFamiliar._promote_attachment_bindings`），但只写元数据：`source_uri` 指向不跨重启保留的附件存储，没有内容快照，返回的引用被丢弃，没有挂到任何记忆上。
- **外部 harness 的工具结果完整保存，但没有被识别为来源**：网页、文件等工具结果作为轮次事件完整保存在交互 Artifact 中，记忆只能引用整个交互 Artifact。
- **DocumentArtifact 模型是齐的**：来源 URI、取回时间、内容快照地址，以及页码、标题路径、行号、引文等定位符都有，只是没有接入主流程（[Artifacts 与来源追踪](../patchouli/artifacts.md)第 2.2、7 节）。

## 3. 外部调研对设计的启发（我的看法）

### 3.1 Anthropic：The components of agent memory

外部事实见[外部参考](../research/memory-core-external-references.md)第 4 节。

1. "环境能重新告诉你的，不记"给"记忆不该像流水账"一个更清楚的判据（[写入链路](./memory-flow-write-path.md)第 2.4 节第 3 点）。对编码 agent，环境就是仓库、git 与 GitHub：PR 状态、代码结构都能重新读到，不该成为记忆，这也解释了 owner 删掉的那批 PR 记录。
2. "怎么知道它成立"应与记忆放在一起：验证状态现在没有写入方（第 2.3 节），而这一项正是让后来者不必再查的关键。
3. 复核或馆长拿到的应是"未来的 agent 需要什么"，而不是一张固定规则清单；被动提示词中的拆分粒度清单就是这类固定规则。
4. 小而常驻的顶层正是日常路径缺少的（第 2.4 节）：HiveMemory 插件已有 SessionStart hook，现在只登记会话，技术上可以注入一份小顶层；放什么、由谁维护，是[写入链路](./memory-flow-write-path.md)第 2.4 节、[库的形状](./memory-library-shape-and-index.md)第 2.1 节的问题。
5. 共享表互相覆盖的教训支持唯一写入者，也提醒馆长不能并发写同一页（[写入链路](./memory-flow-write-path.md)第 3.1 节边界 4）；"关键条款原文照引"对应[写入链路](./memory-flow-write-path.md)第 3.2 节以引文定位来源。
6. HiveMemory 保存了轨迹却无法检索：MCP 的四个工具都只针对记忆，交互原文没有检索入口。
7. 差异：文章的场景是单 agent、文件夹存储，写笔记与做整理的是同一个模型，没有被动提取，前提是强模型自己会判断什么该记；HiveMemory 还要兜住不主动写的 agent 与纯聊天，所以被动路径不能取消，但它的角色应从"提取"转向复核与维护。文中数字来自他们自己的基准（数据库问答、Pokémon、法律文书与公司语料），只作方向参考。

### 3.2 Perplexity：Brain: Agentic memory as a knowledge wiki

外部事实见[外部参考](../research/memory-core-external-references.md)第 5 节。

1. **三层冗余是[库的形状](./memory-library-shape-and-index.md)第 4.1 节的一种形态：保留原子，在上面加一层综合。** sessions 约等于 InteractionArtifact，notes 约等于现在的记忆原子，knowledge 是 HiveMemory 缺的那一层（按主题综合、带引用的页面）。两篇文章的证据一致：综合层帮的是跨会话、跨文档的问题，单次会话的事实没有提升（[外部参考](../research/memory-core-external-references.md)第 4 节）。所以它是叠加在原子之上的派生层，不是替换，也符合[库的形状](./memory-library-shape-and-index.md)第 3.2 节"派生物可以重建"。
2. **证据边精确到会话片段**，正是[写入链路](./memory-flow-write-path.md)第 2.1 节质疑 3 所需的证据身份；"同一份证据只算一次"与"对照轨迹纠错"都要靠它。HiveMemory 现在的引用只能指向整个 Artifact。
3. **上下文边就是没有写入方的关系层**：wikilink 正是 `relates_to` 想表达的东西，区别是 Brain 中由后台 agent 维护，并写在正文里。
4. **索引预载有了第二份证据**，而且是随机对照实验。两篇文章都表明开局的小索引很关键；HiveMemory 的日常路径恰好什么都不注入（第 2.4 节），插件已有的 SessionStart hook 是现成的接入点。
5. **Dream 的护栏是[写入链路](./memory-flow-write-path.md)第 3.1 节"写的一侧收紧"的一个完整实现**：从上一版出发、整体暂存、确定性校验加语义核验、受控同步、版本历史。HiveMemory 已有唯一写入者与版本记录，缺的是"一次运行的改动整体暂存、校验通过后统一提交"，与[库的形状](./memory-library-shape-and-index.md)第 3.2 节"有事务的记录系统"是同一件事。
6. **删除记录与长期指令对应 owner 的使用方式**：owner 删掉的流水账与 PR 记录，应作为决定被复核或馆长尊重，不能下次又提取回来；"不为 PR 单独写工作记录"这类偏好可以作为长期指令交给馆长，而不是写死在生成提示词里，这也呼应[外部参考](../research/memory-core-external-references.md)第 4 节"只说明记忆的用途，不规定写法"。
7. **检索子 agent 提示了馆长的第二个角色**：除了后台整理，馆长也可以在前台当参考咨询台，外部 agent 问"关于 X 我们知道什么"，馆长返回一段带引用的综合，同时保护调用方的上下文窗口。这是对[写入链路](./memory-flow-write-path.md)第 3.1 节的延伸设想。
8. **第 7 节的评判方式有了可借鉴的做法**：组件级对比（只开或只关某一个机制，例如复核，其余不变）、题目与具体证据绑定、固定题目反复回放。它用来验收已建成的机制，不是推出设计，与"样本统计不作为设计依据"（第 1.2 节）不冲突。
9. **差异与保留**：按[写入链路](./memory-flow-write-path.md)第 4 节的划分，Dream 把写入（由新会话触发的第 2 至 4 阶段）与维护（移除过时内容）混在一起，可以借鉴做法，边界仍按 owner 的划分；Perplexity 有被动提取（评估中写明账户经过"记忆提取、会话摘要、Dream 编译"），没有提到 agent 的主动写入，属于被动一类加一个后台编译器；Brain 是单用户、产品内置、在沙盒中运行，HiveMemory 是多个 harness、多个 agent 共用一个库，并有可见范围的约束；数字来自他们自己的评估，离线提升为中等幅度，只作方向参考。

### 3.3 原始历史优先的一派：Obelisk 与 Nowledge Mem

外部事实见[外部参考](../research/memory-core-external-references.md)第 6 节。
1. **印证**：图索引是路由结构而不是真相、赢在读得少（[库的形状](./memory-library-shape-and-index.md)第 2.3、3.2 节），至此 Anthropic、Perplexity 与这一派的证据指向同一点：视图加原文最好，视图不能替代原文。"复述不是验证"（[写入链路](./memory-flow-write-path.md)第 2.1 节）在"锁定设计"的案例中有了现实的反例；局部修改而不是整篇重写（[写入链路](./memory-flow-write-path.md)第 2.1 节对 Mode C 的质疑）与 live-update 的做法和它对 wiki 页面"膨胀、互相覆盖、断链"的批评一致；新旧结论之间有明确关系、被取代的保留在图上，对应[库的形状](./memory-library-shape-and-index.md)第 2.4 节的断定边。
2. **挑战**：HiveMemory 保存了原始交互却不能检索，MCP 的四个工具都只针对记忆（第 2.4 节）；记忆原子作为主要产物有交"记忆税"的风险，所以它只能是补充，不能是唯一的入口（应对方向已在[写入链路](./memory-flow-write-path.md)第 2.1 节、[库的形状](./memory-library-shape-and-index.md)第 2.3 节：原文保留、证据可定位、索引只做导航）。关于"统一格式有损"，owner 指出完整性的上限其实在 harness（第 1.2 节，见[写入链路](./memory-flow-write-path.md)第 2.2 节的两种处境）。
3. **评判的基准**：以"直接检索原始历史"为对照，衡量新流转的净收益（第 4.2 节）。
4. **对类型体系的输入**：执行记忆与社交情节记忆的区分对应 agent 工作式与 chatbot 式，两者靠不同的方式被找到（锚点与语义）；Nowledge 的六层可以对照 HiveMemory：轨迹即交互记录，单元即记忆原子，实体即主题节点，演化即时间轴与版本链，社群即主题聚类，结晶即跨多次经历归纳出的知识或 skill（[写入链路](./memory-flow-write-path.md)第 2.4 节）；它把"决定"单独列为一类，HiveMemory 现在没有这一类。
5. **保留**：三篇论文测的都是问答准确率，没有测跨 agent 的知识传递、偏好的遵守、自主维护的成本或"下一位 agent 该怎么做"，而这些正是 HiveMemory 的目标；帖子把结论说得比论文宽。帖子与演讲提纲中的一些数字（例如"抽成原子事实掉 19.62 分"）没能在摘要中核实。Obelisk 的记忆层要人批准，Nowledge 的维护也要用户批准；owner 的目标是全自主维护，所以审计能力（版本加证据）对 HiveMemory 更重要，它是"人批准"的替代。

## 4. v0.8 的顺序、辅线与评判

### 4.1 v0.8 的顺序、辅线与仍未定论的事项

**v0.8 的顺序**（owner，2026-11-16，第 1.1 节）。2026-11-17 起按版本推进：v0.8.0 的版本目标为生成核心的升级（按 2026-11-16 的顺序取第 1 步，Claude 整理，待 owner 确认），第一个方向是记忆类型的收敛（[类型体系](./memory-type-system.md)第 7 节开头，建立计划前的两项已于 2026-11-17 决定）；馆长 agent 等其余方向在阻塞问题决定后再建立计划。

1. 生成核心的升级：Patchouli 馆长 agent（[写入链路](./memory-flow-write-path.md)第 3.1、3.2 节）与类型的更迭（[类型体系](./memory-type-system.md)，首份计划的范围见其第 7 节开头）。馆长的第一步是以固定流程对照已有记忆复核（[写入链路](./memory-flow-write-path.md)第 2.1 节）；在它之前，先让生成出来的记忆指向具体的交互轮次（同文第 5 节，owner，2026-11-18）。
2. 被动生成：被动结算具体产出什么，主要取决于提示词怎样写。
3. 主动写入。

**图索引的位置：辅线**。图索引在复核之后逐步加入（话题保留为经历节点 → 馆长建立主题节点 → 常驻索引），作用是让复核的候选找得更准。仍未定论的事项：

| 事项 | 状态 |
|:---|:---|
| 被动生成看到什么材料（只看轨迹，还是同时看本轮动过的记忆与相关记忆） | 已决定：被动结算知道话题以来的记忆事件，以全局视角复核（[写入链路](./memory-flow-write-path.md)第 5 节，owner，2026-11-15） |
| 两条写入路径在哪里汇合 | 被动一侧已决定，在中期记忆上汇合；主动写入生成时的材料仍开放，有候选"写入即同步点"（[写入链路](./memory-flow-write-path.md)第 2.2、5 节） |
| 第一当事人"没写"算不算判断；主动写入何时生效 | 后一问已决定：写下即生效，修订前内容有效（[写入链路](./memory-flow-write-path.md)第 5 节）；前一问未决定 |
| 什么值得写成记忆 | 两条原则（"会不会改变决定"、"环境能重新告诉你的不记"，[写入链路](./memory-flow-write-path.md)第 2.4 节、[外部参考](../research/memory-core-external-references.md)第 4 节），未决定 |
| 修订的形态（结构化判定、局部修改） | 候选设计（[写入链路](./memory-flow-write-path.md)第 2.1 节） |
| 证据定位到轮次或事件 | 已决定：生成事件带上轮次与事件的信息（[写入链路](./memory-flow-write-path.md)第 5 节）；先于馆长的第一步实施（owner，2026-11-18），建立计划前要决定粒度与由谁读取两项（同文第 5 节）；外部来源的定位见[写入链路](./memory-flow-write-path.md)第 3.2 节 |
| 结算时机与"一段经历是否结束"的关系 | 只提过，没有讨论 |
| 保存形态 | [库的形状](./memory-library-shape-and-index.md)第 4.2 节，开放 |
| 短期记忆在新流转中的定义 | 有候选定义（[写入链路](./memory-flow-write-path.md)第 2.2 节），未决定 |
| 用什么评判新的流转更好 | 有候选（第 4.2 节），未决定 |
| 记忆的类型体系 | 首份计划（WIP 与可运行的移出、资源记忆的合并、v0.8 关闭 RUN 的用户工具路径，以及已有记忆的迁移）已于 2026-11-17 实施并归档，作为 v0.8.0 的第一个方向；第二份计划 AgentProfileStore（Agent Profile 移出中期库，改为一个 agent 一个文件）已于 2026-11-18 实施并归档，这个方向的两份计划都已完成（[类型体系](./memory-type-system.md)第 7 节开头）。FACT 的拆分、用户一侧、经历等内容类别的重新划分也有决定，但回放显示与旧类别没有可测的区别，不进入首份计划；默认类型、新类型的生命周期等问题搁置（[类型体系](./memory-type-system.md)第 9.2、10 节） |
| Patchouli 馆长 agent | 第一步（固定流程的复核）排在记忆指向交互轮次之后（owner，2026-11-18），建立计划前要决定判定的格式、复核的输入与修订的粒度三项（[写入链路](./memory-flow-write-path.md)第 3.1 节）。其余只有候选设计（同文第 3.1、3.2 节）：循环放在哪里（可能需要一份 ADR）、结构化判定的格式、由什么触发一次运行、与唯一写入者 MemoryGenerationFamiliar 的分工，都未决定；馆长的写操作取决于保存形态，后者宜在馆长之前或同时决定 |
| 读的一侧（日常路径上怎样送达、怎样从读者反推好的写入、使用中的反馈） | 只零星提过（[库的形状](./memory-library-shape-and-index.md)第 2.1 节、[外部参考](../research/memory-core-external-references.md)第 4 节） |
| 现有记忆的去向（新流转上线时旧库要不要复核、重新归类、挂进索引） | 没有讨论 |

### 4.2 评判方式（候选，不是 owner 决定）

按 owner 的判断（第 1.2、1.3 节），记忆的价值在长时间段内衡量。评判分两层：

1. **单轮的底线：不交记忆税**。在同一组题目上，新流转不比"直接检索原始历史"更差（[外部参考](../research/memory-core-external-references.md)第 6 节）；可以与 Perplexity 的组件级对比合用：只开或只关某一个机制，题目与具体证据绑定，固定题目反复回放（第 3.2 节第 8 点）。这是必要条件，远不够。
2. **长程的标准：演化质量与人工介入**，这是 HiveMemory 真正要证明的。可衡量的有：
   - 人工介入量：owner 每周手动删改的记忆数量，即"全自主维护"最直接的度量；
   - 使用效果：记忆被用上的比例，以及用上之后是否有效（[写入链路](./memory-flow-write-path.md)第 2.4 节第 4 点的结果信号）；
   - 库的健康走势：重复、过时与互相矛盾的记忆随时间是收敛还是累积；
   - 维护成本：每周花在复核与维护上的 token。

owner 每天实际使用 HiveMemory，这本身就是现成的长程试验场；这里的数据用于验收已建成的机制，不用于推出设计，与"样本统计不作为设计依据"不冲突。

**类型体系的回放**（owner 的方向，2026-11-16；具体设计是候选）：新的类型体系先在已有的对话数据上回放一遍，检验 LLM 能否稳定把握它的边界（[类型体系](./memory-type-system.md)第 9 节）。

- 数据：已有的结算材料（此前抽样用过的一批是 110 次结算、616 条草稿，owner 删除过的记忆就在其中）。
- 做法：用原型提示词把同一批材料按变体 A（已决定的类型）与变体 B（用户一侧合为一类）分别重新生成，每个变体重复几次，或换一个模型再跑一遍。
- 指标：
  1. 一致性：同一份材料多次分类的结果是否一致，逐条边界统计；
  2. 混淆：哪两类最常被混在一起；
  3. 碎片：每次结算产出几条记忆，与现在对比（第 1.1 节的现象）；
  4. owner 删除过的记忆：新的体系是否还会产出同类内容，例如进度快照、流水账。
- 取舍：默认选更简单的变体；复杂的变体只有在多出来的区分被稳定判对、并确实改变出现位置或修订方式时才保留。
- 离线运行，在开发实例或临时目录中进行，不碰日常实例的数据。

第一轮回放与按描述直接选的对照组已于 2026-11-16 完成，做法、结果与解读见[类型体系](./memory-type-system.md)第 9.1、9.2 节：按描述直接选时新旧类别同样稳定，类别依旧不平衡，能否引导按预期用法使用测不了；首份计划因此收窄（同文第 7 节开头）。数据集与运行结果在开发实例的 `.hivememory/replay/`，不进版本控制；持久的回放需求由[回放脚本](../../scripts/memory_replay.py)承担（同文第 9.1 节）。

**公开评测集**：现在没有带标准答案的数据，owner 于 2026-11-16 决定把公开评测集与 agent 工作流数据集提上日程；候选评测集、接入要解决的问题以及它们在上面两层中的位置见[评测集](./memory-evaluation-benchmarks.md)。

**统一的检验题**：让一个候选机制连续跑一个月，库会收敛还是退化？前面讨论中好几个问题都属于这一类：按相似度合并使细节漂移、整篇重写在多次修订后走样、过时的状态一直累积、索引的错误自我强化（[库的形状](./memory-library-shape-and-index.md)第 2.3 节）。好的机制应有这些长程性质：重复执行稳定（同一份证据再复核一遍，结果不变）、增长有界、错误能被纠正、人工介入作为一项成本记账。

这也连到类型体系：事实会过时、决定会被推翻、状态会到期、偏好缓慢变化、做法越用越好；按行为划分类型时，最核心的维度可能就是"这种记忆在一个月里会发生什么、该怎样维护"。

## 5. 搁置项

- 用强化学习让 agent 自发使用记忆工具：对闭源模型不可行，不在 v0.8 的范围内（owner，2026-11-14）。
- 关系层字段没有生产使用方：关联表引入之前不在其上构建，记录系统与关联表落地时在同一次改动中删除（契约变更，[库的形状](./memory-library-shape-and-index.md)第 2.4 节）。
- Patchouli 的 agent 形态与 agent 循环的归属（Patchouli 自带、下沉共用，或作为 Alice 的 frame 运行）：不阻塞第一步的固定流程，agent_runtime 的通用化作为并行主线（[写入链路](./memory-flow-write-path.md)第 3.1、3.2 节）。按 owner 的顺序，馆长 agent 属于生成核心的升级（第 4.1 节），馆长从固定流程走向 agent 时就要回答这一项。
- "记忆作为工具"的对象与目标要从头考虑（owner，2026-11-14）：不阻塞 v0.8 的流转设计，[写入链路](./memory-flow-write-path.md)第 3.3 节的分析留作出发点，开放问题仍由 [MaaT Idea](./memory-as-tool-skill-plugin-boundary.md) 维护。
- 话题归并由 LLM 判定、不能保证合理：由[跨会话话题 Todo](../todo/cross-conversation-topic-placement.md)跟踪。
- 被动记忆保留"进行中"的措辞、细节过细：不单独打补丁，随流转的重新设计处理（owner，2026-11-13）；数量与保留率见[被动提取数量 Todo](../todo/passive-extraction-volume-and-retention.md)。
- 2026-11-14 统计过多记忆提取以来被动提取的产出（110 次结算、616 条草稿，其中工作状态类被删除的比例明显高于其他类型）。owner 认为它对重新设计基本没有作用，不作为依据；统计脚本没有保留。

## 6. 与其他文档的关系

- 本讨论的子文档：[写入链路](./memory-flow-write-path.md)、[库的形状、图索引与保存形态](./memory-library-shape-and-index.md)、[类型体系](./memory-type-system.md)、[评测集](./memory-evaluation-benchmarks.md)；外部项目与文章的事实见[外部参考](../research/memory-core-external-references.md)。
- [统一的交互记忆](./unified-interaction-memory.md)：轨迹、断言、整理与消费水位的设想；[写入链路](./memory-flow-write-path.md)第 5 节"生成之前汇合"的一种做法。
- [短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md)：短期这一层自身的压缩与折叠。
- [写入意图的后续方向](../archive/ideas/pending-intent-migration.md)（2026-11-16 已归档）：登记、派发与结算的后续已实施；主动生成时的材料由[写入链路](./memory-flow-write-path.md)第 5 节跟踪。
- [生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md)与记忆价值体系的重新设计（[ROADMAP](../ROADMAP.md) 第 4.7 节）：维护一侧；落库之后的演化质量（分裂与合并、冲突检测、对照轨迹纠错）记在其方向 6，边界见[写入链路](./memory-flow-write-path.md)第 4 节。
- [Memory-Centric Agent TDA](./TDA_Memory_Centric_Agent_Ideas.md)：多视图的记忆图能否优于平面检索，与[库的形状](./memory-library-shape-and-index.md)第 4.1 节相关。
- [耐久性与故障恢复](../governance/reliability/durability-and-recovery.md)：跨存储写入的事务边界、对账与 archive、revive 的恢复，与[库的形状](./memory-library-shape-and-index.md)第 3.2 节相关。
- [MaaT 与 Skill/Plugin 边界](./memory-as-tool-skill-plugin-boundary.md)：可执行记忆与 RUN 的长期草案；owner 于 2026-11-14 提出其对象与目标需要从头考虑（本文第 1.2 节、[写入链路](./memory-flow-write-path.md)第 3.3 节）。

## 7. 升级为计划的条件

- owner 判断仍开放的两项：主动写入生成时的材料（[写入链路](./memory-flow-write-path.md)第 5 节）与保存形态（[库的形状](./memory-library-shape-and-index.md)第 4.2 节）；[库的形状](./memory-library-shape-and-index.md)第 4.1 节与[写入链路](./memory-flow-write-path.md)第 5 节的被动一侧已于 2026-11-15 决定。统一交互记忆 Idea 第 5 节中仍开放的是"没写"算不算判断；写入链路上其余未定论的事项见第 4.1 节。
- 确定 v0.8 用什么评判新的流转更好：样本统计不作为依据（第 1.2 节），候选的评判方式见第 4.2 节（单轮不交记忆税，长程看演化质量与人工介入），尚未决定；公开评测集的选择与接入见[评测集](./memory-evaluation-benchmarks.md)第 6、7 节。
- 类型体系：首份计划的范围已于 2026-11-16 决定，2026-11-17 定为 v0.8.0 的第一个方向，同日实施并归档（[类型体系](./memory-type-system.md)第 7 节开头）；内容类别的重新划分搁置（同文第 9.2、10 节）。
- 下一份计划的切片（owner，2026-11-18）：先让生成出来的记忆指向具体的交互轮次（[写入链路](./memory-flow-write-path.md)第 5 节），再做馆长的第一步，以固定流程对照已有记忆复核、输出结构化判定（同文第 2.1、3.1 节）；类型的更迭的两份计划已完成（[类型体系](./memory-type-system.md)第 7 节开头）。
