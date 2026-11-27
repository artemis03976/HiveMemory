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
  - src/hivememory/engines/generation/librarian.py
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
  - docs/archive/ideas/unified-interaction-memory.md
  - docs/ideas/short-term-memory-and-context-compaction.md
  - docs/archive/ideas/memory-as-tool-skill-plugin-boundary.md
  - docs/governance/reliability/durability-and-recovery.md
  - docs/patchouli/artifacts.md
  - docs/todo/generation-dedup-decision-signal.md
  - docs/todo/passive-extraction-volume-and-retention.md
  - docs/todo/cross-conversation-topic-placement.md
last_reviewed: 2026-11-27
---

# 记忆核心：短期到中期的流转

## 0. 文档性质与阅读路径

本文是 v0.8 记忆核心的总纲：v0.8 的目标与范围、跨子文档的原则、各子文档共用的代码现状、v0.8 的顺序与评判方式。各个设计领域分别在子文档中维护，本文不重复它们的设计：

| 文档 | 内容 |
|:---|:---|
| 本文（总纲） | 目标与原则（第 1 节）；共用的代码现状（第 2 节）；外部调研中尚未写进子文档的启发（第 3 节）；顺序、开放问题的位置与评判方式（第 4 节）；搁置项、与其他文档的关系和升级为计划的条件（第 5 至 7 节） |
| [写入链路](./memory-flow-write-path.md) | 对照已有记忆的判断与修订、主动写入的背景与短期记忆、时间线、什么值得写；馆长与结算；来源；写入与维护的边界；两条写入路径的汇合 |
| [库的形状、图索引与保存形态](./memory-library-shape-and-index.md) | 平铺的库、时空图索引、关系层；保存形态 |
| [类型体系](./memory-type-system.md) | 类型的用途、判据、已移出的三处与剩下的类型 |
| [评测集](./memory-evaluation-benchmarks.md) | 用公开评测集与 agent 工作流数据检验记忆核心 |
| [外部参考](../research/memory-core-external-references.md) | 调研过的项目与文章的外部事实 |

三层存储的当前设计见 [MemoryLibrary 与存储层](../patchouli/memory-library.md)，生成、查重与合并见[记忆生成](../patchouli/generation.md)，话题与沉淀见[短期记忆](../architecture/short-term-memory.md)。正文是已定的设计，候选设计标“候选”。

## 1. 目标与原则

### 1.1 v0.8 的目标与范围

- **重新设计机制，不为症状打补丁**：v0.8 重新定义并设计记忆从短期到中期的流转。为一个机制找补缺陷，相当于无限制地允许它复杂化。
- **被动生成的质量是现象，根因在结构上**：一次结算可以产出多条记忆之后，模型倾向于保留每个细节，较长的对话产出三四条甚至更多；碎片多了，系统就依赖查重大量合并，数量一大，查重的失衡就暴露出来。过滤被动生成的重复记忆暂时缓解了碎片，但“流水账”“查重”这些问题已经不是修复被动生成的质量能解决的。根因是记忆系统的主体设计在最初的原型完成后几乎没有变化，又等到项目基建完善才被验证。查重的判定信号重要，但只是下游策略（[查重 Todo](../todo/generation-dedup-decision-signal.md)）。
- **目标是全自主地维护记忆库**：原型证明了能保留重要证据并反哺 agent，但质量与自主程度离目标还很远。维护固然重要，但写入从一开始就决定了维护的难度。
- **三层的含义不变**：短期是话题工作台，中期是 Qdrant 中可检索的记忆，长期是归档记忆的冷藏库。长期层在项目开始时就已定下，计划改用 sqlite 一类数据库存储，有自己的维护规则，不在这次重新设计的范围内。

### 1.2 跨子文档的原则

- **两条写入路径都保留**：被动提取与 agent 的主动写入是大多数 agent 记忆项目会采用的两种做法，人本来就有“潜意识记忆”与“主动记忆”两种行为（[写入链路](./memory-flow-write-path.md)第 1 节）。
- **两种交互形式都在范围内**：chatbot 式的交互天然像流水账，倾向于记住“用户说的”；agent 工作式的交互要记的是工作要点。两侧的信息同样重要，真实情况比这复杂得多，不能只凭这几点概括设计。
- **记忆即知识**：持久化、可寻址、可追溯、可演化的记忆资产是 agent 所有知识的外部积累，记忆生成链路要把材料统一内化，成为真正属于 agent 的知识（[写入链路](./memory-flow-write-path.md)第 2.4 节）。
- **价值由写入者判定**：只有终端的 Patchouli 才有资格决定“看到了什么但不写成记忆”，入口不给价值标记（[写入链路](./memory-flow-write-path.md)第 1 节）。
- **写入与维护分属**：dreaming 一类的工作属于生命周期维护；记忆的分裂与合并、冲突检测属于落库之后的演化质量（[写入链路](./memory-flow-write-path.md)第 4 节）。
- **知识不等于工具**：知识由模型的推理消费，工具由环境执行；“记忆作为工具”的对象与目标要从头考虑，在孵化中（[写入链路](./memory-flow-write-path.md)第 3.3 节）。
- **动态检索、渐进式披露**：记忆动态检索加载，保留从索引到正文的渐进式披露；存储平铺，分层由派生的时空图索引承担（[库的形状](./memory-library-shape-and-index.md)第 1 节）。
- **记忆的价值在长时间段内衡量**：agent memory 的重点在可演化与动态维护；一个系统连续跑一周、一个月，它积累的内容能否有效用在未来的运行中，以及需要多少人类的介入，才是这个领域真正的价值。单轮召回只是 RAG（第 4.2 节）。
- **样本统计不作为设计依据**：设计从底层机制出发，不从数据推出。样本以 owner 自己的 HiveMemory 开发会话为主，从中得到的特征只对 owner 现在的使用成立，不能推广；数据只用来验收已建成的机制。
- **完整性的上限在 harness**：作为第三方拿不到真正完整的会话记录，保存了原始会话记录也改变不了这一点；owner 因此一直在开发自己的 harness，让会话记录透明、可见、可操控（[写入链路](./memory-flow-write-path.md)第 2.2 节）。

## 2. 现状（以代码为准）

### 2.1 写入链路：两条路径、生成材料、判断与修订

**两条写入路径的相遇**

- 交互材料有意不渲染 WRITE、UPDATE 与 CALL（[交互材料](../../src/hivememory/engines/perception/material.py)）。
- 话题结算有两种方式，由用户选择（[记忆生成](../patchouli/generation.md#31-mode-a被动观察--archive)第 3.1 节）：一次性生成只看话题材料，不带中期库中的任何记忆，两条路径只在查重时碰面：控制面记下本话题生成的记忆（`GenerationRequest.topic_generated_memory_ids`），被动草稿查重命中其中一条时丢弃；馆长先以工具查看这些记忆，再检索库中相关的记忆，然后只新建，不查重、不修订。
- 生成提示词按模式分开（被动、写入、更新、合并，以及馆长），被动一侧用同一份提示词处理所有来源，其中已经有针对 agent 工作过程的规则：思考不作为事实依据，工具调用与结果可以作为事实依据，子 Agent 交回的消息按结论对待（[生成提示词](../../src/hivememory/i18n/prompts.py)）。

**主动写入生成时的材料**

- 主动写入生成时的材料是话题的折叠摘要与最近若干个已完成的轮次；话题不存在时按空材料生成（[生成协调](../../src/hivememory/patchouli/control/memory_generation/coordinator.py)）。
- 外部 harness 的交互在轮次结束后才经 Import Bus 导入，所以写入的那一刻，当前这一轮还不在话题中；“一句话直出”时话题是新的、空的，生成侧只有 agent 在 WRITE 中写下的内容、标题与理由。
- 写入前的 hook 读取本地的会话记录，但只为工具参数补上当前一轮的用户消息与之前几轮的文本，用于确定话题，不带本轮的工具事件（[Claude Code hooks](../../integrations/hivememory/connector/claude_code/hooks.py)）。

**判断与修订**

- 外部的价值标记已经移除，代码中没有 `worth_saving`。
- 修订以编辑集落地：馆长的修订、Mode C 与合并都输出编辑集（片段替换、追加、整篇改写正文与逐个增删标签），以计算时读到的版本为底，提交前按同一条规则重放（[记忆生成](../patchouli/generation.md#34-编辑集修订已有记忆的共同形式)第 3.4 节）。
- 修订可以有限地改索引层：标题与摘要只做片段替换，标签逐个增删，编译文本变化时向量重算（[编辑集](../../src/hivememory/engines/generation/edits.py)）。管理页的手动编辑可以改标题、摘要、标签与正文（[记忆生成服务](../../src/hivememory/patchouli/services/memory_generation.py)）；修订、合并与手动编辑都不改类型，类型只在产生时决定一次。
- 证据引用到段：事件记下来自本轮第几次模型响应，段由事件派生；被动提取、主动写入、修订与馆长的模型引用段，代码换算为轮次标识加事件序号区间，保存在这一版记忆的版本记录中。记忆原子中的 `ArtifactRef` 仍指向整个 Artifact；版本记录中的引用现在没有读取方（[记忆生成](../patchouli/generation.md#22-段标签与记忆引用)）。

### 2.2 中期库：平铺、查重与生成形式

**查重只看索引层**：草稿按 `Title / Type / Tags / Summary` 四行编译成稠密向量的输入（[embedding handler](../../src/hivememory/engines/memory_compiler/handlers/embedding.py)），在整个 Workspace 中取最近的一条，按阈值判为相同、相似（先新建，再由合并任务补进命中的记忆）或不同（[记忆生成](../patchouli/generation.md#4-查重与合并)第 4 节）。正文不参与，正文的长短影响不了相似度。查重只服务主动写入与一次性生成。

**中期库是平铺的**：

- `MemoryAtom` 只有 Workspace 归属、来源、读取策略与生命周期，索引层（标题、摘要、标签、类型、alias），正文与 Artifact 引用；关系层的 `relates_to`、`supersedes`、`depends_on` 是预留字段，生产代码不写入，前后端都没有读取方，却被序列化进每一个 Qdrant point，记忆版本快照也把它列为必备字段（[记忆模型](../../src/hivememory/core/models/memory.py)）。
- 检索的业务过滤只有类型、来源 agent、时间范围、标签与最低置信度（`QueryFilters`），没有“属于哪一组”。
- 话题是系统中唯一按内容归组的结构，但只存在于短期：沉淀之后话题留在话题池，退役之后、最后一次沉淀完成时连同它的日志删除（[短期记忆](../architecture/short-term-memory.md#46-退役与删除)第 4.6 节）。之后一条记忆与它的话题只剩间接的联系：它引用的来源 InteractionArtifact 记着 `topic_id`。
- 合并以另一种形式记录了“取代”：被吸收的记忆被删除，旧 alias 改指到保留的记忆，版本记录以 MERGE 来源记下这次合并。

**生成的形式为平铺检索而定**（[生成提示词](../../src/hivememory/i18n/prompts.py)）：被动提取要求“将内容拆解为独立、自包含的知识点”，标题与摘要要让人能单独检索到它；主动写入沿用 agent 拟定的标题，但标题缺少让人检索到它的关键名词时要修改；合并要求标题与摘要覆盖合并后的全部要点，因为检索只使用标题、类型、标签与摘要，不读取正文。

### 2.3 存储：真相、一致性与记忆原子中的非文本部分

**存储层的真相与跨存储的一致性**

- 中期层的真相就是 Qdrant：向量、索引层与正文在同一个 point 上，它是检索、生成与生命周期共同依赖的当前记忆事实库（[MemoryLibrary 与存储层](../patchouli/memory-library.md#12-中期当前可检索书库)第 1.2 节）。
- 跨存储的写入都不在一个事务里（[耐久性与故障恢复](../governance/reliability/durability-and-recovery.md#1-当前状态与问题证据)第 1 节）：MemoryAtom 与 Artifact 的写入失败时可能留下未被引用的 Artifact；archive 与 revive 在中期库与冷藏库之间搬运，失败时可能留下重复副本或中间态。
- 读取视图的失效经进程内的变更事件，尽力而为、不重放；读取视图本身不跨重启，所以这样成立。
- 启动时，记忆 payload 只接受 schema 2.1，其他版本拒绝解码；已有的 Qdrant 集合不核对向量配置，存在即直接使用（[向量存储](../../src/hivememory/infrastructure/storage/vector_store.py)）。

**记忆原子中哪些部分与文本等价**

- 与文本等价：索引层相当于 frontmatter；`payload.content` 本身就是 Markdown 正文；Artifact 引用可以用链接表达。
- 不是文本，并且有生产使用方：
  - 使用状态：检索命中时只刷新最近访问时间；actor 经 READ 读取时记录一次引用，访问次数、生命力与衰减基准随之变化（[actor 记忆操作](../../src/hivememory/workspace/capability/actor/memory.py)、[强化](../../src/hivememory/engines/lifecycle/reinforcement.py)）；生命力参与检索排序（[融合排序](../../src/hivememory/engines/retrieval/fusion.py)）与归档。
  - 置信度：生成时给出，注入时显示，也参与检索排序。
  - 读取策略：读取时按 agent 执行。
  - 版本：每次内容提交都必须带版本记录，变更事件让 workspace 的读取视图失效。
- 字段存在，但没有任何写入方：`verification_status` 永远是 UNVERIFIED，注入时却会按它显示“已验证”“已过时”“幻觉”（[注入渲染](../../src/hivememory/engines/memory_compiler/handlers/prompt.py)）；关系层从未被写入；负反馈在强化中有处理逻辑，但没有任何地方发出这个事件。

### 2.4 读取与交付

- **USER_PROFILE** 是普通类型：由被动提取与写入产出，只能经相同的相似度检索找到，系统中没有任何常驻注入；它不按生命力归档，理由是“长时间未被检索不代表已经过时”（[垃圾回收器](../../src/hivememory/engines/lifecycle/garbage_collector.py)）。合并不排除它，合并结果的类型由模型给出，一条偏好可能被并进其他类型的记忆。
- **Agent Profile** 不在中期库中：它是 AgentProfileStore 中的文件，可用 agent 的列表常驻提示词，检索结果中不再有 Profile（[MemoryLibrary 与存储层](../patchouli/memory-library.md#15-agent-profile人偶图纸)第 1.5 节）。
- **外部入口**：外部 harness 经 MCP 只有检索、读取、写入、更新四个工具（[MCP 入口](../../src/hivememory/mcp_server/server.py)），交互原文没有检索入口；RUN 与 CALL 只在 Alice 的 MTP 中存在，RUN 的用户工具路径已关闭，解析到的记忆一律以不可运行拒绝。
- **预检索只在 Alice 中存在**：Alice 的任务进程在 PREPARE 阶段以主线程做预检索（[任务进程](../workspace/task-process.md)）。外部 harness 上没有预检索：Claude Code 与 Codex 插件的 hook 只在会话开始、结束与每轮结束时登记会话记录，并在写入、更新与读当前话题之前为工具参数补上当前一轮的信息（[hooks 配置](../../integrations/hivememory/hooks/hooks.json)）；Claude Code 压缩会话之后注入的只是话题句柄的指针，不含记忆内容（[短期记忆](../architecture/short-term-memory.md#63-推送会话压缩之后的指针)第 6.3 节）；dsh 与 pi 插件向系统提示词注入的是 MCP 的使用说明，不是记忆。记忆能否进入上下文，取决于 agent 依照这份说明与 skill 自己调用检索。

### 2.5 外部来源与 agent 循环

- **agent 循环**：`agent_runtime` 是 Alice 走 MTP 的单帧执行内核，Patchouli 不能导入它；馆长的工具循环由 Patchouli 自己定义（`engines/generation/librarian.py`），动作是文本中的 JSON。
- **附件只留下了元数据**：Alice 对话中的附件在结算产出新建或更新时被提升为 DocumentArtifact（`MemoryGenerationFamiliar._promote_attachment_bindings`），但只写元数据：`source_uri` 指向不跨重启保留的附件存储，没有内容快照，返回的引用被丢弃，没有挂到任何记忆上。
- **外部 harness 的工具结果完整保存，但没有被识别为来源**：网页、文件等工具结果作为轮次事件完整保存在交互 Artifact 中。
- **DocumentArtifact 模型是齐的**：来源 URI、取回时间、内容快照地址，以及页码、标题路径、行号、引文等定位符都有，只是没有接入主流程（[Artifacts 与来源追踪](../patchouli/artifacts.md)第 2.2、7 节）。

## 3. 外部调研中尚未写进子文档的启发（候选）

外部事实见[外部参考](../research/memory-core-external-references.md)第 4 至 6 节；落进设计的部分已写在各子文档中（例如“环境能重新告诉你的不记”、常驻的小索引、证据定位到会话片段、检索子 agent）。还有几点没有写进子文档：

1. **删除记录与长期指令**：owner 删掉的流水账与 PR 记录，应作为决定被馆长尊重，不能下次又提取回来；“不为 PR 单独写工作记录”这类偏好可以作为长期指令交给馆长，而不是写死在生成提示词里。这也呼应“只说明记忆的用途，不规定写法”（Perplexity 的 Brain 与 Anthropic）。
2. **一次运行的改动整体暂存、校验通过后统一提交**：Perplexity 的 Dream 从上一版出发、整体暂存、确定性校验加语义核验、受控同步、保留版本历史。HiveMemory 已有唯一写入者与版本记录，馆长也已是循环结束后统一提交暂存的草稿；缺的是提交前的校验，与[库的形状](./memory-library-shape-and-index.md)第 3.2 节“有事务的记录系统”是同一件事。
3. **审计是“人批准”的替代**：Obelisk 的记忆层要人批准，Nowledge Mem 的维护也要用户批准；HiveMemory 的目标是全自主维护，所以审计能力（版本加证据）更重要。
4. **差异与保留**：这些项目多是单 agent、单用户、产品内置，写笔记与做整理的是同一个模型，没有被动提取，前提是强模型自己会判断什么该记；HiveMemory 是多个 harness、多个 agent 共用一个库，有可见范围的约束，还要兜住不主动写的 agent 与纯聊天，所以被动路径不能取消，但它的角色从“提取”转向对照已有记忆与维护。文中的数字来自它们自己的基准，只作方向参考；论文测的都是问答准确率，没有测跨 agent 的知识传递、偏好的遵守与自主维护的成本，而这些正是 HiveMemory 的目标。

## 4. 顺序、开放问题与评判

### 4.1 v0.8 的顺序与开放问题的位置

**顺序**：

1. 生成核心的升级：类型的更迭（[类型体系](./memory-type-system.md)）、生成的记忆指向交互轮次与 Patchouli 馆长 agent（[写入链路](./memory-flow-write-path.md)第 3.1、5 节）。
2. 被动生成：被动结算具体产出什么，主要取决于提示词怎样写。馆长 agent 是被动生成体系中的一环。
3. 主动写入。

第 1 步中的类型收敛、AgentProfileStore、轮次指向与馆长已经实现，版本与计划见 [ROADMAP](../ROADMAP.md) 第 3 节。

**辅线**：时空图索引在馆长之后逐步加入，先把话题保留为经历节点，再由馆长建立主题节点，最后加入常驻索引（[库的形状](./memory-library-shape-and-index.md)第 2.2 节），作用是让馆长找相关记忆找得更准。

**开放问题在哪里**（各问题的选项与影响只在所在的文档中维护）：

| 问题 | 位置 |
|:---|:---|
| 主动写入生成时能拿到什么材料 | [写入链路](./memory-flow-write-path.md)第 2.2 节 |
| 什么值得写成记忆 | 孵化中 |
| 第一当事人“没写”算不算判断 | 搁置，[写入链路](./memory-flow-write-path.md)第 6 节 |
| 记忆的保存形态 | [库的形状](./memory-library-shape-and-index.md)第 4.2 节 |
| 首个评测切片、原始时间怎样带进评测 | [评测集](./memory-evaluation-benchmarks.md)第 6 节 |
| 用什么评判新的流转更好 | 本文第 4.2 节 |

内容类别的重新划分已搁置（[类型体系](./memory-type-system.md)第 10 节）。

### 4.2 评判方式（候选）

记忆的价值在长时间段内衡量（第 1.2 节）。评判分两层：

1. **单轮的底线：不交记忆税**。在同一组题目上，新流转不比“直接检索原始历史”更差（[外部参考](../research/memory-core-external-references.md)第 6 节）；可以与组件级对比合用：只开或只关某一个机制，其余不变，题目与具体证据绑定，固定题目反复回放（Perplexity 的做法）。这是必要条件，远不够。
2. **长程的标准：演化质量与人工介入**，这是 HiveMemory 真正要证明的：
   - 人工介入量：owner 每周手动删改的记忆数量，即“全自主维护”最直接的度量；
   - 使用效果：记忆被用上的比例，以及用上之后是否有效，现在验证状态没有写入方、负反馈没有发出方（第 2.3 节）；
   - 库的健康走势：重复、过时与互相矛盾的记忆随时间是收敛还是累积；
   - 维护成本：每周花在整理与维护上的 token。

owner 每天实际使用 HiveMemory，这本身就是现成的长程试验场；这里的数据用于验收已建成的机制，不用于推出设计。公开评测集的选择与接入见[评测集](./memory-evaluation-benchmarks.md)；在已有对话数据上回放检验 LLM 能否把握一套设计，做法与结果见[类型体系](./memory-type-system.md)第 9 节，可复用的部分在[回放脚本](../../scripts/memory_replay.py)。

**统一的检验题**：让一个候选机制连续跑一个月，库会收敛还是退化？按相似度合并使细节漂移、整篇重写在多次修订后走样、过时的状态一直累积、索引的错误自我强化（[库的形状](./memory-library-shape-and-index.md)第 2.3 节）都属于这一类。好的机制应有这些长程性质：重复执行稳定（同一份证据再整理一遍，结果不变）、增长有界、错误能被纠正、人工介入作为一项成本记账。这也连到类型体系：事实会过时、决定会被推翻、状态会到期、偏好缓慢变化、做法越用越好，这种记忆在一个月里会发生什么、该怎样维护，是检验类型的维度之一（[类型体系](./memory-type-system.md)第 5 节）。

## 5. 搁置项

- 读的一侧：日常路径上怎样送达、怎样从读者反推好的写入、使用中的反馈，只零星提过（[库的形状](./memory-library-shape-and-index.md)第 2.1 节）。
- 现有记忆的去向：新流转上线时，旧库要不要重新整理、重新归类、挂进索引，没有讨论。
- 被动记忆保留“进行中”的措辞、细节过细：不单独打补丁，随流转的重新设计处理；数量与保留率见[被动提取数量 Todo](../todo/passive-extraction-volume-and-retention.md)。
- 话题归并由 LLM 判定、不能保证合理：由[跨会话话题 Todo](../todo/cross-conversation-topic-placement.md)跟踪。

## 6. 与其他文档的关系

- 已归档的[统一的交互记忆](../archive/ideas/unified-interaction-memory.md)：轨迹、断言、整理与消费水位的设想，曾被视为 v0.8 可能的出发点；消费水位的语义已经退场（[写入链路](./memory-flow-write-path.md)第 2.3 节），断言层作为“生成之前汇合”的做法不采用为当前方向（同文第 5 节），“没写”算不算判断搁置在同文第 6 节。
- [短期记忆](./short-term-memory-and-context-compaction.md)：短期这一层，话题的进展、生命周期、折叠与推送（v0.8.1）。
- [生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md)与记忆价值体系的重新设计（[ROADMAP](../ROADMAP.md) 第 4.7 节）：维护一侧；落库之后的演化质量（分裂与合并、冲突检测、对照轨迹纠错）记在其方向 6，边界见[写入链路](./memory-flow-write-path.md)第 4 节。
- [Memory-Centric Agent TDA](./TDA_Memory_Centric_Agent_Ideas.md)：多视图的记忆图能否优于平面检索，与[库的形状](./memory-library-shape-and-index.md)第 4.1 节相关。
- [耐久性与故障恢复](../governance/reliability/durability-and-recovery.md)：跨存储写入的事务边界、对账与 archive、revive 的恢复，与[库的形状](./memory-library-shape-and-index.md)第 3.2 节相关。
- 已归档的 [MaaT 与 Skill/Plugin 边界](../archive/ideas/memory-as-tool-skill-plugin-boundary.md)：可执行记忆与 RUN 的早期草案，前提已变（RUN 的用户工具路径关闭、CODE_SNIPPET 并入资源记忆）；“记忆作为工具”的对象与目标在孵化中，已作出的边界见 [ADR-0003](../architecture/decisions/0003-memory-as-a-tool-and-mtp-run-boundary.md)。

## 7. 升级为计划的条件

- 计划所在的设计领域中，阻塞它的开放问题已经决定（第 4.1 节的表）。
- 确定用什么评判新的流转更好：候选的评判方式见第 4.2 节，公开评测集的选择与接入见[评测集](./memory-evaluation-benchmarks.md)第 6、7 节。
- 计划所依据的设计已写在对应的子文档中，而不是只在讨论里。
