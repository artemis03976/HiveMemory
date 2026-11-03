---
title: 主动写入与被动提取的收敛
status: idea
horizon: current
serves_version: v0.7.3
owner: patchouli
scope: active-write-and-passive-extraction-overlap-merge-precedence-and-merge-conflicts
code_paths:
  - src/hivememory/engines/generation/
  - src/hivememory/i18n/prompts.py
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/patchouli/control/memory_generation/
related_docs:
  - docs/patchouli/generation.md
  - docs/patchouli/memory-library.md
  - docs/plans/v0.7.3-topic-active-write-precedence.md
  - docs/todo/generation-dedup-decision-signal.md
  - docs/todo/cross-conversation-topic-placement.md
  - docs/ideas/pending-intent-migration.md
  - docs/archive/ideas/settlement-multi-memory-extraction-and-dedup.md
  - docs/research/deepseek-harness-plugin-survey.md
last_reviewed: 2026-11-03
---

# 主动写入与被动提取的收敛

## 0. 文档性质

本文是 v0.7.3 的设计讨论，为该版本的计划供料，不描述当前行为。主动写入、话题结算的被动提取、查重与合并的当前设计见 [Patchouli 生成](../patchouli/generation.md)第 2.1、3、4、7、8 节。查重的判定信号由[查重决策信号 Todo](../todo/generation-dedup-decision-signal.md)跟踪，本文不讨论信号本身；写入意图的登记、派发与结算见[写入意图 Idea](./pending-intent-migration.md)。

第 3 节的问题已由 owner 于 2026-11-03 决定，第 4 节是据此展开的候选设计。计划的状态见第 7 节。

## 1. 背景

owner 对记忆来源的定位与判断：

- （2026-10-31）记忆主要由执行者经记忆工具主动写入，因为只有执行者知道什么是一手且重要的；话题结算的被动提取（Mode A）作为兜底（[已归档的多条提取与查重 Idea](../archive/ideas/settlement-multi-memory-extraction-and-dedup.md#1-owner-决定)第 1 节）。
- （2026-11-01）多条提取定案时确认：主动写入的内容被 Mode A 再提取一遍是早已接受的代价，由查重与合并完成去重与迭代更新（同上）。
- （2026-11-02）复盘 dsh 的调研记忆后：被动提取对主动写入再做一轮合并是已知缺陷，确实会导致记忆生成的质量降级；合并失败后留下重复本是已知行为，但同样让记忆库变乱；这些改动都涉及记忆库的底层机制，适合写成 Idea 而不是 Todo。
- （2026-11-03）又完成两个仓库的调研之后：v0.7.3 回到记忆主线，以主动写入为主体改进生成；这是 owner 想到更好的记忆生成策略之前能做的最小优化（第 3.3 节）。

现行设计接受这份代价，依赖三条前提：

1. 显式写入与更新不渲染进生成材料，Mode A 就不会把它们的内容再提取一遍（[生成](../patchouli/generation.md#21-材料与渲染)第 2.1 节）；
2. 查重命中的两条记忆地位对等，合并时模型按内容更新时间得知哪条较新，状态冲突时以较新的为准（第 4、4.1 节；提示词见 `i18n/prompts.py` 的合并提示）；
3. 先落库再合并，关闭、队列拒绝或合并失败最多留下两条相似的记忆，不会丢失内容（第 4 节）；并发合并同一条记忆时后写回的失败，已列为当前限制（第 8 节）。

此前的真实使用以 Claude Code 开发 HiveMemory 为主，plugin 模式的主动写入在 v0.7.2 才开放，样本里主动写入很少。dsh 调研是第一批以主动写入为主的样本：agent 按 repo-deep-dive 与 HiveMemory 使用手册 skill 在调研结束时写入结论，子 Agent 只读。三条前提在这批样本上都出现了偏差（第 2 节）。

## 2. 证据

### 2.1 样本与来源

- **调研对象**：
  - 2026-11-02：`jack66g/spatial-memory-engine`（SME）调研两轮，两轮之间 owner 改过 skill，第二轮为 standard 档，另有两轮追问；`doiito/gliding_horse`（Gliding Horse）为 deep 档，另有一轮追问。
  - 2026-11-03（UTC）：`cyanwell/dsh-agent-soul`（standard 档）与 `EverMind-AI/EverOS`（deep 档），都使用 2026-11-02 改写后的 repo-deep-dive。
- **统计范围**：SME 第一轮与第二轮在 PR #168（[已归档的子 Agent 会话计划](../archive/plans/dsh-subagent-sessions.md)）部署之前进行，子 Agent 会话被独立导入，生成了以子任务命名的话题与记忆。下文的统计只取主会话话题，子 Agent 话题的记忆只在附录中处置。其余三个仓库都在修复部署之后。
- **数据来源**：
  - 中期库的当前原子；
  - `memory_creation` 与 `memory_version` Artifact：来源意图、每个版本的来源与被吸收的记忆，以及 Mode A 的完整生成视图；
  - 话题结算时的来源 InteractionArtifact：每轮的完整事件，包括写入调用的标题、正文与理由，第 3.2 节的回放以它为输入；
  - dsh 会话日志中 agent 的 `memory_write` 原始参数；
  - 后端日志的提取、查重、存储与合并记录。日志用 WSL 本地时间（UTC−6），下文统一换算为 UTC。
- **证据包**：本机 `.hivememory/evidence/2026-11-02-dsh-research/` 与 `2026-11-03-dsh-research/`，不入库。删除记忆只移除中期库原子，创建与版本记录仍在 `.hivememory/artifacts/` 中，可以继续回放。

### 2.2 Mode A 复现了主动写入的全集

| 话题 | agent 主动写入 | 结算时 Mode A 的草稿 | 对应关系 |
|:---|:---|:---|:---|
| SME 第二轮主话题（11-02 18:34 结算） | 6 条：架构、空间分区、零漂移兼容、排序层、持久化、记忆生成 | 6 条 | 一一对应，相似度 0.786～0.872，全部判为 SIMILAR |
| SME 记忆动力学追问（11-02 18:54 结算） | 1 条 | 1 条 | 0.843，SIMILAR |
| Gliding Horse 主话题（11-02 23:36 结算） | 4 条：架构、证据收据门、收尾门、工作区租约 | 4 条 | 一一对应；3 条 SIMILAR（0.751～0.841），收尾门与已合并的混合记忆只有 0.683，判为 DISTINCT 另立一条 |
| EverOS 主话题（11-03 05:29 结算） | 8 条：架构、Scope 隔离、存储格式、cascade 同步、边界检测、离线引擎、检索分派、派生数据与运行状态 | 8 条 | 一一对应，全部 SIMILAR（0.765～0.898） |

第 1 条前提在字面上成立：Gliding Horse 的生成视图有 210 个条目、约 34.5 万字符，`memory_write` 调用确实没有渲染。但 Mode A 用的是同一份材料：

- agent 写记忆之前的 THOUGHT 列出了计划写的标题，最后的回答又列出写入的记忆。Gliding Horse 的 4 条草稿中有 3 条、EverOS 的 8 条中有 7 条，标题与 agent 写入时的标题相同或只差一两个字；
- 报告正文作为 `write` 工具的参数完整在材料里（Gliding Horse 约 2.8 万字符）；
- 子 Agent 交回的输出也在材料里（Gliding Horse 6 个，约 8.9 万字符）。

不渲染写入只防住了原文回声，防不住同一批结论被重新推导一遍。agent 认真调研、把结论写成记忆时，话题材料里本来就包含每一条结论，Mode A 因此产出同一组记忆。

只含调研的这四个话题上，Mode A 没有贡献主动写入之外的新记忆（唯一另立的收尾门与已有内容重复）。它的实际作用是触发合并，改写 agent 写入的记忆（第 2.3 节）。

dsh-agent-soul 是一个例外，但原因不在生成：它的第一轮被落进了另一段 Claude Code 会话的话题（[跨会话落位 Todo](../todo/cross-conversation-topic-placement.md)），结算时材料混合了 4 轮 Claude Code 与 1 轮调研。这一次 Mode A 只从调研轮次提取了一条 agent 在回答里提到、但没有写入的环境事实（Windows 沙箱下目录写入被拒，用 `icacls /reset` 恢复），其余草稿都来自 Claude Code 的轮次，其中 2 条重复了 Claude Code 同一话题里主动写入的用户偏好。第 3.2 节的回放中，同一份材料再跑一次，Mode A 就复现了 9 条调研写入。

### 2.3 合并按新旧定主次，被动草稿成了骨架

同一话题的被动草稿总在空闲结算之后产生，必然比主动写入新。2026-11-02 被动草稿并入主动写入、且成功写回的 10 次合并中（SME 与 Gliding Horse 主话题的 9 次，加 1 次来自修复前的子 Agent 话题），合并记录的 changelog 都以"较新的 N"为准、为主干或为骨架。EverOS 的 7 次成功合并同样都按新旧定主次，其中 4 次明说以较新的被动草稿为骨架或主干。示例：

- SME 持久化：「以较新的风险清单记忆为准补充了带行号的风险与未决问题章节」，又一次「以较新的 N 为骨架补入 M 的完整风险清单」；
- SME 排序层：「以较新的 N 为主干补入 M 的定位、设计取舍、权重表」；
- EverOS 边界检测：「以较新的 N 为主干补入 M 的可靠性细节与代价条目……去重后形成九项已知代价」；
- EverOS cascade：「同时完整保留 M 的机制描述与七条边界，合并为九条边界」。

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
| EverOS 离线引擎 | 1791 | 3035 |
| EverOS 边界检测 | 1927 | 2869 |
| EverOS cascade 同步 | 1858 | 2624 |
| EverOS 存储格式 | 1523 | 2102 |
| EverOS Scope 隔离 | 1526 | 4791（先并入 agent 自己的检索分派一条，再并入两条被动草稿） |

第 2 条前提把"较新"当作信息更新的信号；对同一份材料上的主动写入与被动草稿，较新只是流水线时序的结果。合并因此系统性地以被动草稿为主体，把子 Agent 输出里的风险清单、行号与未决问题带进了 agent 有意精简过的记忆，两边的代价清单也被叠加在一起。repo-deep-dive 改写后，agent 写入的记忆讲机制与设计理念，审计式的清单又经由合并回到了记忆里。

### 2.4 同一项目的主动写入在写入时互相合并

agent 有意分开写的几条记忆，写入时被查重合并：

| 仓库 | 被合并的主动写入 | 相似度 |
|:---|:---|:---|
| Gliding Horse | 架构 ← 证据收据门，再 ← 收尾门 | 0.768、0.772 |
| dsh-agent-soul | 架构总览 ← 「能力半与身份半」交付判据 | 0.783 |
| EverOS | Scope 隔离 ← 检索分派 | 0.766 |
| SME 第二轮 | 无（0.714～0.724，都判为 DISTINCT） | — |

没有合并的也贴着阈值：dsh-agent-soul 的换气插件与架构总览为 0.749，EverOS 的离线引擎与存储格式为 0.737。repo-deep-dive 改写（每个发现只放一处）之后调研的两个仓库都出现了。Gliding Horse 的 owner 之后又明确要求单独写一条 Agent OS 记忆，它以 0.767 命中第 2.5 节的合并失败残留，被并了进去。

同项目的不同记忆正落在阈值两侧，这是查重 Todo 所说的"同项目但无关"一类；新增的样本已登记在该 Todo。

### 2.5 合并失败

2026-11-01 起的后端日志共有 42 次 SIMILAR 查重决策，合并失败 4 次：

| 结算 | 命中同一条 M 的草稿 | 前一个合并写回 | 失败 | 残留 |
|:---|:---|:---|:---|:---|
| SME 持久化子 Agent 话题（11-02 17:56） | 0.845、0.760 | 17:56:54 | 17:56:55 | 独立的"存储与持久化调研结论" |
| Gliding Horse 主话题（11-02 23:36） | 0.771、0.751 | 23:36:49 | 23:36:51 | 独立的"Agent OS 架构与定位"，之后吸收了 owner 要求单独写的 Agent OS 记忆 |
| dsh-agent-soul 所在的 Claude Code 话题（11-03 03:24） | 0.758、0.820（命中同一话题主动写入的用户偏好） | 03:24:46 | 03:24:47 | 独立的"repo-deep-dive 技能改写"，owner 已删除 |
| EverOS 主话题（11-03 05:30） | 只有 1 条命中这条 M（0.832） | — | 05:30:57 | 独立的 `ref_rebuildable_vs_state` |

前三次的模式相同：同一次结算的两条草稿命中同一条记忆 M（生成队列并发为 2），前一个合并写回 M 1～2 秒后，后一个合并失败，时序与 `version_changed` 相符。第四次不符合：只有一条草稿命中这条 M，合并期间 M 没有被改动，队列里同时运行的另一个合并改的是另一条记忆。

失败原因无法确认：

- 控制器只记录通用文本"记忆生成失败，请检查服务状态"；生成队列的失败分类对一切异常给出同一个原因 `memory_generation_failed`，`MemoryMergeAbortedError` 的 `reason` 在日志与事件中都看不到。
- `work.failed` 事件带异常类名，但事件流只有前端 Workspace Vision 在页面打开时实时显示，没有持久的观察者。进程内事件缓冲为 1000 条，被调度器的心跳占满（话题空闲扫描每 30 秒、模型卸载检查每 60 秒各一对 started/completed），约 2.8 小时就轮换完；11-03 取证时这几次失败的事件已经查不到。

### 2.6 生成丢掉了 agent 写的标题

Mode B 的提示词只收到写入的正文与保存理由，标题只在模型失败时的兜底草稿里使用（`GenerationEngine._build_fallback_draft`）。dsh-agent-soul 与 EverOS 的 17 条写入，标题全部被改写为"主体：要素清单"的形式，其中 4 条加上了"已知代价""可靠性缺口"一类字眼，例如「策略返回事件而不是发布事件：EverOS 离线记忆引擎的契约设计与它的至多一次缝隙」变成「EverOS 离线记忆引擎 OME 的「事件加策略」架构与可靠性缺口」。正文基本保留（与 agent 原文的字符序列相似度 0.74～0.89），增加的篇幅主要是标题层级，以及把保存理由写成"适用场景"。

使用手册 skill 要求 agent 把标题写成这件事本身、具体可检索，同时说明正式标题由生成重新撰写。检索只比较标题、类型、标签与摘要，agent 斟酌过的标题在这一步丢失；而 Mode A 又从材料里把这些标题原样提取回来（第 2.2 节）。

## 3. 决定与回放

问题：查重与合并怎样对待主动写入。来源（主动写入或被动提取）现在不参与查重与合并，第 2 节显示，以主动写入为主的使用方式下，这让被动提取从"兜底"变成了"改写主动写入"。

### 3.1 考虑过的方向

| 方向 | 内容 | 结论 |
|:---|:---|:---|
| A | 维持现状，只改进查重的判定信号 | 被动草稿与主动写入确实是同一件事，信号再准也判为 SIMILAR，改写照常发生 |
| B | 合并时以主动写入为主体，被动草稿只补入主动写入没有的事实 | 不推进：跨话题的被动草稿可能真的带来新信息，按新旧定主次在那里仍然成立；同一话题内的情况由 E 处理，不必改合并语义 |
| C | 材料把本轮的写入渲染为一行，提示词要求不重复提取 | 回放否定（第 3.2 节） |
| C2 | 在 C 的基础上，提示词开头列出本轮已保存的记忆，要求逐条核对 | 回放否定（第 3.2 节） |
| D | 话题中有主动写入时，Mode A 不提取这个话题 | 不推进：这批样本里 Mode A 唯一独有的贡献（`icacls` 环境事实）恰好来自有主动写入的轮次 |
| E | 话题内主动写入优先：被动草稿命中本话题主动写入结算到的记忆时丢弃；主动写入命中本话题其他主动写入结算到的记忆时另立一条、不合并 | 采用（第 3.3 节） |

### 3.2 离线回放（2026-11-03）

- **样本**：SME 第二轮主话题与记忆动力学追问、Gliding Horse 主话题、EverOS 主话题，以及 dsh-agent-soul 所在的混合话题。输入是话题结算时的来源 InteractionArtifact。
- **方法**：
  - 用生产代码渲染材料、用生产提取器与 Librarian 模型（`deepseek/deepseek-chat`）提取，C、C2 只替换渲染函数中写入调用的一行与提示词。按现行规则重新渲染的材料，与当时 Mode A 实际使用的生成视图逐条一致（EverOS 没有可对照的视图）。
  - 每条草稿按查重的编译文本编码，对当前中期库取 dense top-1，再人工核对它是否重复本话题的主动写入。
  - 现行规则与 C 各跑 2 次，C 与 C2 又各跑 3 次；回放脚本是临时脚本，不在仓库中。
- **结果**（"复现"指草稿逐条重复本话题的主动写入）：

| 话题（主动写入数） | 现行规则 | C | C2 |
|:---|:---|:---|:---|
| SME 第二轮主话题（6） | 2 次都复现 6 条 | 5 次都复现 6 条 | 3 次中 2 次漏出 1 条重复 |
| SME 记忆动力学追问（1） | 2 次都复现（一次拆成 4 条） | 2 次都复现 | 未测 |
| Gliding Horse（4） | 1 次复现 4 条，1 次输出无法解析或为空 | 5 次都复现 4 条 | 不再逐条复现，但每次产出 4～10 条写入之外的内容：测试失败数、配置与代码的矛盾、"宣称与接线"的缺口清单、Windows 编译前置条件 |
| EverOS（8） | 2 次都复现 8 条 | 5 次中 1 次复现 8 条，其余没有产出 | 3 次都没有产出 |
| 混合话题（dsh 9 条、Claude Code 4 条） | 1 次复现 9 条 dsh 写入，1 次与生产一致 | 5 次都复现 9 条 dsh 写入 | 3 次中 2 次各重新提取了 1 条 Claude Code 的写入（其中用户偏好一条标题一字不差），1 次没有产出 |

C 没有压住重新推导，在混合话题上反而更稳定地复现：写入的标题此时就在材料里。C2 大体压住了逐条复现，但仍会漏，而且把 Mode A 推向"写入没有覆盖的事"，在调研话题里就是 agent 按 skill 有意没写的审计发现。

E 不依赖模型遵从提示词。按生产中实际的查重结果估算：上述话题中 21 条重复主动写入的被动草稿，有 20 条命中本话题的主动写入、会被丢弃（剩下的是 Gliding Horse 收尾门，0.683，本来就判为 DISTINCT）；第 2.5 节的 4 次合并失败中有 3 次出在这些合并上，同样不会发生；第 2.4 节的 4 次同轮合并都不会发生。

同时回放了 Mode B 收到 agent 标题的效果：dsh-agent-soul 与 EverOS 的 17 条写入，提示词加入 agent 拟定的标题与"沿用，除非不符或缺少可检索的名词"的说明后，17 条都原样沿用。

### 3.3 决定（owner，2026-11-03）

- v0.7.3 回到记忆主线，以主动写入为主体改进生成，在想到更好的记忆生成策略之前先做这一组最小优化；
- Mode A 的改动采用 E；owner 先选了 C，回放否定 C 之后改选 E；
- Mode B 沿用 agent 写的标题；
- 生成任务失败带原因码，经失败分类进入工作记录、事件与日志（owner 评估事件与日志的现状后，由"只在执行处记日志"改定）；运行时事件的持久记录、调度器降噪与日志写法规范作为版本之外的观测工作，随时可以开始，不在本文范围；
- 这三项都归入 v0.7.3。

## 4. 候选设计（分析，不是 owner 决定，可在计划评审时调整）

### 4.1 话题内主动写入优先（E）

- **记录**：Patchouli 生成控制面在进程内按话题记录"本话题主动写入结算到的记忆"：WRITE 与 UPDATE 任务的结算目标，包括新建的记忆、只触碰的已有记忆，以及被合并时保留的那条。话题结算任务读取后清除这个话题的记录。
- **Mode A**：草稿查重命中（IDENTICAL 或 SIMILAR）的已有记忆在本话题的记录中时，丢弃这条草稿：不新建、不触碰、不提名合并，只记日志。
- **Mode B**：SIMILAR 命中本话题记录中的记忆时，照常新建，但不提名合并；IDENTICAL 照常只触碰。
- **不处理**：低于 SIMILAR 下限的重复（Gliding Horse 收尾门一类）照常新建；跨话题的查重与合并不变；进程重启前没有经过 shutdown 结算的话题，记录随进程丢失，退回现行行为。
- **理由**：只用已经存在的结算结果，不改材料、提示词、合并语义与持久化结构；被动提取仍然运行，写入没有覆盖的事实照常进入记忆。

### 4.2 主动写入沿用 agent 的标题

- `GenerationEngine._process_mode_b` 把 `WriteFocus.title` 交给 `extract_write`；Mode B 的用户提示词在草稿内容之前加入"Agent 拟定的标题"，没有标题时写"未提供"，中英文同步。
- 系统提示词加入沿用规则，措辞以第 3.2 节回放所用的版本为起点：Agent 拟定了标题时沿用它，不改写措辞；只有它与草稿内容不符，或缺少让人检索到它的关键名词（项目、组件、工具名）时才修改，修改时保留原标题的判断与表述。
- MCP 写入工具的说明（`mcp_server/server.py` 的 `_WRITE_DESCRIPTION`，现为"title 简短概括"）与使用手册 skill 中"正式记忆的标题、摘要与标签由生成重新撰写"一句随之改写；skill 的 dsh 副本由 owner 重新复制。
- 不处理：摘要、标签与正文的生成方式；合并对标题的改写。
- 测试：Mode B 发给模型的消息包含 agent 的标题（提取器真实、LLM 服务替身），引擎把 `WriteFocus.title` 交给提取器；模型是否遵从由回放验收。

### 4.3 生成任务失败带原因码

生成队列的失败分类（`MemoryGenerationHandler.classify_failure`，现在对一切异常给出 `memory_generation_failed`）按异常给出稳定原因码：

| 异常 | 原因码 |
|:---|:---|
| `MemoryMergeAbortedError` | `merge_aborted.<details 中的 reason>` |
| `ModelContextWindowExceededError` | `model_context_window_exceeded` |
| `ModelProviderError` | `model_provider_failed` |
| 其他 `DomainError` | 该错误的 `code` |
| 其他异常 | `memory_generation_failed` |

Work Queue Runtime 已经把分类的原因写进工作记录并由 `work.failed` 投影，不需要改动；控制器发布 `memory.task.failed` 时，失败任务的 `reason` 取工作记录的原因码（现在只填取消原因），失败日志带标签、来源、话题、原因码与异常类名。这符合事件契约"失败 reason 使用稳定原因码"的要求；任务快照的 `error` 文本与任务 API 不变，前端只显示 `reason`、不按它分支。测试放在生成队列与控制器的现有测试中，表中每类异常一个用例。四种合并中止原因的处置不同：`version_changed` 是可重做的冲突，`model_output_invalid` 指向提示词或模型，`memory_missing` 与 `not_mergeable` 是正常的放弃；第 2.5 节第四次失败不属于前三次的模式，需要真实原因才能判断。

### 4.4 合并遇到版本冲突时重做一次（不进入 v0.7.3 的计划）

`version_changed` 时重新读取 M 与 N，以最新的 M 重做一次合并。E 之后，第 2.5 节前三次那样"同一次结算两条草稿命中同一条 M"的情况大部分不再产生合并；剩余的冲突有多少、是不是 `version_changed`，等 4.3 落地后用真实原因判断，再决定是否需要。

## 5. 搁置项

- 同一个发现写进多条主动写入：属于 agent 写法，已由 DeeperResearch 工作区的 repo-deep-dive skill 改写处理（2026-11-02，不在本仓库）。
- Mode A 在调研话题上仍然完整提取一遍，E 只在查重之后丢弃，模型调用的成本不变。
- dsh-agent-soul 的调研被落进 Claude Code 话题，见[跨会话落位 Todo](../todo/cross-conversation-topic-placement.md)；它不改变本文的结论。

## 6. 验证方式与指标

| 指标 | 现状（第 2 节） | E 的估算（第 3.2 节） |
|:---|:---|:---|
| 只含调研的话题中，Mode A 草稿重复主动写入的比例 | 19/19 | 不变（仍然提取） |
| 重复草稿最终新建或合并进主动写入的比例 | 19/19 | 1/19 |
| 被动草稿并入本话题主动写入的合并次数 | 17 次成功、3 次失败 | 0 |
| 同一轮主动写入之间的合并 | 4 个仓库中 3 个出现，共 4 次 | 0 |
| 主动写入沿用 agent 标题的比例 | 0/17 | 17/17（回放） |

各计划的验收以离线回放与日常使用中的下一批调研为准，见各计划的验收一节。

## 7. 计划

同一时间只有一份生效计划：

- 当前计划：[话题内主动写入优先](../plans/v0.7.3-topic-active-write-precedence.md)（第 4.1 节）；
- 尚未建立计划：主动写入沿用 agent 的标题（第 4.2 节）、生成任务失败带原因码（第 4.3 节），都已有 owner 决定，轮到时依据本文建立计划；
- 第 4.4 节等失败原因的数据。

## 附录：样本现状与处置建议

处置由 owner 经管理入口执行。删除只移除中期库原子，证据仍可从 Artifact 回放。本附录只记录取证时的状态，处置之后不再更新。

### 2026-11-02 取证

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

### 2026-11-03 取证

| 记忆 | 来源与经历 | 建议 |
|:---|:---|:---|
| `ref_rebuildable_vs_state` | EverOS 被动草稿，合并失败残留 | 删除：与 `ref_derived_state_separation` 重复 |
| `fact_everos_memory_scope` | 主动写入，写入时并入检索分派一条，结算时再并入两条被动草稿，4791 字符 | 保留；Scope 隔离与检索分派是两个机制，可以人工拆开，或下次调研时重写 |
| `fact_dsh_agent_soul_overview` | 主动写入，写入时并入「能力半与身份半」一条，之后被 agent 修正 | 保留 |
| dsh-agent-soul 的 9 条主动写入 | 贡献者里多了 `claude_code`（跨会话落位所致） | 保留；贡献者不影响检索与读取 |
