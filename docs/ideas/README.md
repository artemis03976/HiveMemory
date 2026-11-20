---
title: Ideas
status: current
owner: project
scope: unimplemented-design-rules-and-index
last_reviewed: 2026-11-20
---

# Ideas

本目录保存尚未实施的设计：当前版本方向的设计、尚未排期的候选方向，以及长期假设。Idea 不代表项目承诺；它不能用未来类名、配置草案或阶段编号替代当前设计，更不能因为已有少量基础设施就被解释为功能已经排期。写法与分类见文末的“规则”，本索引按 `horizon` 分区。

外部系统、论文与技术方案的独立调研需求和报告统一收录于 [Research](../research/README.md)；本目录保留 HiveMemory 的设计，通过链接引用调研资料。

## 当前版本的设计（`current`）

| Idea | 服务的版本 | 简述 |
|:---|:---|:---|
| [记忆核心：短期到中期的流转](./memory-flow-short-to-mid-term.md)（总纲） | v0.8 | v0.8 重新设计记忆从短期到中期的流转：目标与原则、共用的代码现状、顺序与评判方式 |
| [记忆核心：写入链路](./memory-flow-write-path.md) | v0.8 | 两条写入路径的汇合、短期记忆的定义、Patchouli 作为馆长，以及生成的记忆指向交互轮次 |
| [记忆核心：库的形状、图索引与保存形态](./memory-library-shape-and-index.md) | v0.8 | 中期库的形状、派生的时空图索引与记忆的保存形态 |
| [记忆核心：类型体系](./memory-type-system.md) | v0.8 | 记忆类型的用途、划分与回放检验 |
| [记忆核心：评测集](./memory-evaluation-benchmarks.md) | v0.8 | 用公开评测集与 agent 工作流数据检验记忆核心 |

已实现、被替代或不再跟踪的 Idea 登记在 [Archived Ideas](../archive/ideas/README.md)。

## 候选方向（`candidate`）

| Idea | 简述 |
|:---|:---|
| [全链路原始资料保存](./PatchouliPageFoldingRawEvidenceDesign.md) | 从会话到话题再到 Artifact 的原始资料保存 |
| [短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md) | Alice 会话压缩与话题折叠的视图与摘要质量 |
| [身份与访问体系](./identity-and-access-model.md) | 身份、认证与授权的单独主线 |
| [长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md) | 长 turn 内的原子事件流、分段、封口与恢复 |
| [孵化场、试验场与垃圾回收场：低成本地定型、验证与搁置想法](./experiment-field.md) | 孵化场定型模糊的想法，试验场在冻结的真实数据上验证机制，垃圾回收场冷藏说不清的想法与失败的方案 |
| [轻量化：模型依赖、运行内存与包体](./lightweight-footprint.md) | 减少模型依赖、常驻内存与安装包体积 |

## 长期方向与概念（`long-term`）

| Idea | 简述 |
|:---|:---|
| [外部 Actor 的接入登记与运行时访问](./external-actor-registration-and-runtime-access.md) | 外部 harness 与 agent 的接入登记和运行时访问 |
| [执行环境](./execution-environment.md) | 执行任务的 CPU 与执行环境 |
| [外部会话消息的接收与 Topic 投影](./external-session-and-topic-projection.md) | 外部会话的输入、交接与话题投影 |
| [任务进程表与任务请求唯一注册入口](./task-process-table-and-registration-entry.md) | 任务进程表、注册入口与四阶段骨架的演进 |
| [第五代架构：Workspace 网络](./fifth-generation-architecture.md) | 第五代架构的核心理念及其检验问题 |
| [AE2 与 HiveMemory 的架构同构性](./ae2-hivememory-architecture-analogy.md) | 以 AE2 的子网、Pattern 与合成 CPU 类比 HiveMemory 的架构方向 |
| [生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md) | 生命力评分由真实使用数据校准，以及落库后的演化质量 |
| [Agent 轨迹与多 Agent TDA](./TDA_Agent_Research_Ideas.md) | 用拓扑数据分析解释 agent 轨迹的成功、成本与失败 |
| [Memory-Centric Agent TDA](./TDA_Memory_Centric_Agent_Ideas.md) | 多视图记忆图能否形成优于平面检索的信号 |

这里的材料均不代表路线图承诺，也不能作为当前能力引用；`current` 类中写下的 owner 决定只约束后续 Plan 的编写，不等于已经实现。

## 规则

### Idea 写什么

Idea 的读者是写计划、判断设计是否成立的人。它写设计现在是什么，不写设计怎样形成：

- `current` 的 Idea 按设计组织：目标与原则；组成与数据流；每一个组成部分写设计、现状与差距（以代码为准）、开放问题；最后是非目标。
- `candidate` 与 `long-term` 的 Idea 写到它实际的成熟度：观察到的问题与证据、假设与潜在价值、可能的路径、主要未知项与验证方式，以及升级为 Plan 的条件。

设计陈述写的是要实现的设计，不宣称已经实现，也不用确定语气宣称功能一定会实现；已经实现的部分以当前事实文档为准，Idea 链接过去，不再复述。

### 决定、候选与开放问题

- owner 的决定写成设计陈述，放在它约束的那一部分：不带日期，不写“取代某项决定”，被取代的内容直接改写；不设决定一览、变更记录或讨论经过，历史由版本控制保留。
- 正文默认是已定的设计；候选设计显式标“候选”，可以给出合理的默认值，在计划评审时调整。
- 开放问题只列阻塞下一份计划的，写背景、选项与影响；一个主题只设一个问题，不拆成子问题。owner 决定后问题即关闭，结论改写进对应部分的设计，回答问题不应让未决问题变多；不阻塞的发现只记一行搁置项。

### 不写进 Idea 的

- 讨论过程、各轮解读、试过又放弃的做法与回放过程：写进 `.hivememory/fields/` 的工作笔记，它不提交，也不受文档治理，写法见[孵化场、试验场与垃圾回收场](./experiment-field.md)。其他 Idea 不链接这些笔记，也不链接或依据那份 Idea。
- 外部系统与论文的事实：放进 [Research](../research/README.md)，Idea 只链接。
- 阶段划分、交付、验收与文档更新清单：属于 [Plan](../plans/README.md)。

### 分类

Idea 以 frontmatter 的 `horizon` 字段分为三类，本索引按此分区：

| `horizon` | 含义 | 额外要求 |
|:---|:---|:---|
| `current` | 当前版本方向的设计，直接为下一份 Plan 供料 | 以 `serves_version` 写明服务的版本，并在 [Plans](../plans/README.md) 对应版本下有入口；版本收尾时要么已形成 Plan，要么改为 `candidate` 或 `long-term` |
| `candidate` | 问题具体、尚未排期的功能或技术方向 | 进入某个版本的方向后改为 `current` |
| `long-term` | 长期方向、研究假设或概念性指导 | 不绑定版本 |

类别变化只修改字段与本索引，不移动文件。

### 退回与归档

- 尚未开始实施的计划，如果方向仍然成立但前提已经改变，退回 Idea：移入 `ideas/`，删除计划安排（阶段划分、交付与验收、跨计划依赖、文档更新清单），保留设计，把原来的“决定”“冻结”改写为候选设计或开放问题，入链随之更新。作废且不再讨论的计划直接删除（[文档治理规范](../DOCUMENTATION.md)第 10 节）。
- 已经实现、被替代或不再跟踪的 Idea 整体移入 `archive/ideas/`（[Archive](../archive/README.md)）。

### 升级为 Plan

Idea 进入实施前至少需要：

1. 有来自真实运行、可复现实验或明确产品场景的问题证据；
2. 能说明目标、非目标、受影响的所有权与稳定契约；
3. 能用基线、指标和失败样本证明方案价值，而不只是证明可以实现；
4. 有分阶段迁移、兼容、隐私/耐久性和回滚考虑；
5. 建立独立 Plan，并列出完成后必须更新的当前文档。

在这些条件满足前，Ideas 不进入 Roadmap，也不应拆成没有独立完成语义的 Todo。
