---
title: Ideas
status: current
owner: project
scope: uncommitted-exploration
last_reviewed: 2026-11-18
---

# Ideas

本目录保存尚未形成项目承诺的开放设想、研究假设和候选方向。Idea 可以解释一项机会为什么值得探索，也可以保留暂时无法收敛的多种路径；它不能用未来类名、配置草案或阶段编号替代当前设计，更不能因为已有少量基础设施就被解释为功能已经排期。

外部系统、论文与技术方案的独立调研需求和报告统一收录于 [Research](../research/README.md)；本目录保留 HiveMemory 的开放设计，通过链接引用调研资料。

## 分类

Idea 以 frontmatter 的 `horizon` 字段分为 `current`、`candidate` 与 `long-term` 三类，含义与要求见[文档治理规范](../DOCUMENTATION.md)第 8.4 节，本索引按类别分区。类别变化只修改该字段与本索引，不移动文件。

## 当前版本的设计讨论（`current`）

| Idea | 服务的版本 | 简述 |
|:---|:---|:---|
| [记忆核心：短期到中期的流转](./memory-flow-short-to-mid-term.md)（总纲） | v0.8 | v0.8 重新设计记忆从短期到中期的流转：决定一览、代码现状与评判方式 |
| [记忆核心：写入链路](./memory-flow-write-path.md) | v0.8 | 两条写入路径的汇合、短期记忆的定义、Patchouli 作为馆长，以及生成的记忆指向交互轮次 |
| [记忆核心：库的形状、图索引与保存形态](./memory-library-shape-and-index.md) | v0.8 | 中期库的形状、派生的时空图索引与记忆的保存形态 |
| [记忆核心：类型体系](./memory-type-system.md) | v0.8 | 记忆类型的用途、划分与回放检验 |
| [记忆核心：评测集](./memory-evaluation-benchmarks.md) | v0.8 | 用公开评测集与 agent 工作流数据检验记忆核心 |

已实现、被替代或不再跟踪的 Idea 登记在 [Archived Ideas](../archive/ideas/README.md)。

## 候选方向（`candidate`）

| Idea | 简述 |
|:---|:---|
| [统一的交互记忆：轨迹、断言与整理](./unified-interaction-memory.md) | 以同一套架构把对话与长程 agent 的交互沉淀为记忆 |
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
| [MaaT 与 Skill/Plugin 边界的资产消费模型](./memory-as-tool-skill-plugin-boundary.md) | 记忆作为工具、技能与插件被消费的边界 |
| [生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md) | 生命力评分由真实使用数据校准，以及落库后的演化质量 |
| [Agent 轨迹与多 Agent TDA](./TDA_Agent_Research_Ideas.md) | 用拓扑数据分析解释 agent 轨迹的成功、成本与失败 |
| [Memory-Centric Agent TDA](./TDA_Memory_Centric_Agent_Ideas.md) | 多视图记忆图能否形成优于平面检索的信号 |

这里的材料均不代表路线图承诺，也不能作为当前能力引用；`current` 类中记录的 owner 决定只约束后续 Plan 的编写，不等于已经实现。

## 升级规则

Idea 进入实施前至少需要：

1. 有来自真实运行、可复现实验或明确产品场景的问题证据；
2. 能说明目标、非目标、受影响的所有权与稳定契约；
3. 能用基线、指标和失败样本证明方案价值，而不只是证明可以实现；
4. 有分阶段迁移、兼容、隐私/耐久性和回滚考虑；
5. 建立独立 Plan，并列出完成后必须更新的当前文档。

在这些条件满足前，Ideas 不进入 Roadmap，也不应拆成没有独立完成语义的 Todo。
