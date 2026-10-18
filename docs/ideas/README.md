---
title: Ideas
status: current
owner: project
scope: uncommitted-exploration
last_reviewed: 2026-10-17
---

# Ideas

本目录保存尚未形成项目承诺的开放设想、研究假设和候选方向。Idea 可以解释一项机会为什么值得探索，也可以保留暂时无法收敛的多种路径；它不能用未来类名、配置草案或阶段编号替代当前设计，更不能因为已有少量基础设施就被解释为功能已经排期。

## 分类

Idea 以 frontmatter 的 `horizon` 字段分为三类，规则见[文档治理规范](../DOCUMENTATION.md)第 8.4 节。类别变化只修改该字段与本索引，不移动文件。

| horizon | 含义 |
|:---|:---|
| `current` | 当前版本方向的设计讨论，直接为下一份 Plan 供料；以 `serves_version` 写明服务的版本，可以记录 owner 已作出的决定 |
| `candidate` | 问题具体、尚未排期的功能或技术方向 |
| `long-term` | 长期方向、研究假设或概念性指导，不绑定版本 |

## 当前版本的设计讨论（`current`）

当前没有为下一份 Plan 供料的 Idea；Alice 自动压缩已经完成，相关剩余方向与调研快照转为 candidate。版本内状态见 [Plans 导航](../plans/README.md#v070)。

## 候选方向（`candidate`）

| Idea | 当前已经具备的基础 | 仍需验证的核心问题 |
|:---|:---|:---|
| [短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md) | Alice 自动压缩已完成；Topic page folding 现有实现 | Topic folding 优化、真实模型摘要质量与成本评估；两边独立维护视图 |
| [Agent Harness 会话压缩技术调研](./agent-context-compaction-research.md) | 2026-10-16 固定源码与资料快照，已为 Alice 归档计划提供依据 | 作为未排期的压缩扩展参考，不代表外部项目现状 |
| [外部会话消息的接收与 Topic 投影](./external-session-and-topic-projection.md) | 会话与 Alice 迁移已于 2026-10-15 完成，当前事实见[归档计划](../archive/plans/v0.7.0-conversation-session.md)和 Workspace 架构 | 外部输入、流式交接、路由结果、会话持久化与 Gateway 会话提示；Alice compact 已迁入独立 Idea |
| [任务进程表与任务请求唯一注册入口](./task-process-table-and-registration-entry.md) | 注册入口、进程表、四阶段骨架、CPU 分配与 CPU 端口已落地，当前事实见 [System 应用服务](../system/application-services.md)第 3、4 节 | 交互被接纳后即关闭进程与 Profile 解析时点（随话题是否预建决定）、进程记录与工作集的剩余槽位、四阶段骨架的失效条件、被动请求；Q-3b、Q-5a、Q-5b、Q-7、Q-8 |
| [执行单元、执行线程与执行环境](./execution-unit-thread-and-environment.md) | 执行线程各自的线程 context、操作请求与单一操作入口、CALL 子线程以被调用方身份运行已落地，当前事实见 [Workspace 架构](../architecture/workspace.md)第 4.3、5.4 节 | agent 与执行环境分别选择（前提 9、10）；跨 Workspace 目标的表达（T-5b）、组合的有效性（T-7）、选择的承载与系统层级控制（T-8）、CPU 输出与会话历史的词汇归属（T-10） |
| [身份与访问体系](./identity-and-access-model.md) | 身份模型、两阶段认证与两阶段授权、密封的访问 context、资源归属与发起者的拆分已落地，当前事实见 [Workspace 架构](../architecture/workspace.md)第 3、4 节 | 受限穿透访问的授予模型、资源授权的扩展约束、访问 context 可能增加的内容；WorkspaceAsset 与 `/ingest` 两项已登记例外 |
| [外部 Actor 的接入登记与运行时访问](./external-actor-registration-and-runtime-access.md) | 两类接入登记（配置装载）、统一认证网关（HTTP 与进程注册已接入；外部 harness 尚未接入）、ingest HTTP 入口与能力层 | 原计划 B 退回：外部 Actor 分为 plugin 与 controller 两种接入模式（controller 模式为 v0.7.1 的首个真实接入，plugin 模式在其后的 v0.7.x 完善）；外部 Actor 如何登记进系统、运行时如何访问系统，以及操作适配器的接口边界（E-1 已决定维持配置装载；E-3a 已决定以归一化定义操作适配器；E-5 已决定外部执行单元不识别子线程、所有 actor 共享主线程的 context；E-6a 已决定外部 harness 是执行环境、与 agent 分别选择；E-6b 已决定不允许 CALL 的目标位于另一个执行单元；E-2、E-3b、E-4 待决） |
| [写入意图（PendingAtom）的后续方向](./pending-intent-migration.md) | workspace 唯一登记与统一引用解析，actor 操作内实时派发（生成队列接纳后 ACK），结算事件订阅；当前事实见 [Workspace 架构](../architecture/workspace.md)第 5.4 节与 [Patchouli 生成](../patchouli/generation.md) | 结算对账与结果未知、句柄寿命与过期、持久化、意图可见性与 policy、UPDATE 的基础版本、外部 Actor 的提交与回读契约、完整的生成材料 |
| [Import Bus：只向网络输入的记忆来源](./import-bus.md) | 现有 Passive Ingress 链路（`/ingest`、turn buffer、交互提交队列）；owner 前提：只向网络输入、两种接收方式、不在核心拓扑上 | Topic 落位（Q-11）、价值信号（Q-12）、完整交互记录的接收范围（Q-13）、接入认证与目标 Workspace（D-8a），以及与 plugin 模式对话回流的关系 |
| [长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md) | TurnEvent、封口交互、运行时与被动事件接入 | 原子事件流、长 turn 分段、封口、跨入口与恢复；CPU 检查点与 Patchouli 摄入水位分别推进，通用压缩与证据保存见各自 Idea |
| [全链路原始资料保存](./PatchouliPageFoldingRawEvidenceDesign.md) | ConversationSession、Topic 资料投影、InteractionArtifact 与 Generation | 贯通 Session → Topic 完整资料 → 来源 Artifact → Memory 引用；保留、交接、删除与未来编辑能力，会话持久化另行计划 |
| [Chat Run 生命周期后续候选](./chat-run-lifecycle-follow-ups.md) | 已完成的取消最小闭环、SSE 与 run registry | 哪些候选具有独立收益，是否值得分别立项，而不是实施一次性大重构 |
| [复合意图分解](./composite-intent-decomposition.md) | `COMPOSITE` 分类信号与私有 `sub_intents` | 真实样本能否证明单主意图路径存在稳定缺口，以及 envelope、消费所有权与 fallback 如何冻结 |
| [全项目时间使用统一规范](./project-wide-time-semantics-standardization.md) | A2-P 已统一 Memory 域时间（UTC-aware、四字段职责、局部 now 注入）；附录 D 记录的域外调用清单 | monotonic 域统一、topic 域字段分类、耗时统计与展示 allowlist 的治理路径与 owner |

## 长期方向与概念（`long-term`）

| Idea | 当前已经具备的基础 | 仍需验证的核心问题 |
|:---|:---|:---|
| [第五代架构：Workspace 网络](./fifth-generation-architecture.md) | 第五代的核心理念 Idea，有效期到需要下一代架构为止；Workspace 网络的拓扑、能力层、任务进程、访问模型、库外设施与包分层已落地，当前事实见 [Workspace 架构](../architecture/workspace.md) | 十条核心理念（唯一能力提供者、一切任务都是进程、执行者可以替换、库内与库外、两阶段认证与授权、身份数据形态、调用与事件、显式写入即时提交、上下文与记忆材料分离、包分层）及其检验问题；演进方向分属各子 Idea；架构级开放问题 P-1、P-3、P-4a、P-5、P-7–P-10a、core 整理与负缓存 |
| [AE2 与 HiveMemory 的架构同构性](./ae2-hivememory-architecture-analogy.md) | Patchouli 存储平面、Alice Frame/编排、MTP 能力契约与当前可见性过滤 | Workspace/软件子网、Harness-to-Harness、Workflow Memory、Job Graph、Mount/Bridge、AIOS 资源抽象与隔离执行是否值得进入真实验证 |
| [MaaT 与 Skill/Plugin 边界的资产消费模型](./memory-as-tool-skill-plugin-boundary.md) | `MemoryCompiler` target 体系、MTP `RUN` 两层分发、MemoryAtom/Artifact 分离 | 如何让 MTP RUN 编译为结构化执行意图，证明相对外部 harness + Patchouli 的结构性优势，以及 CODE_SNIPPET 的迁移边界、ToolInvocationIR 与共享 MemoryCompiler IR 的关系、执行证据回流与终端实践的晋升机制 |
| [生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md) | 当前 vitality 公式、强化事件、gardening 与显式 archive/revive | 哪些新状态能由真实使用数据校准，而不是继续叠加启发式参数 |
| [Agent 轨迹与多 Agent TDA](./TDA_Agent_Research_Ideas.md) | TurnEvent、AgentAction、RuntimeEvent 与单层 CALL | 拓扑特征能否比普通计数/规则更稳定地解释成功、成本和失败类型 |
| [Memory-Centric Agent TDA](./TDA_Memory_Centric_Agent_Ideas.md) | MemoryAtom 关系预留、检索信号与来源/版本 Artifact | 多视图记忆图能否形成可重复、可行动且优于平面检索的信号 |

本索引于 2026-09-27 按 `horizon` 重新分区。这里的材料均不代表路线图承诺，也不能作为当前能力引用；`current` 类中记录的 owner 决定只约束后续 Plan 的编写，不等于已经实现。已实现或不再跟踪的 Idea 移入 [Archived Ideas](../archive/ideas/README.md)，例如已随 v0.6.2 实现的 Chat Attachments 初步设计备忘。既有 Ideas 的历史分类依据见[文档迁移最终收口审计](../archive/plans/documentation-migration-finalization-audit.md)。

## 升级规则

Idea 进入实施前至少需要：

1. 有来自真实运行、可复现实验或明确产品场景的问题证据；
2. 能说明目标、非目标、受影响的所有权与稳定契约；
3. 能用基线、指标和失败样本证明方案价值，而不只是证明可以实现；
4. 有分阶段迁移、兼容、隐私/耐久性和回滚考虑；
5. 建立独立 Plan，并列出完成后必须更新的当前文档。

在这些条件满足前，Ideas 不进入 Roadmap，也不应拆成没有独立完成语义的 Todo。
