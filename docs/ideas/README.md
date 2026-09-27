---
title: Ideas
status: current
owner: project
scope: uncommitted-exploration
last_reviewed: 2026-09-27
---

# Ideas

本目录保存尚未形成项目承诺的开放设想、研究假设和候选方向。Idea 可以解释一项机会为什么值得探索，也可以保留暂时无法收敛的多种路径；它不能用未来类名、配置草案或阶段编号替代当前设计，更不能因为已有少量基础设施就被解释为功能已经排期。

## 当前 Ideas

| Idea | 当前已经具备的基础 | 仍需验证的核心问题 |
|:---|:---|:---|
| [长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md) | TurnEvent、LogicalBlock、Agent runtime、topic Page Folding 与 passive event ingress | 如何在一个 turn 内多次 compact，同时保持执行连续性、记忆生成语义、原始证据和跨入口契约 |
| [Page Folding Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md) | `state_summary` 折叠、InteractionArtifact 与异步 Generation | 保存原始折叠页是否值得引入新的耐久性、隐私与去重成本 |
| [Agent 轨迹与多 Agent TDA](./TDA_Agent_Research_Ideas.md) | TurnEvent、AgentAction、RuntimeEvent 与单层 CALL | 拓扑特征能否比普通计数/规则更稳定地解释成功、成本和失败类型 |
| [Memory-Centric Agent TDA](./TDA_Memory_Centric_Agent_Ideas.md) | MemoryAtom 关系预留、检索信号与来源/版本 Artifact | 多视图记忆图能否形成可重复、可行动且优于平面检索的信号 |
| [全项目时间使用统一规范](./project-wide-time-semantics-standardization.md) | A2-P 已统一 Memory 域时间（UTC-aware、四字段职责、局部 now 注入）；附录 D 记录的域外调用清单 | monotonic 域统一、topic 域字段分类、耗时统计与展示 allowlist 的治理路径与 owner |
| [生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md) | 当前 vitality 公式、强化事件、gardening 与显式 archive/revive | 哪些新状态能由真实使用数据校准，而不是继续叠加启发式参数 |
| [AE2 与 HiveMemory 的架构同构性](./ae2-hivememory-architecture-analogy.md) | Patchouli 存储平面、Alice Frame/编排、MTP 能力契约与当前可见性过滤 | Workspace/软件子网、Harness-to-Harness、Workflow Memory、Job Graph、Mount/Bridge、AIOS 资源抽象与隔离执行是否值得进入真实验证 |
| [Workspace 网络与任务进程架构](./workspace-network-task-process-architecture.md) | Passive buffer 与统一交互提交队列、A1 访问准入；已按第二部分决定完成的包分层（components / config / system 顶点） | 第一部分：Workspace 网络的全局拓扑，以及 Passive 退化为零主动交互的纯输入之后，被动交互的 Topic 落位、价值信号与接收范围（Q-11–Q-13）、迁移问题（M-1–M-7），以及第 7.1 节收录的原边界宪章库外状态候选裁定（缓存、resolver、事件协作）；第二部分：system 包的边界（D-1–D-9 已决定并实施；passive 的接入认证 D-8a、chat 编排的最终归属 D-9、engines 的既有向上导入、core 的内容整理与全局拓扑一致性待决）；第三部分：注册入口两阶段认证、能力层操作授权与资源边界授权的衔接（管理员操作已决定采用不建进程的直接通道；P-1–P-10 待决） |
| [任务进程表与任务请求唯一注册入口](./task-process-table-and-registration-entry.md) | Chat run 注册表、Alice PendingAtomRuntime、记忆库内部队列与维护调度 | 当前唯一的有效计划方向：进程何时关闭、中间产物可见范围、唯一入口的职责、Gateway 位置、触发型请求方、外部 CPU 的进程、对话连续性、CALL 与主动进程的交互记录去向（Q-1–Q-10、Q-14，自总 Idea 拆出） |
| [外部会话消息的接收与 Topic 投影](./external-session-and-topic-projection.md) | `InteractionPayload`、`TurnEvent`、`InteractionApplyRecord` 与 Passive turn buffer | 原 A3 计划退回：Topic 体系如何接收外部 Actor 的会话消息；`ConversationSession` 是任务进程 Idea Q-9 选项 B 的一种形态，模型字段与 Topic 生命周期事项待定 |
| [写入意图（PendingAtom）体系的迁移](./pending-intent-migration.md) | Alice `PendingAtomRuntime`、`PendingAtomMaterializeTask` 与主动生成路径 | 原 A4 计划退回：写入意图的权威持有者、共同引用读取与结算跟随；Workspace 级共享 registry 是任务进程 Idea Q-2 选项 C 的一种形态 |
| [外部 Actor 的接入登记与运行时访问](./external-actor-registration-and-runtime-access.md) | 两类接入登记（配置装载）、统一认证网关（无生产调用方）、ingest HTTP 入口与能力层 | 原计划 B 退回：外部 Actor 如何登记进系统、运行时如何访问系统，以及 adapter 的接口边界（E-1–E-4） |
| [Workspace MVP 与 Chat Attachments 初步设计](./workspace-mvp-chat-attachments-design.md) | 已完成 W0 公共契约，以及 W1 Chat Attachments 的设计推导 | W0 当前事实见 [Workspace 架构](../architecture/workspace.md)，实施历史见[归档 Plan](../archive/plans/v0.6.2-workspace-mvp.md)；本文继续保存 W1 上传解析、Context Compiler、Materialization promotion 等开放问题，不表示 W1 已排期 |
| [Chat Run 生命周期后续候选](./chat-run-lifecycle-follow-ups.md) | 已完成的取消最小闭环、SSE 与 run registry | 哪些候选具有独立收益，是否值得分别立项，而不是实施一次性大重构 |
| [复合意图分解](./composite-intent-decomposition.md) | `COMPOSITE` 分类信号与私有 `sub_intents` | 真实样本能否证明单主意图路径存在稳定缺口，以及 envelope、消费所有权与 fallback 如何冻结 |
| [MaaT 与 Skill/Plugin 边界的资产消费模型](./memory-as-tool-skill-plugin-boundary.md) | `MemoryCompiler` target 体系、MTP `RUN` 两层分发、MemoryAtom/Artifact 分离 | 如何让 MTP RUN 编译为结构化执行意图，证明相对外部 harness + Patchouli 的结构性优势，以及 CODE_SNIPPET 的迁移边界、ToolInvocationIR 与共享 MemoryCompiler IR 的关系、执行证据回流与终端实践的晋升机制 |

本索引已于 2026-09-07 对照当前代码与文档分类规范复核；2026-09-27 新增的四篇中，三篇由尚未开始实施的 v0.7.0 计划（A3、A4、计划 B）退回，一篇自总 Idea 拆出。这里的材料均保留为 `idea`：除 W0 已完成的历史实施稿外，没有一篇已经形成近期排期、依赖闭包和可验收实施范围，也没有一篇可以作为当前能力引用。Workspace Idea 中已完成部分只通过当前架构和归档 Plan 引用，不把 W1 开放设计提升为路线图承诺。既有 Ideas 的逐篇分类依据见[文档迁移最终收口审计](../archive/plans/documentation-migration-finalization-audit.md)。

## 升级规则

Idea 进入实施前至少需要：

1. 有来自真实运行、可复现实验或明确产品场景的问题证据；
2. 能说明目标、非目标、受影响的所有权与稳定契约；
3. 能用基线、指标和失败样本证明方案价值，而不只是证明可以实现；
4. 有分阶段迁移、兼容、隐私/耐久性和回滚考虑；
5. 建立独立 Plan，并列出完成后必须更新的当前文档。

在这些条件满足前，Ideas 不进入 Roadmap，也不应拆成没有独立完成语义的 Todo。
