---
title: Ideas
status: current
owner: project
scope: uncommitted-exploration
last_reviewed: 2026-08-19
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
| [生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md) | 当前 vitality 公式、强化事件、gardening 与显式 archive/revive | 哪些新状态能由真实使用数据校准，而不是继续叠加启发式参数 |
| [AE2 与 HiveMemory 的架构同构性](./ae2-hivememory-architecture-analogy.md) | Patchouli 存储平面、Alice Frame/编排、MTP 能力契约与当前可见性过滤 | Workspace/软件子网、Harness-to-Harness、Workflow Memory、Job Graph、Mount/Bridge、AIOS 资源抽象与隔离执行是否值得进入真实验证 |
| [Workspace MVP 与 Chat Attachments 初步设计](./workspace-mvp-chat-attachments-design.md) | `v0.6.2 W0/W1` 依赖链、System-owned WorkspaceAssetStore、SemanticBuffer binding、两级状态机与 Artifact provenance | W0 已进入[正式 Plan](../plans/v0.6.2-workspace-mvp.md)；本文继续保存设计推导和 W1 上传解析、Context Compiler、Materialization promotion 等开放问题 |
| [Chat Run 生命周期后续候选](./chat-run-lifecycle-follow-ups.md) | 已完成的取消最小闭环、SSE 与 run registry | 哪些候选具有独立收益，是否值得分别立项，而不是实施一次性大重构 |
| [复合意图分解](./composite-intent-decomposition.md) | `COMPOSITE` 分类信号与私有 `sub_intents` | 真实样本能否证明单主意图路径存在稳定缺口，以及 envelope、消费所有权与 fallback 如何冻结 |

本索引已于 2026-08-19 对照当前代码与文档分类规范复核。这里的材料均保留为 `idea`：没有一篇已经形成近期排期、依赖闭包和可验收实施范围，也没有一篇可以作为当前能力引用。既有 Ideas 的逐篇分类依据见[文档迁移最终收口审计](../archive/plans/documentation-migration-finalization-audit.md)。

## 升级规则

Idea 进入实施前至少需要：

1. 有来自真实运行、可复现实验或明确产品场景的问题证据；
2. 能说明目标、非目标、受影响的所有权与稳定契约；
3. 能用基线、指标和失败样本证明方案价值，而不只是证明可以实现；
4. 有分阶段迁移、兼容、隐私/耐久性和回滚考虑；
5. 建立独立 Plan，并列出完成后必须更新的当前文档。

在这些条件满足前，Ideas 不进入 Roadmap，也不应拆成没有独立完成语义的 Todo。
