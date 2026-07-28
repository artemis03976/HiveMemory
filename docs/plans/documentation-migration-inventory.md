---
title: Documentation Migration Inventory
status: active
owner: project
scope: docs-migration
updates:
  - docs/PROJECT.md
  - docs/ROADMAP.md
  - docs/architecture/
  - docs/system/
  - docs/patchouli/
  - docs/alice/
  - docs/gateway/
  - docs/contracts/
last_reviewed: 2026-07-28
---

# 文档体系迁移清单

## 1. 目的与边界

本清单覆盖第二步开始前 `docs/` 中的 74 篇 Markdown 文档，用于确定其目标分类、迁移动作和主要去向。

本轮判断依据包括：

- 文档标题、自述状态和内容形态；
- 当前 `src/hivememory/` 的目录和主要实现入口；
- 当前测试所覆盖的系统行为；
- `ROADMAP.md` 对版本阶段的声明。

这是一份迁移工作清单，不是对每篇文档内容正确性的最终认证。标记为 `merge` 的文档只表示其中存在值得提取的当前事实；事实进入 `current` 文档前仍需逐项对照代码、测试和配置。

## 2. 分类与动作

### 2.1 目标分类

| 分类 | 含义 |
|:---|:---|
| `current` | 保留为当前规范入口，但需要核验或重写 |
| `merge` | 提取已实现事实到当前文档，原文随后归档或退役 |
| `plan` | 尚未完全落地的完整功能或重构计划 |
| `idea` | 未形成承诺的开放探索 |
| `todo` | 小范围缺陷或技术债 |
| `help` | 安装、配置、使用或排障指南 |
| `product` | 应用规格、用户场景或验证材料 |
| `archive` | 已经明确只具备历史价值 |

### 2.2 迁移动作

| 动作 | 含义 |
|:---|:---|
| 保留并核验 | 位置和职责基本正确，补充元数据并根据实现校验 |
| 重写 | 保留入口角色，但正文需要按当前事实重新组织 |
| 迁移 | 内容类型明确，移动到目标目录并修复链接 |
| 拆分 | 一篇文档混合多个状态，分别进入 Current、Plan、Todo、ADR 或 Archive |
| 合并后归档 | 提取有效事实到唯一当前文档，随后冻结原文 |
| 直接归档 | 不再作为当前依据，仅补充归档元数据和替代入口 |
| 退役 | 内容已被新索引完全替代，确认无独立历史价值后删除或归档 |

## 3. 已发现的全局偏差

1. `pyproject.toml` 仍声明 `0.1.0-beta`，`src/hivememory/__init__.py` 声明 `0.6.0`，根 README 仍以 v0.1.0 为口径，版本信息尚未统一。
2. `ROADMAP.md` 将 v0.6.0 标记为“下一阶段”，但 GatewaySystem、GatewayRuntime、Commands、Passive ingress 及其测试已经存在。v0.6.0 设计文档必须按已实现与未实现部分拆分。
3. `docs/engines/`、源码目录 README、`PROJECT.md` 和多个重构计划并行描述同一模块，且文件结构和能力状态不同。
4. `DataModelImmutabilityStatusAndRoadmap.md`、`I18nStatusAndRoadmap.md` 等文档混合当前状态与未来治理计划。
5. `docs/engines/perception.md` 和 `docs/protocols/i18n/MemoryCompilerI18nMigrationPlan.md` 存在明显字符编码损坏，迁移时不能直接复制正文。
6. 多篇标记为 `Draft` 的设计已经有对应实现，多篇标记为“当前”的文档又包含已被删除的类或文件。自述状态不能作为最终分类依据。

## 4. 顶层入口与治理文档

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/DOCUMENTATION.md` | current | 保留并核验 | 文档治理规范；后续结构变化需同步更新 |
| `docs/PROJECT.md` | merge | 重写 | 收敛为当前项目总览和全局索引；详细架构、模块与历史内容分别下沉 |
| `docs/VISION.md` | current | 保留并核验 | 保留长期愿景；继续维持事实、假设与远期愿景的明确分层 |
| `docs/ROADMAP.md` | merge | 重写 | 根据代码重新校正阶段状态；详细设计链接到 Plans，完成记录链接到 Archive |
| `docs/SETUP.md` | help | 迁移 | 核验命令、端口和配置后迁入 `docs/help/setup.md` |
| `docs/TODO.md` | todo | 拆分 | 未完成小项迁入 `docs/todo/` 或 Issue；已完成记录归档；实现示例不继续保留在待办索引 |
| `docs/ObservabilityDesign.md` | merge | 合并后归档 | P1 已合并到 `docs/system/observability.md` 并标记原文 `superseded`；P2 再统一移动 |

## 5. Architecture

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/architecture/README.md` | current | 重写 | 成为当前架构索引，只链接当前架构、边界、数据模型和 ADR |
| `docs/architecture/DataModelImmutabilityStatusAndRoadmap.md` | merge | 拆分 | 当前约束进入 `architecture/data-model.md`；治理阶段进入 Plan；关键边界裁定评估为 ADR |
| `docs/architecture/evolution/README.md` | archive | 迁移 | 迁入 `archive/legacy-architecture/` 并改为历史索引 |
| `docs/architecture/evolution/SystemArchitecture_v2.0.md` | archive | 直接归档 | 历史架构，不再作为当前入口 |
| `docs/architecture/evolution/SystemArchitecture_v3.0.md` | archive | 直接归档 | 历史草案，不再作为当前入口 |
| `docs/architecture/evolution/SystemArchitecture_v4.0.md` | merge | 合并后归档 | 核验后提取当前 System/Patchouli/Alice 边界到 `architecture/overview.md` 与 `boundaries.md` |
| `docs/architecture/evolution/SystemArchitecture_v4_RouterToApplicationService_Refactor.md` | merge | 合并后归档 | 已实现结构进入 `system/application-services.md`；重要分层理由评估为 ADR |

## 6. System、Contracts 与 i18n

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/protocols/README.md` | merge | 退役 | 由 `docs/contracts/README.md` 取代；迁移完成后移除旧索引 |
| `docs/protocols/MemoryToolProtocol.md` | merge | 重写 | 根据 parser、models、runtime、formatter 和测试重建 `docs/contracts/mtp.md` |
| `docs/protocols/MTPErrorStructureDesign.md` | merge | 合并后归档 | 已实现错误与 warning 语义进入 `contracts/error-model.md` 和 `contracts/mtp.md` |
| `docs/protocols/PatchouliUnifiedMaintenanceSchedulerDesign.md` | merge | 合并后归档 | P1 已将调度器所有权并入 `system/runtime-and-bus.md`，idle flush/gardening 业务并入 Patchouli Perception/Lifecycle，并标记原文 `superseded` |
| `docs/protocols/i18n/README.md` | merge | 退役 | P1 已由 `system/i18n.md` 取代并标记旧索引 `superseded` |
| `docs/protocols/i18n/I18nFoundationDesign.md` | merge | 合并后归档 | P1 已将落地基础设施与限制并入 `system/i18n.md`，原文已标记 `superseded` |
| `docs/protocols/i18n/I18nStatusAndRoadmap.md` | merge | 拆分 | P1 已将当前状态与已知缺口并入 `system/i18n.md`，原文已标记 `superseded` |
| `docs/protocols/i18n/KoakumaMTPBackfillTextI18nInventory.md` | merge | 合并后归档 | P1 已由 MTP/Error Contracts 与 `system/i18n.md` 取代，原文已标记 `superseded` |
| `docs/protocols/i18n/MemoryCompilerI18nMigrationPlan.md` | archive | 直接归档 | 主要阶段已完成且正文存在编码损坏；剩余工作以 Status 文档核对后另建 Todo/Plan |

## 7. Patchouli 与 Engines

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/patchouli/README.md` | current | 重写 | P1 已建立 Patchouli 职责、非职责、运行分层、流程、模块索引、代码入口与真实限制 |
| `docs/patchouli/ActiveMemoryGenerationDecouplingDesign.md` | merge | 合并后归档 | P1 已并入 `patchouli/generation.md` 并标记原文 `superseded` |
| `docs/patchouli/PatchouliPassiveIngestRefactorPlan.md` | merge | 合并后归档 | P1 已由 `system/passive-ingress.md` 与 `patchouli/perception.md` 取代，并标记原文 `superseded` |
| `docs/patchouli/PatchouliTranscriptDualViewRefactor.md` | merge | 合并后归档 | P1 已将结构化事实、Generation 视图与 artifact 边界并入三个当前模块文档，并标记原文 `superseded` |
| `docs/engines/README.md` | merge | 退役 | P1 已由 Patchouli/Gateway 当前索引取代并标记 `superseded`；代码目录继续保留 |
| `docs/engines/perception.md` | merge | 合并后归档 | P1 已根据代码重建 `patchouli/perception.md` 并标记编码损坏原文 `superseded` |
| `docs/engines/generation.md` | merge | 合并后归档 | P1 已并入 `patchouli/generation.md` 与 `artifacts.md` 并标记原文 `superseded` |
| `docs/engines/retrieval.md` | merge | 合并后归档 | P1 已按当前 retriever/fusion/reranker 与 Compiler 边界重建并标记原文 `superseded` |
| `docs/engines/lifecycle.md` | merge | 合并后归档 | P1 已按 MemoryLibrary 与 Lifecycle 当前实现重建并标记原文 `superseded` |
| `docs/engines/memory_compiler/README.md` | merge | 退役 | P1 已由 `patchouli/memory-compiler.md` 取代并标记旧索引 `superseded` |
| `docs/engines/memory_compiler/Phase1RenderConvergenceDesign.md` | merge | 合并后归档 | P1 已将当前 IR/target/预算事实并入 Compiler 文档并标记原文 `superseded`；RUN 仍明确未实现 |

## 8. Alice 与 Agent Runtime

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/alice/README.md` | current | 重写 | 建立 Alice 当前职责、runtime 组成、模块索引和契约入口 |
| `docs/alice/phases/README.md` | archive | 直接归档 | Phase 索引完成历史职责后退役 |
| `docs/alice/phases/Phase1.md` | merge | 合并后归档 | 已实现 Agent Profile/Runtime 事实进入 Alice 当前文档 |
| `docs/alice/phases/Phase2.md` | merge | 合并后归档 | 已实现 CALL/PendingAtom/Orchestrator 事实进入 Alice 与 Contracts 当前文档 |
| `docs/agent_runtime/README.md` | merge | 退役 | Agent Runtime 归入 `alice/agent-runtime.md`，不再作为平级子系统目录 |
| `docs/agent_runtime/pending_atom/README.md` | merge | 退役 | 由 `alice/pending-atom.md` 取代 |
| `docs/agent_runtime/pending_atom/PendingAtomCacheDesign.md` | merge | 拆分 | 核验当前 shadow/cache 行为后并入 `alice/pending-atom.md`；未落地设想转 Plan/Idea |
| `docs/agent_runtime/pending_atom/PendingAtomMaterializeTaskDesign.md` | merge | 合并后归档 | 当前 materialize 与 AgentRunResult 边界进入 Alice/Patchouli 契约说明 |
| `docs/agent_runtime/pending_atom/PendingAtomRuntimeDesign.md` | merge | 合并后归档 | 当前状态应用、settlement 和 alias 行为进入 `alice/pending-atom.md` |
| `docs/agent_runtime/pending_atom/PendingAtomStatusUnificationDesign.md` | merge | 合并后归档 | 当前状态机和不变量进入 `alice/pending-atom.md` 与数据模型文档 |

## 9. Gateway

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/engines/gateway.md` | merge | 合并后归档 | P1 已重建 `gateway/analysis.md`、`commands.md` 与 `workflow.md` 并标记原文 `superseded`；P2 再统一移动 |

## 10. Applications 与 Frontend

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/applications/MealAssistantProductSpec.md` | product | 保留并核验 | 补充元数据；核对其对 v0.6 和现有能力的引用，不作为后端当前设计入口 |
| `docs/frontend/FrontendDesign.md` | merge | 拆分 | 已实现 UI 结构进入当前前端文档，尚未实现内容进入 Plan/Idea |
| `docs/frontend/frontend-state-persistence-research.md` | merge | 合并后归档 | 提炼已接受的状态所有权与持久化决策到当前前端文档或 ADR，保留调研为历史证据 |
| `docs/frontend/MemoryGardenUI.md` | merge | 拆分 | 根据当前页面核对；已实现交互并入当前文档，未实现设计进入 Plan/Idea，原稿归档 |

## 11. Ideas

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md` | idea | 保留并核验 | 保持非承诺性质，补充升级为 Plan 的条件 |
| `docs/ideas/TDA_Agent_Research_Ideas.md` | idea | 保留并核验 | 保持研究设想，不进入当前 Alice 能力描述 |
| `docs/ideas/TDA_Memory_Centric_Agent_Ideas.md` | idea | 保留并核验 | 保持研究设想，不进入当前 Patchouli 能力描述 |
| `docs/ideas/VitalityScoringLongTermEvolutionIdeas.md` | idea | 保留并核验 | 与当前 lifecycle 文档分离，只保留长期演进方向 |

## 12. 已归档文档

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/archive/README.md` | archive | 重写 | 成为统一 Archive 索引，解释 plans、legacy-architecture、legacy-docs 的边界 |
| `docs/archive/mod/README.md` | archive | 迁移 | 由 `archive/plans/README.md` 取代后退役 |
| `docs/archive/mod/EnableLifecycleMaintenanceDesign.md` | archive | 保留并核验 | P1 已补充归档元数据并链接 `patchouli/lifecycle.md` 与 System scheduler 当前入口 |

## 13. docs/mod 迁移

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/mod/AgentLoopDecouplingDesign.md` | merge | 合并后归档 | 当前 loop/executor 边界进入 `alice/agent-runtime.md` |
| `docs/mod/AgentRuntimeBoundaryDesign.md` | merge | 合并后归档 | 当前 Alice/Agent/MTP 边界进入 Alice 与 Contracts；关键裁定评估为 ADR |
| `docs/mod/MemoryCompilerIRDesign.md` | merge | 合并后归档 | P1 已并入 `patchouli/memory-compiler.md` 并标记原文 `superseded` |
| `docs/mod/MemoryCompilerRetrievalRefactorPlan.md` | merge | 合并后归档 | P1 已将 Retrieval/Compiler 分工并入两个当前文档并标记原文 `superseded` |
| `docs/mod/MemoryGenerationManagementEnhancementPlan.md` | merge | 合并后归档 | P1 已将任务控制、失败隔离、等待/取消和 shutdown 时序并入 `patchouli/generation.md` 并标记原文 `superseded` |
| `docs/mod/PatchouliSubsystemRefactorPlan.md` | merge | 合并后归档 | P1 已将 MemoryLibrary、service/runtime、Familiar/Coordinator 和引擎所有权并入 Patchouli 当前文档并标记原文 `superseded` |
| `docs/mod/PendingAtomLifecycleDesign.md` | merge | 合并后归档 | 当前生命周期和句柄回收进入 `alice/pending-atom.md` |
| `docs/mod/RuntimeEventPublishingRefactorDesign.md` | merge | 合并后归档 | 当前发布抽象进入 `system/observability.md` 与 `contracts/routes-and-events.md`；重要取舍评估为 ADR |
| `docs/mod/V0.4.0RuntimeControlAndObservabilityPlan.md` | merge | 合并后归档 | v0.4 已完成；当前控制和事件事实进入 System 文档 |
| `docs/mod/V0.5.0DataDurabilityAndAsyncColdPathPlan.md` | merge | 合并后归档 | P1 已将 artifact/provenance/cold path 事实与耐久性缺口并入 Patchouli 当前文档并标记原文 `superseded` |
| `docs/mod/V0.5.1InfraCleanupPlan.md` | merge | 合并后归档 | 提取当前配置所有权和 NoOp/cancel 不变量后归档 |
| `docs/mod/V0.5.2AsyncNativeAdaptationPlan.md` | archive | 直接归档 | P1 已将有效 async 边界并入 MemoryLibrary/Generation/Retrieval 并标记实施记录 `superseded`；P2 再物理移动 |
| `docs/mod/V0.6.0CompositeIntentDecompositionDesign.md` | plan | 迁移 | 移入 `plans/v0.6.0-composite-intent-decomposition.md`；与当前私有 `sub_intents` 行为明确区分 |
| `docs/mod/V0.6.0GatewaySystemDesign.md` | merge | 拆分 | P1 已将已实现事实并入 Gateway/System 当前文档并标记原文 `superseded`；未落地内容不再作为当前能力 |
| `docs/mod/V0.6.0GlobalCommandSystemDesign.md` | merge | 拆分 | P1 已将注册、分发、安全和短路语义并入 `gateway/commands.md`，原文已标记 `superseded` |
| `docs/mod/V0.6.0PassiveIngressDesign.md` | merge | 合并后归档 | P1 已并入 `system/passive-ingress.md` 与 Gateway workflow，原文已标记 `superseded` |
| `docs/mod/V0.6.0UserQueryAnalysisGen1TechDebt.md` | todo | 拆分 | P1 已将当前行为、矛盾和技术债并入 `gateway/analysis.md`；原稿停止作为当前设计入口 |
| `docs/mod/V0.6.1LocalWorkQueueRuntimePlan.md` | plan | 迁移 | 移入 `plans/v0.6.1-local-work-queue-runtime.md`，继续保持 Future/Planned 状态 |

## 14. 迁移批次

为避免先移动文件、后验证内容，后续按以下批次执行：

1. [x] **P0：全局入口校正**：版本口径、`PROJECT.md`、`ROADMAP.md`、`architecture/overview.md` 与 `boundaries.md`；
2. [x] **P0：跨子系统契约**：MTP、错误模型、routes/events 和子系统边界；
3. [x] **P1：System 与 Gateway**：已消除 v0.6.0 已实现但仍被描述为未来的偏差；
4. [x] **P1：Patchouli**：MemoryLibrary、artifacts、perception、generation、retrieval、lifecycle、MemoryCompiler；
5. **P1：Alice**：Agent Runtime、orchestration、PendingAtom 与 MTP runtime；
6. **P2：Frontend、Applications 与 Help**；
7. **P2：Archive 重组、源码 README 收敛和全库链接检查**。

## 15. 本步骤完成条件

- [x] 74 篇既有文档均有初始分类和目标动作；
- [x] System、Gateway、Contracts、Help、Plans、Todo、Ideas 和 ADR 已有可导航目录入口；
- [x] Archive 已建立 Plans、Legacy Architecture 和 Legacy Docs 目标分区；
- [x] Applications 与 Frontend 已有局部索引；
- [x] 已记录版本状态、v0.6.0 状态和编码损坏等阻塞性偏差；
- [ ] 每篇 `merge` 文档的事实尚未逐条通过代码验证；
- [ ] P1/P2 旧文件尚未全部物理移动至 Archive；
- [x] P0 当前设计主干和契约已经重写。
- [x] P1 System 与 Gateway 当前设计已经核验、重写并关闭旧入口。
- [x] P1 Patchouli 当前设计已经核验、重写并关闭旧入口。

剩余 Alice 事实核验与旧文件物理处理属于 P1/P2 迁移批次。

## 16. P0 迁移结果

P0 已于 2026-07-28 完成：

- `PROJECT.md` 已收敛为当前项目总览与全局文档索引；
- `ROADMAP.md` 已区分最新发布标签 `v0.5.0` 与未发布开发基线 `v0.6.0`；
- 当前架构由 `architecture/overview.md` 与 `architecture/boundaries.md` 统一维护；
- 跨子系统契约由 `contracts/subsystem-contracts.md`、`routes-and-events.md`、`mtp.md` 和 `error-model.md` 统一维护；
- v2/v3/v4 与 Router 收口材料已移入 `archive/legacy-architecture/`；
- 旧 MTP 与错误设计已标记 `superseded`，当前链接改指 Contracts；
- 根中英文 README 的版本与高层架构入口已经校正。

## 17. P1 System 与 Gateway 迁移结果

本批已于 2026-07-28 完成：

- `system/` 已建立组合根、应用服务、Passive Ingress、runtime/bus、配置、可观测性与 i18n 当前文档；
- `gateway/` 已建立子系统总览、固定 workflow、话题/查询分析和全局命令当前文档；
- 文档同时保留设计问题、所有权理由、失败边界、技术债与矛盾检查，没有把未落地计划写成当前事实；
- Gateway Engine、v0.6.0 Gateway/Command/Passive、Observability 与主要 i18n 旧文档已标记 `superseded` 并链接替代入口；
- 旧文件的物理移动继续留给 P2 Archive 重组，避免在事实核验批次中同时大规模改路径；
- `MemoryCompilerI18nMigrationPlan.md` 因原文件存在已知编码损坏仍只保留迁移清单记录，待 P2 以原始字节安全归档，不从其正文复制当前事实。

本批之后进入 **P1：Patchouli**；本节不提前记录其结果，完成情况见下一节。

## 18. P1 Patchouli 迁移结果

本批已于 2026-07-28 完成：

- `patchouli/` 已建立子系统总览，以及 MemoryLibrary、Artifacts、Perception、Generation、Retrieval、Lifecycle 与 MemoryCompiler 七篇当前模块文档；
- 文档保留“大图书馆”、记忆资产、工作视图、热/冷路径、原始证据和保守演化等核心设计理念，同时逐项核对 Runtime、Familiar、Coordinator、Engine、Store、配置与测试所表达的当前事实；
- 当前文档明确记录 token overflow 丢失 raw blocks、artifact 非强一致、任务非持久化、archive/revive 非事务、Retrieval filters/keywords 缺口与 Compiler 预算不一致等设计张力，没有把已有模型或配置字段等同于已完成能力；
- Patchouli 原设计、平行 Engines 文档、MemoryCompiler 计划、子系统重构、v0.5 cold path/async 记录和统一维护调度稿均已标记 `superseded` 并链接当前入口；
- `EnableLifecycleMaintenanceDesign.md` 已补齐 Archive 元数据与替代入口；
- 旧文件仍保留原路径，统一物理移动、源码 README 收敛和全库链接检查继续留给 P2。

下一批进入 **P1：Alice**；本批不提前重建 Alice 文档。
