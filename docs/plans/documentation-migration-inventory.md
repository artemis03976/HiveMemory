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
| `docs/ObservabilityDesign.md` | merge | 合并后归档 | 当前事实进入 `docs/system/observability.md`，并与 RuntimeEvent 相关设计统一 |

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
| `docs/protocols/PatchouliUnifiedMaintenanceSchedulerDesign.md` | merge | 合并后归档 | 当前调度器进入 `system/runtime-and-bus.md`；Patchouli 注册职责进入其 runtime/lifecycle 文档 |
| `docs/protocols/i18n/README.md` | merge | 退役 | 由 `system/i18n.md` 和相关当前文档取代 |
| `docs/protocols/i18n/I18nFoundationDesign.md` | merge | 合并后归档 | 已落地基础设施进入 `system/i18n.md`；未完成范围不得继续保留为当前事实 |
| `docs/protocols/i18n/I18nStatusAndRoadmap.md` | merge | 拆分 | 当前状态进入 `system/i18n.md`；剩余工作进入 Plan 或 Todo |
| `docs/protocols/i18n/KoakumaMTPBackfillTextI18nInventory.md` | merge | 合并后归档 | 结构化回填与语言边界进入 MTP 契约和 `system/i18n.md` |
| `docs/protocols/i18n/MemoryCompilerI18nMigrationPlan.md` | archive | 直接归档 | 主要阶段已完成且正文存在编码损坏；剩余工作以 Status 文档核对后另建 Todo/Plan |

## 7. Patchouli 与 Engines

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/patchouli/README.md` | current | 重写 | 建立 Patchouli 当前职责、非职责、模块索引和代码入口 |
| `docs/patchouli/ActiveMemoryGenerationDecouplingDesign.md` | merge | 合并后归档 | 当前生成协调边界进入 `patchouli/generation.md`，跨系统部分链接 Contracts |
| `docs/patchouli/PatchouliPassiveIngestRefactorPlan.md` | merge | 合并后归档 | 有效摄入语义并入 `system/passive-ingress.md`；被新 v0.6 设计替代的部分作为历史保留 |
| `docs/patchouli/PatchouliTranscriptDualViewRefactor.md` | merge | 合并后归档 | 当前双视图与 artifact 事实进入 `patchouli/artifacts.md` 或架构数据模型文档 |
| `docs/engines/README.md` | merge | 退役 | 各引擎归入 Patchouli 或 Gateway 当前文档，不再保留平行 Engines 文档树 |
| `docs/engines/perception.md` | merge | 合并后归档 | 根据当前代码重建 `patchouli/perception.md`；原文存在编码损坏，禁止直接搬运 |
| `docs/engines/generation.md` | merge | 合并后归档 | 当前事实进入 `patchouli/generation.md` |
| `docs/engines/retrieval.md` | merge | 合并后归档 | 根据当前 retriever/fusion/reranker 与 MemoryCompiler 边界重建 `patchouli/retrieval.md` |
| `docs/engines/lifecycle.md` | merge | 合并后归档 | 根据 MemoryLibrary 与当前 lifecycle engine 重建 `patchouli/lifecycle.md` |
| `docs/engines/memory_compiler/README.md` | merge | 退役 | 由 `patchouli/memory-compiler.md` 取代 |
| `docs/engines/memory_compiler/Phase1RenderConvergenceDesign.md` | merge | 合并后归档 | 已实现表达收敛事实进入 `patchouli/memory-compiler.md`；未来 RUN 内容保留在独立 Plan/Idea |

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
| `docs/engines/gateway.md` | merge | 合并后归档 | 按当前独立 Gateway 子系统重建 `gateway/analysis.md`、`commands.md` 与 `workflow.md`；旧 Patchouli/TheEye 边界不再保留为当前事实 |

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
| `docs/archive/mod/EnableLifecycleMaintenanceDesign.md` | archive | 保留并核验 | 补充归档元数据及指向未来 `patchouli/lifecycle.md` 的替代链接 |

## 13. docs/mod 迁移

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/mod/AgentLoopDecouplingDesign.md` | merge | 合并后归档 | 当前 loop/executor 边界进入 `alice/agent-runtime.md` |
| `docs/mod/AgentRuntimeBoundaryDesign.md` | merge | 合并后归档 | 当前 Alice/Agent/MTP 边界进入 Alice 与 Contracts；关键裁定评估为 ADR |
| `docs/mod/MemoryCompilerIRDesign.md` | merge | 合并后归档 | 当前 IR 与 handler/target 事实进入 `patchouli/memory-compiler.md` |
| `docs/mod/MemoryCompilerRetrievalRefactorPlan.md` | merge | 合并后归档 | 当前 Retrieval/Compiler 分工分别进入两个 Patchouli 模块文档 |
| `docs/mod/MemoryGenerationManagementEnhancementPlan.md` | merge | 合并后归档 | 当前任务控制与协调语义进入 `patchouli/generation.md` 和 System runtime 文档 |
| `docs/mod/PatchouliSubsystemRefactorPlan.md` | merge | 合并后归档 | 当前 MemoryLibrary、service/runtime 和引擎所有权进入 Patchouli 当前文档 |
| `docs/mod/PendingAtomLifecycleDesign.md` | merge | 合并后归档 | 当前生命周期和句柄回收进入 `alice/pending-atom.md` |
| `docs/mod/RuntimeEventPublishingRefactorDesign.md` | merge | 合并后归档 | 当前发布抽象进入 `system/observability.md` 与 `contracts/routes-and-events.md`；重要取舍评估为 ADR |
| `docs/mod/V0.4.0RuntimeControlAndObservabilityPlan.md` | merge | 合并后归档 | v0.4 已完成；当前控制和事件事实进入 System 文档 |
| `docs/mod/V0.5.0DataDurabilityAndAsyncColdPathPlan.md` | merge | 合并后归档 | v0.5 已完成；当前 artifact/provenance/cold path 事实进入 Patchouli 与 System 文档 |
| `docs/mod/V0.5.1InfraCleanupPlan.md` | merge | 合并后归档 | 提取当前配置所有权和 NoOp/cancel 不变量后归档 |
| `docs/mod/V0.5.2AsyncNativeAdaptationPlan.md` | archive | 直接归档 | 实施记录保留历史；仍有效的 async 不变量并入当前模块文档 |
| `docs/mod/V0.6.0CompositeIntentDecompositionDesign.md` | plan | 迁移 | 移入 `plans/v0.6.0-composite-intent-decomposition.md`；与当前私有 `sub_intents` 行为明确区分 |
| `docs/mod/V0.6.0GatewaySystemDesign.md` | merge | 拆分 | Phase 3A-3F 等已实现事实进入 Gateway/System 当前文档；未完成阶段进入 Plan；原设计归档 |
| `docs/mod/V0.6.0GlobalCommandSystemDesign.md` | merge | 拆分 | 已实现命令注册、分发和短路进入 `gateway/commands.md`；剩余内容进入 Plan/Todo |
| `docs/mod/V0.6.0PassiveIngressDesign.md` | merge | 合并后归档 | 当前实现进入 `system/passive-ingress.md`，并链接 Gateway ingress 契约 |
| `docs/mod/V0.6.0UserQueryAnalysisGen1TechDebt.md` | todo | 拆分 | 第一代当前行为进入 `gateway/analysis.md`；具体技术债迁入 `todo/` 或 Issue |
| `docs/mod/V0.6.1LocalWorkQueueRuntimePlan.md` | plan | 迁移 | 移入 `plans/v0.6.1-local-work-queue-runtime.md`，继续保持 Future/Planned 状态 |

## 14. 迁移批次

为避免先移动文件、后验证内容，后续按以下批次执行：

1. [x] **P0：全局入口校正**：版本口径、`PROJECT.md`、`ROADMAP.md`、`architecture/overview.md` 与 `boundaries.md`；
2. [x] **P0：跨子系统契约**：MTP、错误模型、routes/events 和子系统边界；
3. **P1：System 与 Gateway**：优先消除 v0.6.0 已实现但仍被描述为未来的偏差；
4. **P1：Patchouli**：MemoryLibrary、artifacts、perception、generation、retrieval、lifecycle、MemoryCompiler；
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
- [ ] P1/P2 旧文件尚未移动或归档；
- [x] P0 当前设计主干和契约已经重写。

剩余事实核验与旧文件处理属于 P1/P2 迁移批次。

## 16. P0 迁移结果

P0 已于 2026-07-28 完成：

- `PROJECT.md` 已收敛为当前项目总览与全局文档索引；
- `ROADMAP.md` 已区分最新发布标签 `v0.5.0` 与未发布开发基线 `v0.6.0`；
- 当前架构由 `architecture/overview.md` 与 `architecture/boundaries.md` 统一维护；
- 跨子系统契约由 `contracts/subsystem-contracts.md`、`routes-and-events.md`、`mtp.md` 和 `error-model.md` 统一维护；
- v2/v3/v4 与 Router 收口材料已移入 `archive/legacy-architecture/`；
- 旧 MTP 与错误设计已标记 `superseded`，当前链接改指 Contracts；
- 根中英文 README 的版本与高层架构入口已经校正。

下一批从 **P1：System 与 Gateway** 开始。
