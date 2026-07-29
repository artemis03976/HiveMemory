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
last_reviewed: 2026-07-29
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
| `docs/SETUP.md` | help | 迁移 | P2 已重建 `help/setup.md`、`configuration.md` 与 `troubleshooting.md`，并将旧入口标记 `superseded`；物理移动留给 Archive 批次 |
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
| `docs/alice/README.md` | current | 重写 | P1 已建立 Alice 职责、非职责、执行/编排分层、Profile、主流程、模块入口与真实限制 |
| `docs/alice/phases/README.md` | archive | 直接归档 | P1 已由 Alice 当前索引取代并标记 `superseded`；P2 再统一移动 |
| `docs/alice/phases/Phase1.md` | merge | 合并后归档 | P1 已将 Agent Profile、权限与单 Agent Runtime 事实并入 Alice 当前文档并标记原文 `superseded` |
| `docs/alice/phases/Phase2.md` | merge | 合并后归档 | P1 已将 CALL、PendingAtom 与 Orchestrator 事实并入 Alice/Contracts 并标记原文 `superseded` |
| `docs/agent_runtime/README.md` | merge | 退役 | P1 已由 `alice/agent-runtime.md`、`pending-atom.md` 与 `mtp-runtime.md` 取代并标记旧索引 `superseded` |
| `docs/agent_runtime/pending_atom/README.md` | merge | 退役 | P1 已由 `alice/pending-atom.md` 取代并标记旧索引 `superseded` |
| `docs/agent_runtime/pending_atom/PendingAtomCacheDesign.md` | merge | 拆分 | P1 已将当前 shadow/cache、三级解析与隔离缺口并入 Alice 当前文档并标记原文 `superseded` |
| `docs/agent_runtime/pending_atom/PendingAtomMaterializeTaskDesign.md` | merge | 合并后归档 | P1 已将 materialize/settlement 对偶与 AgentRunResult 边界并入 Alice/Patchouli 当前文档并标记原文 `superseded` |
| `docs/agent_runtime/pending_atom/PendingAtomRuntimeDesign.md` | merge | 合并后归档 | P1 已将状态命令、store 所有权、settlement 与回收行为并入 `alice/pending-atom.md` 并标记原文 `superseded` |
| `docs/agent_runtime/pending_atom/PendingAtomStatusUnificationDesign.md` | merge | 合并后归档 | P1 已将状态机、resolution 与 snapshot 不变量并入 `alice/pending-atom.md` 并标记原文 `superseded` |

## 9. Gateway

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/engines/gateway.md` | merge | 合并后归档 | 2026-07-29 逐篇审计通过后已移入 `archive/legacy-docs/engines/gateway.md`；当前入口为 `gateway/` |

## 10. Applications 与 Frontend

| 现有文档 | 分类 | 动作 | 目标或处理说明 |
|:---|:---:|:---|:---|
| `docs/applications/MealAssistantProductSpec.md` | product | 保留并核验 | 保持 `planned`；2026-07-29 复核当前链路、七种通用 MemoryType、未接线 `time_range` 与无 session/history 恢复边界，继续留在 Applications |
| `docs/frontend/FrontendDesign.md` | merge | 拆分 | 2026-07-29 逐篇审计通过后已移入 `archive/legacy-docs/frontend/FrontendDesign.md`；当前入口为 `frontend/` |
| `docs/frontend/frontend-state-persistence-research.md` | merge | 合并后归档 | 2026-07-29 逐篇审计通过后已移入 `archive/legacy-docs/frontend/frontend-state-persistence-research.md` |
| `docs/frontend/MemoryGardenUI.md` | merge | 拆分 | 2026-07-29 逐篇审计通过后已移入 `archive/legacy-docs/frontend/MemoryGardenUI.md` |

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
| `docs/mod/AgentLoopDecouplingDesign.md` | archive | 已迁移 | 理念与事实由 Alice 当前文档承接；历史稿进入 `archive/plans/implementation/agent-loop-decoupling.md` |
| `docs/mod/AgentRuntimeBoundaryDesign.md` | archive | 已迁移 | 边界裁定由 Alice 当前文档承接；历史稿进入 `archive/plans/implementation/agent-runtime-boundary.md` |
| `docs/mod/MemoryCompilerIRDesign.md` | archive | 已迁移 | IR 与 target 事实由 `patchouli/memory-compiler.md` 承接；历史稿进入 implementation archive |
| `docs/mod/MemoryCompilerRetrievalRefactorPlan.md` | archive | 已迁移 | Retrieval/Compiler 分工由两个当前文档承接；历史稿进入 implementation archive |
| `docs/mod/MemoryGenerationManagementEnhancementPlan.md` | archive | 已迁移 | spec 隔离、等待、取消和 shutdown drain 已实现；durable queue 缺口进入 Local Work Queue Plan |
| `docs/mod/PatchouliSubsystemRefactorPlan.md` | archive | 已迁移 | Patchouli 所有权与分层由当前模块文档承接；历史稿进入 implementation archive |
| `docs/mod/PendingAtomLifecycleDesign.md` | archive | 已迁移 | 状态机、settlement 与回收语义由 `alice/pending-atom.md` 承接；历史稿进入 implementation archive |
| `docs/mod/RuntimeEventPublishingRefactorDesign.md` | plan | 已迁移 | 原分类纠正：Publisher/Emitter/payload 类型化尚未落地，迁入 `plans/runtime-event-publishing-refactor.md` |
| `docs/mod/V0.4.0RuntimeControlAndObservabilityPlan.md` | archive | 已迁移 | v0.4 当前事实进入 System/Generation，历史稿进入 implementation archive |
| `docs/mod/V0.5.0DataDurabilityAndAsyncColdPathPlan.md` | archive | 已迁移 | artifact/provenance 与耐久性边界进入 Patchouli 当前文档，历史稿进入 implementation archive |
| `docs/mod/V0.5.1InfraCleanupPlan.md` | archive | 已迁移 | 配置、NoOp 与取消接线进入 System/Patchouli/Alice 当前文档，历史稿进入 implementation archive |
| `docs/mod/V0.5.2AsyncNativeAdaptationPlan.md` | archive | 已迁移 | async-native 主链已实现，历史实施记录进入 implementation archive |
| `docs/mod/V0.6.0CompositeIntentDecompositionDesign.md` | plan | 已迁移 | 迁入 `plans/v0.6.0-composite-intent-decomposition.md`，并按当前私有 `sub_intents` 事实修正 |
| `docs/mod/V0.6.0GatewaySystemDesign.md` | archive | 已迁移 | Gateway 主体事实由当前四篇文档承接；复合 intent 独立进入 Plan |
| `docs/mod/V0.6.0GlobalCommandSystemDesign.md` | archive | 已迁移 | 已实现命令语义进入 `gateway/commands.md`，未实现设想只留在历史稿 |
| `docs/mod/V0.6.0PassiveIngressDesign.md` | archive | 已迁移 | 当前事实进入 `system/passive-ingress.md` 与 Gateway workflow |
| `docs/mod/V0.6.0UserQueryAnalysisGen1TechDebt.md` | archive | 已迁移 | 当前技术债与指标先行原则进入 `gateway/analysis.md`；第二代尚不足以建立小型 Todo 或排期 Plan |
| `docs/mod/V0.6.1LocalWorkQueueRuntimePlan.md` | plan | 已迁移 | 迁入 `plans/v0.6.1-local-work-queue-runtime.md`，继续保持 Planned 状态 |

## 14. 迁移批次

为避免先移动文件、后验证内容，后续按以下批次执行：

1. [x] **P0：全局入口校正**：版本口径、`PROJECT.md`、`ROADMAP.md`、`architecture/overview.md` 与 `boundaries.md`；
2. [x] **P0：跨子系统契约**：MTP、错误模型、routes/events 和子系统边界；
3. [x] **P1：System 与 Gateway**：已消除 v0.6.0 已实现但仍被描述为未来的偏差；
4. [x] **P1：Patchouli**：MemoryLibrary、artifacts、perception、generation、retrieval、lifecycle、MemoryCompiler；
5. [x] **P1：Alice**：Agent Runtime、orchestration、PendingAtom 与 MTP runtime；
6. [x] **P2：Frontend、Applications 与 Help**；
7. **P2：Archive 重组、源码 README 收敛和全库链接检查**：清单第 4～10 节与 `docs/mod/` 已完成分批审计和物理迁移；Ideas、其他源码 README 与额外批次不在本轮范围，仍需后续独立复核。

## 15. 本步骤完成条件

- [x] 74 篇既有文档均有初始分类和目标动作；
- [x] System、Gateway、Contracts、Help、Plans、Todo、Ideas 和 ADR 已有可导航目录入口；
- [x] Archive 已建立 Plans、Legacy Architecture 和 Legacy Docs 目标分区；
- [x] Applications 与 Frontend 已有局部索引；
- [x] 已记录版本状态、v0.6.0 状态和编码损坏等阻塞性偏差；
- [ ] Archive/源码 README 批次仍有本轮范围外的 Ideas、其他源码 README 与额外材料需要最终链接和状态复核；
- [ ] 已纳入第 4～10 节和 `docs/mod/` 的旧文件均已物理迁移，但迁移清单的其他后续批次尚未全部关闭；
- [x] P0 当前设计主干和契约已经重写。
- [x] P1 System 与 Gateway 当前设计已经核验、重写并关闭旧入口。
- [x] P1 Patchouli 当前设计已经核验、重写并关闭旧入口。
- [x] P1 Alice 当前设计已经核验、重写并关闭旧入口。
- [x] P2 Frontend、Applications 与 Help 已经核验、重写并关闭旧入口。

剩余源码 README、Archive 物理处理与全库链接检查属于最后一个 P2 迁移批次。

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

随后已进入并完成 **P1：Alice**，结果见下一节。

## 19. P1 Alice 迁移结果

本批已于 2026-07-28 完成：

- `alice/` 已建立子系统总览，以及 Agent Runtime、多 Agent 编排、PendingAtom 与 MTP Runtime 四篇当前模块文档；
- 文档保留“人偶使”、人偶图纸、CPU/调度器分层、store buffer 与文本协议等设计隐喻和边界理由，同时逐项核对 frame、loop、Profile、Koakuma、resolver、PendingAtom、syscall、配置和测试表达的当前事实；
- 当前文档明确记录 Profile fail-open、进程级共享 cache、Identity 隔离缺口、共享 scheduler/cancel 状态、子帧结果收割偏差、非持久化运行状态、配置未接线和 RUN 非强安全沙箱等设计张力；
- Alice Phase 1/2、平行 Agent Runtime 索引、PendingAtom 系列设计稿、执行层边界/解耦稿与 v0.5.1 基础设施清理稿均已标记 `superseded` 并链接当前入口；
- MTP 契约保留调用方 Identity 不变量，并明确 L0/L1 命中尚未重新校验身份是当前实现偏差，而不是新的契约口径；
- 旧文件仍保留原路径，统一物理移动、源码 README 收敛和全库链接检查继续留给 P2。

随后已完成 **P2：Frontend、Applications 与 Help**，结果见下一节；Archive 物理重组仍未在本批提前执行。

## 20. P2 Frontend、Applications 与 Help 迁移结果

本批已于 2026-07-28 完成：

- `frontend/` 已建立应用壳与视觉、Chat 工作区、管理页面、状态/持久化/传输四篇当前文档，保留透明、沉浸、可观测、机器态/人类态分离、日月与五行水晶等设计语义；
- 当前前端事实已对照 React/Vite 源码、stores、services、FastAPI routers 与部署配置核验，明确记录 Terminal 空入口、Topic 非会话切换、chat 不恢复、Memory 伪 semantic search、mock fallback、Agent CALL 权限遗漏和 Settings 配置结构偏差；
- 三份旧前端设计/调研稿已标记 `superseded`，其中未实现的 Pin、真正语义检索、可拖拽布局和统计大屏没有被写成当前事实；
- `applications/MealAssistantProductSpec.md` 已标记 `planned`，保留产品动机、Agent Profile、system prompt 与验收流程，同时明确当前没有独立应用实现或真实用户证据；
- `help/` 已形成安装、配置和排障指南，区分开发 `5173/8769` 与 Docker `8000`，校正 Provider/Model 配置、health/readiness 和 mock/Settings 风险；旧 `SETUP.md` 已停止维护；
- 本批没有物理移动旧文件。Archive 重组、源码 README 收敛和最终全库链接检查继续作为下一批 P2 工作。

## 21. 第 4～6 节逐篇复核与物理迁移结果

本批于 2026-07-29 完成，范围严格限定为清单第 4～6 节。完整逐文档结论、承接位置、拒绝项和最终路径见[迁移审计记录](../archive/plans/documentation-migration-audit-sections-4-6.md)。

- 顶层治理：`DOCUMENTATION.md` 增加逐篇迁移门禁；PROJECT/VISION/ROADMAP 完成背景、当前边界和文档入口复核；
- 数据模型：旧混合文档拆为 `architecture/data-model.md`、ADR-0001 与未排期治理 Plan，并保留 MemoryAtom 语义事务与冰山结构理念；
- 顶层旧入口：SETUP、TODO、Observability 已分别在 Help、两个 Todo、System/Contracts 中充分承接后移入 `archive/legacy-docs/`；
- Architecture：evolution redirect 已移入 `archive/legacy-architecture/evolution-index.md`，已归档 v2/v3/v4 与 Router 材料再次复核并修正替代链接；
- Contracts/System：MTP 的语法取舍、串行控制、READ 批量语义和 Agent 行动门槛，错误 formatter 边界，scheduler 历史原因，以及 observability 分组来由已补回当前文档；
- i18n：补齐 Relay/Generation 文本域、自然语言与机器契约分界、分域 catalog 取舍及文案行为风险；旧 i18n 树已移入 Archive；
- 编码损坏的 `MemoryCompilerI18nMigrationPlan.md` 采用原字节移动，没有从损坏正文复制当前事实；
- 第 7 节及以后文件未在本批移动，仍需按后续批次逐篇审计。

## 22. 第 7 节 Patchouli 与 Engines 逐篇复核与物理迁移结果

本批于 2026-07-29 完成，范围严格限定为清单第 7 节的十一篇文档。完整逐篇结论、承接位置、拒绝继承项和最终路径见[迁移审计记录](../archive/plans/documentation-migration-audit-section-7.md)。

- `patchouli/README.md` 及七篇当前模块文档已再次对照代码、配置、测试与契约复核；补齐主动 WRITE/UPDATE 不触发 settlement 的原因、被动 `MessageTurnBuffer` 的结构化事件边界、`target_topic` 在 user 到达时绑定，以及 MemoryCompiler unit/envelope 的责任分界；
- `ActiveMemoryGenerationDecouplingDesign.md`、`PatchouliPassiveIngestRefactorPlan.md`、`PatchouliTranscriptDualViewRefactor.md` 已确认其设计理念和真实限制分别进入 Generation、System Passive Ingress、Perception 与 Artifacts 当前入口；
- `docs/engines/` 的 README、Perception、Generation、Retrieval、Lifecycle 和 MemoryCompiler 两篇旧文档已确认不再拥有独立真相源，当前事实分别并入 Patchouli 模块文档；
- 明确拒绝固定容量/置信度公式、检索 miss 自动 revive、完整多格式 renderer、`read_memory` 稳定懒加载与 MTP RUN 已可执行等历史承诺；`engines/perception.md` 的 U+FFFD 损坏正文按原样保留，仅修正迁移后的替代入口链接，未从损坏段落复制事实；
- 审计通过后，十篇 Patchouli/Engines 旧文档已物理移动至 `archive/legacy-docs/patchouli/` 与 `archive/legacy-docs/engines/`，并修复归档后的当前文档相对链接；
- 用户追加范围中的四篇 `src/hivememory/engines/{perception,generation,retrieval,lifecycle}/README.md` 已逐篇复核并移入 `archive/legacy-docs/source-readmes/engines/`；Generation 补回 LLM 提取的取舍，Lifecycle 补回 caller-scoped refresh/GC 编排理由和逐用户反馈状态缺口；
- 本批仍不处理第 8 节及以后文档、`docs/mod/` 或其他源码 README；剩余源码 README 收敛与最终全库链接检查属于后续 Archive 批次。

## 23. 第 8 节 Alice 与 Agent Runtime 逐篇复核与物理迁移结果

本批于 2026-07-29 完成，范围严格限定为清单第 8 节的十篇文档。完整逐篇结论、承接位置、拒绝继承项和最终路径见[迁移审计记录](../archive/plans/documentation-migration-audit-section-8.md)。

- `alice/README.md` 与 Agent Runtime、多 Agent 编排、PendingAtom、MTP Runtime 四篇当前文档已再次对照代码、测试与契约复核，并统一更新审计日期；
- Phase 1 中“万物皆记忆”、persona/permissions 分离、Prompt/Runtime 双层权限、多角色历史与 Omni-Doll fallback 的有效理念已经确认进入 Alice 当前入口；Patchouli Kernel 全职责、话题原地切换 Agent 和默认 GLOBAL 写入等旧口径明确退出；
- Phase 2 的 CALL trap、Profile 发现、显式 `context_refs`、瞬态子帧、黑盒隔离、单层星型拓扑与两类 alias/task 收割已被充分承接；当前编排文档补回隐式 RETURN 的设计理由，同时保留对子帧退出状态检查不足这一实现缺口；
- PendingAtom 系列旧稿中的 store buffer、handle/intent、三级解析、Task/Settlement 对偶、状态与 resolution 正交等理念已被承接；当前文档进一步明确 `PendingAtom.status/settlement` 是业务真相，alias/intent/canonical 映射只做反查索引；
- 明确拒绝 durable ledger/TTL/事件重放已经存在、平行 `_resolution/_redirects` 映射作为真相、`ChatResult` 旧三字段、Patchouli 反向读取可变 PendingAtom，以及递归 CALL/强沙箱/无限自组织网络等历史或未来口径；
- 审计通过后，Alice Phase 三篇与 Agent Runtime/PendingAtom 六篇旧文档已物理移动至 `archive/legacy-docs/alice/phases/` 与 `archive/legacy-docs/agent_runtime/`，并修复当前入口、源码、`docs/mod/` 和其他历史文档的相对链接；
- 本批不处理第 9 节 Gateway、`docs/mod/` 的物理归档、其他源码 README 或最终全库迁移门禁，仍由后续 Archive 批次继续完成。

## 24. 第 9～10 节 Gateway、Applications 与 Frontend 逐篇复核与物理迁移结果

本批于 2026-07-29 完成，范围严格限定为清单第 9～10 节的五篇文档。完整逐篇结论、承接位置、拒绝继承项和最终路径见[迁移审计记录](../archive/plans/documentation-migration-audit-sections-9-10.md)。

- Gateway 当前四篇文档已对照固定 workflow、公共 `GatewayDecision`、Patchouli 消费边界和测试复核；补回“Compute Once, Use Everywhere”的有效动机，并把它收敛为同一入口消息的一份冻结、受限分析投影，而不是下游共同真相；
- 明确拒绝 Gateway 作为中枢神经、单一 `GatewayResult` 拥有检索/感知/生成、`worth_saving` 是最终写入决定、所有 CHAT 默认 RAG、被动 buffer 与 renderer 由 Gateway 所有，以及动态/持久化 workflow 已存在等旧口径；
- Frontend 当前五篇文档已复核视觉隐喻、Chat 结构化事件、管理页面、store 与 transport；应用壳补回“动效必须由真实状态事件驱动”，状态文档补回 localStorage 作为 `name/version/migrate/partialize` 本地契约的治理原则；
- `MealAssistantProductSpec.md` 继续保持 `planned` 和原路径；最小闭环改为 Gateway -> Patchouli prepare/retrieval -> Alice run/MTP -> Patchouli finalize，并澄清七种类型是产品映射、`time_range` 尚未端到端接线、“新会话”只是新的独立 chat run 验收语义；
- 审计通过后，Gateway 一篇与 Frontend 三篇旧稿已物理移动至 `archive/legacy-docs/engines/` 和 `archive/legacy-docs/frontend/`，并修复当前入口与源码相对链接；
- 本批 20 篇相关 Markdown 的严格 UTF-8 与相对链接检查通过，Gateway/System/Patchouli 定向测试 48 项通过，前端 lint 与生产构建通过；
- 本批没有处理 `docs/mod/`、Ideas、其他源码 README 或最后的全库迁移门禁，后续仍需按清单继续推进。

## 25. `docs/mod` 逐篇复核与物理迁移结果

本批于 2026-07-29 完成。完整逐篇承接、理念、拒绝项、分类纠正与最终路径见 [`docs/mod` 迁移审计](../archive/plans/documentation-migration-audit-docs-mod.md)。

- Alice 三篇边界/生命周期稿已确认由 Agent Runtime、Orchestration 与 PendingAtom 当前文档充分承接，并进入 `archive/plans/implementation/`；
- Patchouli/MemoryCompiler/v0.5.x 七篇稿已核对 IR、renderer 分离、spec 并发与失败隔离、wait/shutdown drain、NoOp、AsyncQdrantClient 和 async-native；尚存的 durable queue、backpressure、原子 provenance 等缺口没有被误写成已完成；
- v0.4、Gateway、Commands、Passive Ingress 与 Query Analysis 五篇稿已由 System/Gateway 当前文档承接后归档；Query Analysis 第二代因范围跨模块且尚无排期，没有机械创建成小型 Todo；
- `RuntimeEventPublishingRefactorDesign.md` 的原分类经代码复核后由 archive 纠正为 Plan：统一 Publisher、领域 emitter 与 payload 类型化尚未实现；
- Composite Intent Plan 已按当前代码重写基线：只有 `COMPOSITE` 与 Engine 私有 `sub_intents`，没有公共 branch/envelope/merge 能力；
- Local Work Queue 继续保持 Planned，明确一套 runtime、多 lane、Scheduler 非 Queue 和单机优先的边界；
- 最终十五篇完成/被替代稿进入 `archive/plans/implementation/`，三篇当前计划进入 `docs/plans/`，原 `docs/mod/` 清空；
- Roadmap、Project、Plans/Archive/System/Gateway 索引、源码注释、历史文档互链和迁移清单均已改指最终路径。
