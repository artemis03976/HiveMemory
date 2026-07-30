---
title: Legacy Documents
status: current
owner: project
scope: superseded-documentation
last_reviewed: 2026-07-30
---

# Legacy Documents

本目录保存不属于已完成计划或历史架构、但仍需保留追溯价值的旧文档。进入本目录前必须指出替代文档；无法指出替代入口的内容不得仅为清理目录而归档。

## 已归档材料

### 顶层旧入口

- [环境搭建指南](./SETUP.md)：由 [Help](../../help/README.md)取代；
- [顶层 TODO](./TODO.md)：已拆分到 [Todo](../../todo/README.md)；
- [可观测性与日志流设计](./ObservabilityDesign.md)：由 [System 可观测性](../../system/observability.md)与[事件契约](../../contracts/routes-and-events.md)取代。

### 已完成 Todo

- [运行时配置接线漂移](./runtime-configuration-wiring-drift.md)：两个无消费者的占位字段已经从后端、默认 YAML 与前端配置面统一移除；当前配置事实由 [System 配置](../../system/configuration.md)、[Retrieval](../../patchouli/retrieval.md)和[Lifecycle](../../patchouli/lifecycle.md)承接。

### Architecture 混合文档

- [不可变数据模型现状与规划](./architecture/DataModelImmutabilityStatusAndRoadmap.md)：已拆分为[当前数据模型](../../architecture/data-model.md)、[ADR-0001](../../architecture/decisions/0001-data-model-mutability-and-boundary-projection.md)与[治理计划](../../plans/data-model-mutability-governance.md)。

### Protocols 与 i18n

- [旧 Protocols 索引](./protocols/README.md)：由 [Contracts](../../contracts/README.md)取代；
- [旧 MTP 规范](./protocols/MemoryToolProtocol.md)与[错误结构设计](./protocols/MTPErrorStructureDesign.md)：由当前 MTP / Error Contracts 取代；
- [统一维护调度设计](./protocols/PatchouliUnifiedMaintenanceSchedulerDesign.md)：由 [System runtime](../../system/runtime-and-bus.md)及 Patchouli Perception/Lifecycle 取代；
- [旧 i18n 索引](./protocols/i18n/README.md)：由[全局 i18n](../../system/i18n.md)取代。`MemoryCompilerI18nMigrationPlan.md` 保留原始损坏字节，替代入口和处置结论以索引及审计记录为准。

### Patchouli 与 Engines

- [Patchouli 旧设计](./patchouli/ActiveMemoryGenerationDecouplingDesign.md)、[被动摄入改造计划](./patchouli/PatchouliPassiveIngestRefactorPlan.md)与[Transcript 双视图重构](./patchouli/PatchouliTranscriptDualViewRefactor.md)：分别由 [Perception](../../patchouli/perception.md)、[Generation](../../patchouli/generation.md)、[Artifacts](../../patchouli/artifacts.md) 与 [System Passive Ingress](../../system/passive-ingress.md) 取代；
- [Engines 旧索引](./engines/README.md)：代码目录仍存在，但文档入口已由 [Patchouli](../../patchouli/README.md) 与 [Gateway](../../gateway/README.md) 接管；
- [旧 Perception](./engines/perception.md)、[Generation](./engines/generation.md)、[Retrieval](./engines/retrieval.md) 与 [Lifecycle](./engines/lifecycle.md)：已分别并入 Patchouli 当前模块文档；Perception 的损坏正文按原样保留，仅修正归档后的替代入口链接；
- [旧 MemoryCompiler 索引与 Phase 1 设计](./engines/memory_compiler/README.md)：已由 [MemoryCompiler 当前设计](../../patchouli/memory-compiler.md)取代。
- [Engines 源码 README](./source-readmes/README.md)：原先散落于 `src/hivememory/engines/` 的 Perception、Generation、Retrieval 与 Lifecycle 说明；有效理念已经并入当前 Patchouli 文档，旧 API 和目录清单仅供追溯。

### Gateway

- [旧 Gateway Engine 设计](./engines/gateway.md)：保留 The Eye、漏斗式入口与“Compute Once, Use Everywhere”的历史背景；当前独立子系统、固定 workflow 与受限复用边界以 [Gateway](../../gateway/README.md) 为准。

### Alice 与 Agent Runtime

- [Alice Phase 1/2](./alice/phases/README.md)：保留 Agent Profile、权限、CALL 与子帧 IPC 的阶段性设计背景；当前职责与行为以 [Alice](../../alice/README.md)及其模块文档为准；
- [旧 Agent Runtime 索引](./agent_runtime/README.md)：共享执行层仍存在于代码中，但文档入口已经并入 Alice；
- [PendingAtom 分阶段设计](./agent_runtime/pending_atom/README.md)：Cache、状态统一、Runtime 与 MaterializeTask 的演进记录；当前状态机、物化交接和 alias 解析以 [Alice PendingAtom](../../alice/pending-atom.md)为准。

### Frontend

- [旧 Frontend UI/UX 设计](./frontend/FrontendDesign.md)：保留透明、沉浸、可观测、日月五行水晶与同屏工作台的早期探索；当前事实以 [Frontend](../../frontend/README.md) 为准；
- [前端状态持久化调研](./frontend/frontend-state-persistence-research.md)：保留 UI 偏好与业务真相分层的调研过程；当前 store、传输和本地契约治理以[状态与传输](../../frontend/state-and-transports.md)为准；
- [Memory Garden 概念](./frontend/MemoryGardenUI.md)：保留记忆资产与人工干预隐喻；真正语义检索、Pin/Lock、归档和统计大屏仍不是当前能力。

逐篇去向、未继承设计和验证口径见[第 4～6 节迁移审计](../plans/documentation-migration-audit-sections-4-6.md)、[第 7 节迁移审计](../plans/documentation-migration-audit-section-7.md)、[第 8 节迁移审计](../plans/documentation-migration-audit-section-8.md)与[第 9～10 节迁移审计](../plans/documentation-migration-audit-sections-9-10.md)。
