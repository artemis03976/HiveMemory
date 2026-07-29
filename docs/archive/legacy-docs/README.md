---
title: Legacy Documents
status: current
owner: project
scope: superseded-documentation
last_reviewed: 2026-07-29
---

# Legacy Documents

本目录保存不属于已完成计划或历史架构、但仍需保留追溯价值的旧文档。进入本目录前必须指出替代文档；无法指出替代入口的内容不得仅为清理目录而归档。

## 已归档材料

### 顶层旧入口

- [环境搭建指南](./SETUP.md)：由 [Help](../../help/README.md)取代；
- [顶层 TODO](./TODO.md)：已拆分到 [Todo](../../todo/README.md)；
- [可观测性与日志流设计](./ObservabilityDesign.md)：由 [System 可观测性](../../system/observability.md)与[事件契约](../../contracts/routes-and-events.md)取代。

### Architecture 混合文档

- [不可变数据模型现状与规划](./architecture/DataModelImmutabilityStatusAndRoadmap.md)：已拆分为[当前数据模型](../../architecture/data-model.md)、[ADR-0001](../../architecture/decisions/0001-data-model-mutability-and-boundary-projection.md)与[治理计划](../../plans/data-model-mutability-governance.md)。

### Protocols 与 i18n

- [旧 Protocols 索引](./protocols/README.md)：由 [Contracts](../../contracts/README.md)取代；
- [旧 MTP 规范](./protocols/MemoryToolProtocol.md)与[错误结构设计](./protocols/MTPErrorStructureDesign.md)：由当前 MTP / Error Contracts 取代；
- [统一维护调度设计](./protocols/PatchouliUnifiedMaintenanceSchedulerDesign.md)：由 [System runtime](../../system/runtime-and-bus.md)及 Patchouli Perception/Lifecycle 取代；
- [旧 i18n 索引](./protocols/i18n/README.md)：由[全局 i18n](../../system/i18n.md)取代。`MemoryCompilerI18nMigrationPlan.md` 保留原始损坏字节，替代入口和处置结论以索引及审计记录为准。

逐篇去向、未继承设计和验证口径见[第 4～6 节迁移审计](../plans/documentation-migration-audit-sections-4-6.md)。
