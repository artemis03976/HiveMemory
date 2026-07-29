---
title: Legacy Protocols Migration Index
status: archived
owner: system
scope: protocols-migration
archived_at: 2026-07-29
superseded_by: docs/contracts/README.md
---

# Protocols 迁移入口

跨子系统稳定契约已经迁入 [Contracts](../../../contracts/README.md)。

当前规范：

- [Memory Tool Protocol](../../../contracts/mtp.md)
- [跨边界错误模型](../../../contracts/error-model.md)
- [公开路由与事件](../../../contracts/routes-and-events.md)

已完成迁移的历史材料：

- [Patchouli 统一维护调度设计](./PatchouliUnifiedMaintenanceSchedulerDesign.md)：当前 scheduler 与业务所有权已分别进入 System runtime 和 Patchouli 文档；
- [i18n](./i18n/README.md)：当前语言解析与文本域已进入 System i18n 文档。

旧 [MemoryToolProtocol.md](./MemoryToolProtocol.md) 和 [MTPErrorStructureDesign.md](./MTPErrorStructureDesign.md) 已标记为 `superseded`，只保留历史参考。
