---
title: Legacy Protocols Migration Index
status: superseded
owner: system
scope: protocols-migration
last_reviewed: 2026-07-28
---

# Protocols 迁移入口

跨子系统稳定契约已经迁入 [Contracts](../contracts/README.md)。

当前规范：

- [Memory Tool Protocol](../contracts/mtp.md)
- [跨边界错误模型](../contracts/error-model.md)
- [公开路由与事件](../contracts/routes-and-events.md)

仍待 P1 迁移：

- [Patchouli 统一维护调度设计](./PatchouliUnifiedMaintenanceSchedulerDesign.md)：将合并到 System runtime 与 Patchouli lifecycle 当前文档；
- [i18n](./i18n/README.md)：将合并到 System i18n 当前文档。

旧 [MemoryToolProtocol.md](./MemoryToolProtocol.md) 和 [MTPErrorStructureDesign.md](./MTPErrorStructureDesign.md) 已标记为 `superseded`，只保留历史参考。
