---
title: System
status: draft
owner: system
scope: top-level-composition-and-runtime
last_reviewed: 2026-07-27
---

# System

本目录是 HiveMemory 顶层系统当前设计的目标入口，负责说明系统装配、应用服务、全局运行时能力和跨子系统基础设施。

计划收录的当前设计包括：

- `composition.md`：`HiveMemorySystem`、`SystemAssembler` 与子系统装配；
- `application-services.md`：顶层应用服务及其编排边界；
- `passive-ingress.md`：外部对话的被动记忆摄入；
- `runtime-and-bus.md`：全局总线、运行控制与维护调度；
- `configuration.md`：配置所有权、覆盖关系和装配入口；
- `observability.md`：运行事件、日志、追踪与健康状态；
- `i18n.md`：全局语言解析与面向 Agent 的文本治理。

当前目录仍处于文档迁移阶段。上述文件尚未建立前，应根据代码、测试和[迁移清单](../plans/documentation-migration-inventory.md)核对旧文档，不得将本索引视为已经完成的系统设计说明。
