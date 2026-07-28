---
title: System
status: current
owner: system
scope: top-level-composition-and-runtime
code_paths:
  - src/hivememory/system/system.py
  - src/hivememory/system/assembler.py
  - src/hivememory/system/application/
  - src/hivememory/system/runtime/
  - src/hivememory/system/services/passive/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/architecture/boundaries.md
last_reviewed: 2026-07-28
---

# System

本目录是 HiveMemory 顶层 System 当前设计的入口，负责说明系统装配、应用服务、全局运行时能力和被动摄入基础设施。

System 的职责不是把 Gateway、Patchouli 和 Alice 的领域行为重新实现一遍，而是回答一次顶层用例如何开始、如何跨边界交接、如何取消和如何收尾。它是组合根和应用编排层，也是全局总线、维护调度器、观测旁路和配置/注册表的所有者。

当前设计文档：

- [组合根与生命周期](./composition.md)：`HiveMemorySystem`、`SystemAssembler`、子系统装配和启停顺序；
- [应用服务](./application-services.md)：主动 chat、取消、API facade 和跨子系统编排；
- [被动摄入](./passive-ingress.md)：外部对话事件、turn buffer、outbox 和降级语义；
- [运行时与总线](./runtime-and-bus.md)：GlobalSystemBus、维护调度器和运行控制；
- [配置与注册表](./configuration.md)：配置来源、所有权、模型/Provider 注册和覆盖边界；
- [可观测性](./observability.md)：RuntimeEvent、operation observer、健康状态和旁路原则；
- [i18n](./i18n.md)：语言归一化、解析优先级和各文本领域的当前入口。

System 与 Gateway 的跨边界交接仍以[子系统公共契约](../contracts/subsystem-contracts.md)、[公开路由与事件](../contracts/routes-and-events.md)和[系统边界](../architecture/boundaries.md)为准。本目录中的应用服务文档描述“谁编排”，不会重新定义 route 字符串或子系统内部状态。

本批涉及的旧 `docs/mod/` 设计、`docs/protocols/i18n/` 索引与可安全更新的材料，以及 `docs/ObservabilityDesign.md` 已标记为 `superseded`，仅作为迁移证据保留。已知编码损坏的 MemoryCompiler i18n 计划等待 P2 安全归档；任何旧材料与当前文档冲突时，均以当前代码、测试和本目录文档为准。
