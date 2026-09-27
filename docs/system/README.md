---
title: System
status: current
owner: system
scope: top-level-composition-and-runtime
code_paths:
  - src/hivememory/system/system.py
  - src/hivememory/system/assembler.py
  - src/hivememory/system/application/
  - src/hivememory/system/access/
  - src/hivememory/system/services/passive/
  - src/hivememory/config/
  - src/hivememory/components/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/architecture/boundaries.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-09-26
---

# System

本目录是 HiveMemory 顶层 System 当前设计的入口，负责说明系统装配、门面与应用服务、共享运行时设施、配置与注册表和被动摄入基础设施。

System 的职责不是把 Gateway、Patchouli 和 Alice 的领域行为重新实现一遍，而是装配所有组件、决定它们如何启动和收尾，并通过门面把服务交给入口。它是依赖图的顶点：除入口外不被任何包导入。System 拥有根配置的加载与分段注入、模型/Provider 注册表、调用来源接入登记与 Principal authentication，以及被动摄入这一系统级能力；总线、维护调度器、work queue 与观测旁路的机制实现位于 `components` 包，System 负责装配这些实例并管理其启停。资源能力（`workspace.capability`）与 chat 编排（`alice.application`，暂置）同样由 System 装配并经门面交给入口，但不属于 System。

Workspace 是跨 System 与各领域 Store 的资源归属坐标，不是由 System 复制出的独立运行时。System 负责在入口和后台交接中保留 `IdentityScope`，并装配、在关闭时最后清理进程级唯一的 `WorkspaceAssetStore`（属于 workspace）；Topic、Memory 和 Artifact 的领域语义仍由 Patchouli 所有。详见 [Workspace 架构](../architecture/workspace.md)。

当前设计文档：

- [组合根与生命周期](./composition.md)：`HiveMemorySystem`、`SystemAssembler`、子系统装配和启停顺序；
- [应用服务](./application-services.md)：门面提供的服务——chat 编排与取消（`alice.application`）、资源能力层（`workspace.capability`）、就绪检查与跨子系统编排；
- [被动摄入](./passive-ingress.md)：外部对话事件、turn accumulator、submission queue 和降级语义；
- [运行时与总线](./runtime-and-bus.md)：`components` 包中的 GlobalSystemBus、维护调度器、work queue 与串行门，以及 System 对它们的装配与启停；
- [Workspace 架构](../architecture/workspace.md)：身份坐标、资源归属、AssetStore 生命周期与共享底座边界；
- [配置与注册表](./configuration.md)：`config` 包的配置段与根配置加载、所有权、模型/Provider 注册和覆盖边界；
- [可观测性](./observability.md)：RuntimeEvent、operation observer、健康状态和旁路原则；
- [i18n](./i18n.md)：语言归一化、解析优先级和各文本领域的当前入口。

System 与 Gateway 的跨边界交接仍以[子系统公共契约](../contracts/subsystem-contracts.md)、[公开路由与事件](../contracts/routes-and-events.md)和[系统边界](../architecture/boundaries.md)为准。本目录中的应用服务文档描述“谁编排”，不会重新定义 route 字符串或子系统内部状态。

清单第 4～6 节和原 `docs/mod/` 中的 System 材料已经逐篇复核并移入 Archive 或 Plans；包括编码损坏的 MemoryCompiler i18n 计划在内，均不再保留原路径入口。审计结论见[第 4～6 节迁移记录](../archive/plans/documentation-migration-audit-sections-4-6.md)与[`docs/mod` 迁移记录](../archive/plans/documentation-migration-audit-docs-mod.md)。任何历史稿与当前文档冲突时，均以当前代码、测试和本目录文档为准。
