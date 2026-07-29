---
title: Historical Implementation Plans
status: current
owner: project
scope: completed-or-superseded-implementation-records
last_reviewed: 2026-07-29
---

# 历史实施计划

本目录保存从原 `docs/mod/` 迁入、已经完成或被后续设计替代的实施稿。它们记录问题是如何被识别、方案为何采用以及迁移如何分阶段发生，但不再描述当前代码；读者应先阅读每篇顶部的 `superseded_by`，再把正文作为演化证据使用。

## Alice 与 Agent Runtime

- [Agent Runtime 边界裁定](./agent-runtime-boundary.md)：当前入口为 Alice 总览、Agent Runtime 与多 Agent 编排；
- [Agent loop 与 Orchestrator 解耦](./agent-loop-decoupling.md)：当前入口为 Agent Runtime 与多 Agent 编排；
- [PendingAtom 生命周期](./pending-atom-lifecycle.md)：当前入口为 Alice PendingAtom。

## Patchouli 与 MemoryCompiler

- [启用生命周期维护](./enable-lifecycle-maintenance.md)：当前入口为记忆生命周期与 System Runtime；原稿在迁移前已经位于 `archive/mod/`，本轮并入统一历史实施计划分类；
- [MemoryCompiler IR](./memory-compiler-ir.md)：当前入口为 MemoryCompiler；
- [MemoryCompiler 与 Retrieval 分离](./memory-compiler-retrieval-refactor.md)：当前入口为 MemoryCompiler 与记忆检索；
- [记忆生成任务管理增强](./memory-generation-management-enhancement.md)：当前入口为记忆生成；
- [Patchouli 子系统重构](./patchouli-subsystem-refactor.md)：当前入口为 Patchouli 及其模块文档；
- [v0.5.0 Data Durability 与 Async Cold Path](./v0.5.0-data-durability-and-async-cold-path.md)：当前入口为 Artifacts、MemoryLibrary 与记忆生成；
- [v0.5.2 Async-Native Adaptation](./v0.5.2-async-native-adaptation.md)：当前入口为 MemoryLibrary、记忆生成与记忆检索。

## System 与 Gateway

- [v0.4.0 Runtime Control 与 Observability](./v0.4.0-runtime-control-and-observability.md)：当前入口为 System 应用服务、Runtime 与可观测性；
- [v0.5.1 Infrastructure Cleanup](./v0.5.1-infrastructure-cleanup.md)：当前入口为配置、Artifacts、Agent Runtime 与 MTP Runtime；
- [v0.6.0 Gateway System](./v0.6.0-gateway-system.md)：当前入口为 Gateway 总览、workflow、analysis 与 commands；
- [v0.6.0 Global Command System](./v0.6.0-global-command-system.md)：当前入口为 Gateway commands；
- [v0.6.0 Passive Ingress](./v0.6.0-passive-ingress.md)：当前入口为 System Passive Ingress 与 Gateway workflow；
- [User Query Analysis 第一代技术债](./v0.6.0-user-query-analysis-gen1-tech-debt.md)：当前入口为 Gateway analysis；第二代方向尚未形成独立排期。

三篇仍有效的未来工作没有进入本目录：[RuntimeEvent 生产端发布抽象重构](../../../plans/runtime-event-publishing-refactor.md)、[复合意图分解](../../../plans/v0.6.0-composite-intent-decomposition.md)和 [Local Work Queue Runtime](../../../plans/v0.6.1-local-work-queue-runtime.md)。逐篇分类依据见 [`docs/mod` 迁移审计](../documentation-migration-audit-docs-mod.md)。
