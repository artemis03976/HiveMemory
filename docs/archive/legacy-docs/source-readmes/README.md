---
title: Legacy Source READMEs
status: current
owner: project
scope: archived-source-directory-documentation
last_reviewed: 2026-07-29
---

# Legacy Source READMEs

本目录保存曾经散落在源码树中的模块 README。它们能够解释早期实现意图，但会随着类名、目录结构和控制面迁移迅速失真，因此不再承担当前设计说明；源码附近的局部注释只解释实现，跨模块职责、公共契约和真实限制统一由 `docs/` 当前文档维护。

## Engines

- [Perception](./engines/perception/README.md)：由[感知与短期话题](../../../patchouli/perception.md)取代；
- [Generation](./engines/generation/README.md)：由[记忆生成](../../../patchouli/generation.md)取代；
- [Retrieval](./engines/retrieval/README.md)：由[记忆检索](../../../patchouli/retrieval.md)取代；
- [Lifecycle](./engines/lifecycle/README.md)：由[记忆生命周期](../../../patchouli/lifecycle.md)取代。

逐篇承接与拒绝继承项见[第 7 节迁移审计](../../plans/documentation-migration-audit-section-7.md)。
