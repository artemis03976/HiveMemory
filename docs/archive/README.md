---
title: Archive
status: current
owner: project
scope: archive-rules-and-historical-document-index
last_reviewed: 2026-11-20
---

# Archive

本目录保存已经完成、被替代或不再描述当前系统的材料。归档内容只用于追溯，不得作为当前实现依据。

## 分类

- [Legacy Architecture](./legacy-architecture/README.md)：被当前架构替代的系统总纲与演进记录；
- [Archived Plans](./plans/README.md)：已经完成或被替代的实施计划；
- [Archived Todo](./todo/README.md)：已完成的缺陷修复与技术债处置记录；
- [Archived Ideas](./ideas/README.md)：已经实现、被替代或不再跟踪的 Idea；
- [Legacy Documents](./legacy-docs/README.md)：其他被替代但仍有追溯价值的文档。

当前设计统一从[项目总览](../PROJECT.md)进入；跨版本质量目标从 [Governance](../governance/README.md) 进入；已承诺实施工作从 [Plans](../plans/README.md) 进入；未承诺方向从 [Ideas](../ideas/README.md) 进入。迁移清单中的旧材料、Ideas、源码 README 和迁移前遗留的 `archive/mod/` 已完成逐篇复核；完整结论、保留理念、拒绝项与最终门禁见[文档迁移最终收口审计](./plans/documentation-migration-finalization-audit.md)。

## 规则

**收录什么**。以下内容进入 `archive/`：已被替代的旧设计；只用于某个历史迁移阶段的调查和清单；已失效但仍有追溯价值的架构说明；文档重构前无法直接删除的历史材料；已经实现、被替代或不再跟踪的 Idea。归档时把状态改为 `archived` 或 `superseded`，写明归档日期与替代它的当前文档，尽可能记录实现版本、PR 或决策，并更新仍把它当作当前依据的入链。归档文档原则上冻结，只修复严重错误、断链或补充追溯信息。

计划与 Todo 不进入本目录（owner，2026-11-18），完成后在收尾时删除，分拣与删除的要求见[文档治理规范](../DOCUMENTATION.md)第 10 节。`archive/plans/` 与 `archive/todo/` 中此前归档的文档保持冻结，不再新增，也不专门整理。

**归档索引**（`archive/plans/`、`archive/todo/` 与 `archive/ideas/` 的 README）每条一行，按版本或时间分节：归档的文档写标题链接、归档日期、实现所在的 PR（尚未合并时写提交）与替代它的当前文档（至多两个链接）；删除的事项写标题、日期、实现所在的 PR 或去向，以及删除前最后版本所在的 commit。索引不复述实施过程、验收结果与决定。
