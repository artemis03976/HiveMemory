---
title: Runtime Configuration Wiring Drift
status: todo
owner: system
scope: unwired-runtime-configuration-fields
related_docs:
  - docs/system/configuration.md
  - docs/patchouli/retrieval.md
  - docs/patchouli/lifecycle.md
  - docs/patchouli/perception.md
last_reviewed: 2026-07-30
---

# 清理已声明但未接线的运行时配置

## 问题与证据

当前配置模型中仍有字段容易被误认为已经生效：

- `RetrievalModeConfig.time_weight` 已声明，但 Retrieval fusion 主路径尚未消费它；
- Lifecycle 的 `high_watermark` 已声明，但当前 Engine/GC 只按 low watermark 归档；

已移出清单：Page Folding 的 `fold_retain_recent_blocks` 已完成接线，不再属于未生效配置。

## 影响

- 运维或用户修改配置后，实际行为可能不变；
- 配置文件、当前文档和代码之间形成隐性分叉；
- 调试时很难区分“算法没有效果”和“配置根本未接线”。

## 完成条件

- 对每个字段作出明确裁定：接入主路径、改名为 reserved/planned，或删除；
- 若接入，补齐所有模式的语义、默认值、边界和测试，并在当前文档说明它影响哪一层；
- 若保留为未来字段，配置 schema、Help 和文档明确标注未生效，不让生产配置静默接受假能力；
- 增加配置变更后的行为断言，避免只测试“字段可以解析”；
- 同步更新 `docs/system/configuration.md`、相关 Patchouli 文档和示例配置。
