---
title: Runtime Configuration Wiring Drift
status: archived
owner: system
scope: unwired-runtime-configuration-fields
archived_at: 2026-07-30
superseded_by:
  - docs/system/configuration.md
  - docs/patchouli/retrieval.md
  - docs/patchouli/lifecycle.md
related_docs:
  - docs/system/configuration.md
  - docs/patchouli/retrieval.md
  - docs/patchouli/lifecycle.md
  - docs/patchouli/perception.md
last_reviewed: 2026-07-30
---

# 清理已声明但未接线的运行时配置

## 问题与证据

处理前的配置模型中有两个字段容易被误认为已经生效：

- `RetrievalModeConfig.time_weight` 曾经声明，但 Retrieval fusion 主路径没有消费它；
- Lifecycle 的 `high_watermark` 曾经声明，但 Engine/GC 只按 low watermark 归档；

已移出清单：Page Folding 的 `fold_retain_recent_blocks` 已完成接线，不再属于未生效配置。

## 处理结果

本 Todo 于 2026-07-30 完成，两个冗余字段均选择删除，而不是为尚未形成的能力保留占位配置：

- `RetrievalModeConfig.time_weight` 已从字段声明和四种 Adaptive Fusion 默认模式中移除。时间衰减继续由 DenseRetriever 的 `enable_time_decay` 与 `time_decay_days` 单一解释；
- Lifecycle `high_watermark` 已从后端模型、默认 YAML、前端 TypeScript 配置结构和 mock 配置中移除。当前 GC 只公开并消费 `garbage_collector.low_watermark`；
- Pydantic 子模型继续遵循项目级 `extra="ignore"` 兼容策略，因此旧输入中的两个键会被校验模型裁掉，并在配置 API 下一次持久化时消失，不会进入公开序列化结果；
- 配置回归测试同时约束 JSON schema、默认序列化、旧输入裁剪和 Adaptive Fusion 仍有效的 dense/sparse 模式权重。

## 处理前影响

- 运维或用户修改这些配置后，实际行为不会改变；
- 配置文件、当前文档和代码之间形成了隐性分叉；
- 调试时很难区分“算法没有效果”和“配置根本未接线”。

## 完成条件（已满足）

- 对每个字段作出明确裁定：接入主路径、改名为 reserved/planned，或删除；
- 若接入，补齐所有模式的语义、默认值、边界和测试，并在当前文档说明它影响哪一层；
- 若保留为未来字段，配置 schema、Help 和文档明确标注未生效，不让生产配置静默接受假能力；
- 增加配置变更后的行为断言，避免只测试“字段可以解析”；
- 同步更新 `docs/system/configuration.md`、相关 Patchouli 文档和示例配置。
