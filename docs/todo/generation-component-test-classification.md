---
title: 生成组件测试的分类与重复覆盖
status: todo
owner: patchouli
scope: generation-component-test-classification-and-dedup-assertions
priority: unscheduled
code_paths:
  - tests/e2e/component/test_generation_e2e.py
  - tests/unit/engines/generation/test_deduplicator.py
related_docs:
  - docs/governance/testing/test-design-standards.md
last_reviewed: 2026-10-31
---

# 生成组件测试的分类与重复覆盖

## 问题与证据

审查查重与更新修正时发现，[生成组件测试](../../tests/e2e/component/test_generation_e2e.py)直接调用提取器、查重器与生成引擎，并使用真实模型和向量存储，没有经过公共入口，现位于 `tests/e2e/` 且整文件标记 `e2e`、`live_llm`。

- `TestDeduplicationLogic.test_decision_identical_duplicate` 同时接受 IDENTICAL 与 SIMILAR，不能区分内容一致性判断错误和向量分数波动；DISTINCT、IDENTICAL、SIMILAR 的判断矩阵已有[归属层单元测试](../../tests/unit/engines/generation/test_deduplicator.py)。
- 同文件的提取与 Schema 用例也以组件作为入口，运行条件和主类型没有分开表达。

旧的三个覆盖更新用例及其专用数据已随契约变更删除。其余分类与重复覆盖问题不属于查重写入方式修正，保留为未排期的测试治理事项。

## 完成条件

- 将需要真实模型或存储的组件协作验证归入 `tests/integration/`，按实际依赖标记 `live_llm`、`real_infra`；公共入口链路才留在 `tests/e2e/`。
- 查重决策矩阵只由现有单元测试维护；较宽层只保留真实向量召回接入查重的代表性用例，并明确固定样本的预期，移除允许多种决策的断言。
- 检索现有提取与 Schema 测试，合并重复项，确认测试仍能捕获其声称保护的缺陷。

执行时遵循[测试设计规范](../governance/testing/test-design-standards.md)，不以新增 marker 代替分类调整。
