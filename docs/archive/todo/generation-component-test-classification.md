---
title: 生成组件测试的分类与重复覆盖
status: archived
archived_at: 2026-11-01
implemented_by: PR #164
superseded_by: docs/patchouli/generation.md
owner: patchouli
scope: generation-component-test-classification-and-dedup-assertions
priority: unscheduled
code_paths:
  - tests/integration/patchouli/test_generation_dedup_recall.py
  - tests/integration/patchouli/test_generation_topic_extraction.py
  - tests/integration/patchouli/test_retrieval_real_models.py
  - tests/unit/engines/generation/test_deduplicator.py
  - tests/unit/engines/memory_compiler/test_compiler.py
  - .github/workflows/ci.yml
related_docs:
  - docs/governance/testing/test-design-standards.md
  - docs/todo/generation-dedup-decision-signal.md
last_reviewed: 2026-11-01
---

# 生成组件测试的分类与重复覆盖

> 状态：Archived（2026-11-01，已处理，处置见文末一节）。查重查询的当前事实见 [Patchouli 生成](../../patchouli/generation.md#4-查重与合并)第 4 节。以下为登记时的原文，指向已删除测试文件的链接改为纯文本。

## 问题与证据

审查查重与更新修正时发现，生成组件测试（`tests/e2e/component/test_generation_e2e.py`，已删除）直接调用提取器、查重器与生成引擎，并使用真实模型和向量存储，没有经过公共入口，现位于 `tests/e2e/` 且整文件标记 `e2e`、`live_llm`。

- `TestDeduplicationLogic.test_decision_identical_duplicate` 同时接受 IDENTICAL 与 SIMILAR，不能区分内容一致性判断错误和向量分数波动；DISTINCT、IDENTICAL、SIMILAR 的判断矩阵已有[归属层单元测试](../../../tests/unit/engines/generation/test_deduplicator.py)。
- 同文件的提取与 Schema 用例也以组件作为入口，运行条件和主类型没有分开表达。

旧的三个覆盖更新用例及其专用数据已随契约变更删除。其余分类与重复覆盖问题不属于查重写入方式修正，保留为未排期的测试治理事项。

## 完成条件

- 将需要真实模型或存储的组件协作验证归入 `tests/integration/`，按实际依赖标记 `live_llm`、`real_infra`；公共入口链路才留在 `tests/e2e/`。
- 查重决策矩阵只由现有单元测试维护；较宽层只保留真实向量召回接入查重的代表性用例，并明确固定样本的预期，移除允许多种决策的断言。
- 检索现有提取与 Schema 测试，合并重复项，确认测试仍能捕获其声称保护的缺陷。

执行时遵循[测试设计规范](../../governance/testing/test-design-standards.md)，不以新增 marker 代替分类调整。

## 处置（2026-11-01）

调查时用真实 embedding 量出：内容与已有记忆相同的草稿只得 0.902，到不了 IDENTICAL 所需的 0.95。原因是查重查询只拼接 `title summary`，入库向量却由带字段标签的 `DENSE_EMBEDDING` 编译文本编码。owner 决定先修这个缺陷：查询改为按同一编译目标编译草稿，判定方式与阈值不变；修复前后的分数与真实库最近邻分布的变化记录在[查重决策信号 Todo](../../todo/generation-dedup-decision-signal.md)。

测试按归属层收敛（owner 决定只留事实与闲聊两个真实模型用例，并一起清理检索组件测试）：

- **生成**：`tests/e2e/component/test_generation_e2e.py` 与其测试数据删除。查重矩阵、提取解析与引擎路由只由现有单元测试维护；集成层保留两份：
  - [查重召回](../../../tests/integration/patchouli/test_generation_dedup_recall.py)（`real_infra`）：真实向量下内容相同的草稿只触碰已有记忆，修复前的查询下这一用例失败；
  - [话题提取](../../../tests/integration/patchouli/test_generation_topic_extraction.py)（`live_llm`）：经生产渲染，事实性对话产出带服务名的记忆，闲聊不产出。原样本是让模型记住 API Key，与提取提示词不写入密钥的要求冲突，改为一项配置决定。
  - 删去的用例：直接调用提取器且对话格式不是生产渲染的四个提取用例；在测试中复写草稿转换再检查字段存在的 Schema 用例；只断言 `len > 0` 的引擎流程用例。
- **检索**：`tests/e2e/component/test_retrieval_e2e.py` 删除，测试数据只保留黄金记忆。
  - full、compact、cascade 三种检索上下文策略原先只有这里覆盖，改为[编译器单元测试](../../../tests/unit/engines/memory_compiler/test_compiler.py)中的三个用例，按策略精确断言正文与摘要的取舍；
  - [真实模型的召回与排序](../../../tests/integration/patchouli/test_retrieval_real_models.py)（`real_infra`）保留语义召回、精确标识召回、同名歧义的区分、接入真实重排模型后的首位与 Dense 阈值五个用例，去掉 `or` 兜底；
  - 删去的用例：“无关查询的重排分数低”在配置的混合召回下没有任何候选到达重排，`all` 对空列表恒真，从未验证重排；只断言非空的两个流程用例；以 `or` 拼接阈值的指标用例。
- **其他**：根 `conftest.py` 中只服务这两个文件的 rich 打印工具与 `console` fixture 删除；CI 手动任务同时运行 integration 中的 `live_llm` 与 `real_infra` 测试，Qdrant 镜像随 qdrant-client 升到 v1.19.2。
