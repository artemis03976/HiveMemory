---
title: Hybrid 检索启用 reranker 时分数阈值不生效
status: todo
owner: patchouli
scope: hybrid-retriever-score-threshold-with-reranker
code_paths:
  - src/hivememory/engines/retrieval/retriever.py
  - src/hivememory/engines/retrieval/reranker.py
  - src/hivememory/engines/retrieval/engine.py
  - src/hivememory/patchouli/services/retrieval.py
related_docs:
  - docs/patchouli/retrieval.md
last_reviewed: 2026-10-04
---

# Hybrid 检索启用 reranker 时分数阈值不生效

## 问题与证据

`HybridRetriever.retrieve`（[`engines/retrieval/retriever.py`](../../src/hivememory/engines/retrieval/retriever.py)）在融合与可选重排之后按分数阈值过滤，但条件与它的注释和日志相反：

- 代码只在 reranker 为 `NoopReranker` 时过滤，启用真实 reranker（`CrossEncoderReranker`）时跳过；
- 注释写“仅当使用了非 NoopReranker 时才应用阈值”，跳过时的日志写“当前使用的是 RRF 分数，不适用 Cosine 阈值”——这两处描述的都是相反的条件：不重排时分数是 RRF 分数，重排后才是 reranker 分数；
- `CrossEncoderReranker._normalize_score` 用 sigmoid 把分数映射到 0–1，其说明写明是“以便于阈值过滤”。

在默认配置下的结果（2026-10-04 核对）：

- `configs/config.yaml` 启用 reranker（`BAAI/bge-reranker-base`）；
- `RetrievalFamiliar.retrieve` 不传阈值，`RetrievalEngine.retrieve` 的默认阈值是 0.75，传给 Hybrid 后因启用 reranker 而跳过；
- 因此生产检索实际上从不应用分数阈值，只按 top-k 截取。

e2e 测试 `tests/e2e/component/test_retrieval_e2e.py::TestEndToEndFlow::test_empty_result_handling` 暴露了这一点：以无意义查询和 0.9 阈值检索，期望空结果，实际返回 5 条，最高分约 0.348。该条件自 2026-01 的混合检索实现起就存在，不是身份第二批引入的；第二批修复该 e2e 文件的事件循环问题后，测试才运行到这一断言。

## 影响范围

- Hybrid 检索的阈值过滤：prepare 的检索、MTP SEARCH 与能力层的语义检索都经过它；
- 召回数量与提示上下文：直接改正条件会开始按 0.75 过滤 sigmoid 后的 reranker 分数。上述 e2e 中相关结果的分数约在 0.35 左右，按 0.75 过滤可能丢掉大部分结果。

## 待决选项

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 改正条件：启用 reranker 时按 reranker 分数过滤，不重排时不对 RRF 分数套用阈值 | 需要为 sigmoid 后的 reranker 分数选定阈值，并重新审视 `RetrievalEngine` 的 0.75 默认值；召回数量会变化，需要以真实模型核对 |
| B | 维持“启用 reranker 时不应用阈值”，改正注释与日志 | 检索行为不变；`test_empty_result_handling` 的期望需要改写，`_normalize_score` 的说明与引擎默认阈值的意义需要重新表述 |

## 完成条件

- `HybridRetriever` 的阈值条件、注释与日志一致，`RetrievalEngine` 的默认阈值与所选语义相符；
- `test_empty_result_handling` 按所选语义断言并通过；
- [Patchouli 检索](../patchouli/retrieval.md)第 5、9 节按最终代码更新阈值语义与限制。

## 追踪

- 2026-10-04：修复检索 e2e 的事件循环问题后发现（分支 `refactor/identity-access-batch-2`）；未绑定版本或 Issue。
