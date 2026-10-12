---
title: 启用 reranker 时检索分数阈值不生效
status: todo
owner: patchouli
scope: retrieval-score-threshold-with-reranker
code_paths:
  - src/hivememory/engines/retrieval/engine.py
  - src/hivememory/engines/retrieval/retriever.py
  - src/hivememory/engines/retrieval/reranker.py
  - src/hivememory/patchouli/services/retrieval.py
related_docs:
  - docs/patchouli/retrieval.md
last_reviewed: 2026-10-12
---

# 启用 reranker 时检索分数阈值不生效

## 问题与证据

重排序是 `RetrievalEngine`（[`engines/retrieval/engine.py`](../../src/hivememory/engines/retrieval/engine.py)）中召回之后的独立阶段：引擎按重排器声明的候选数向任意检索器取候选，交给重排器后截取 top-k；`retrieval.reranker` 关闭或缺少重排服务时由 `NoopReranker` 顶替。阈值在各阶段的处理是：

- Dense/Sparse 把阈值交给存储召回，按 cosine/BM25 原始分数过滤；
- Hybrid 融合后不套用阈值（融合分数与相似度阈值不可比），其内部的 Dense/Sparse 使用各自配置的阈值，不接收调用方传入的阈值；
- 引擎把阈值交给检索器，重排后的分数不再过滤；
- `CrossEncoderReranker._normalize_score` 用 sigmoid 把分数映射到 0–1，其说明写明是“以便于阈值过滤”，但当前没有任何阶段按重排分数过滤。

在默认配置下的结果（2026-10-12 核对）：

- `configs/config.yaml` 使用 Hybrid 召回并启用 reranker（`BAAI/bge-reranker-base`）；
- `RetrievalFamiliar.retrieve` 不传阈值，`RetrievalEngine.retrieve` 的默认阈值是 0.75，交给 Hybrid 后被忽略；
- 因此生产检索实际上从不应用分数阈值，只按 top-k 截取。

e2e 测试 `tests/e2e/component/test_retrieval_e2e.py::TestEndToEndFlow::test_empty_result_handling` 暴露了这一点：以无意义查询和 0.9 阈值检索，期望空结果，实际返回 5 条，最高分约 0.348。该条件自 2026-01 的混合检索实现起就存在；2026-10-04 修复该 e2e 文件的事件循环问题后，测试才运行到这一断言。

2026-10-12 重排序从 Hybrid 内部移出、成为 `RetrievalEngine` 中适用于任意检索器的独立阶段，原 Hybrid 中“仅对 `NoopReranker` 过滤”的反向条件随之删除（生产装配从未构造 `NoopReranker`，该分支从未生效）。待决的阈值语义不变，范围从 Hybrid 扩大到重排阶段。

## 影响范围

- 启用重排时的阈值过滤：prepare 的检索、MTP SEARCH 与能力层的语义检索都经过它；
- 召回数量与提示上下文：若开始按 0.75 过滤 sigmoid 后的 reranker 分数，上述 e2e 中相关结果的分数约在 0.35 左右，可能丢掉大部分结果。

## 待决选项

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 引擎的重排阶段按 reranker 分数过滤；召回阶段与融合分数不套用调用方阈值 | 需要为 sigmoid 后的 reranker 分数选定阈值，并重新审视 `RetrievalEngine` 的 0.75 默认值；召回数量会变化，需要以真实模型核对 |
| B | 维持“重排后不过滤”，阈值只作用于 Dense/Sparse 的存储召回 | 检索行为不变；`test_empty_result_handling` 的期望需要改写，`_normalize_score` 的说明与引擎默认阈值的意义需要重新表述 |

## 完成条件

- 重排阶段、Hybrid 融合与 Dense/Sparse 召回的阈值处理与所选语义一致，`RetrievalEngine` 的默认阈值与之相符；
- `test_empty_result_handling` 按所选语义断言并通过；
- [Patchouli 检索](../patchouli/retrieval.md)第 5、9 节按最终代码更新阈值语义与限制。

## 追踪

- 2026-10-04：修复检索 e2e 的事件循环问题后发现（分支 `refactor/identity-access-batch-2`）；未绑定版本或 Issue。
- 2026-10-12：重排序改为独立阶段后重新核对证据（分支 `fix/mypy-typing-baseline`）。
