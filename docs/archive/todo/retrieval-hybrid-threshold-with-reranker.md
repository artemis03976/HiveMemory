---
title: 检索分数阈值与候选截取收敛
status: archived
archived_at: 2026-10-12
implemented_by: fix/mypy-typing-baseline（本归档提交）
superseded_by: docs/patchouli/retrieval.md
owner: patchouli
scope: retrieval-score-threshold-and-candidate-truncation
code_paths:
  - src/hivememory/engines/retrieval/engine.py
  - src/hivememory/engines/retrieval/interfaces.py
  - src/hivememory/engines/retrieval/retriever.py
  - src/hivememory/engines/retrieval/fusion.py
  - src/hivememory/engines/retrieval/reranker.py
  - src/hivememory/config/patchouli.py
  - src/hivememory/infrastructure/storage/vector_store.py
  - src/hivememory/patchouli/services/retrieval.py
  - src/hivememory/patchouli/contracts/local_routes.py
  - src/hivememory/core/protocol/models.py
  - src/hivememory/workspace/contracts/operations.py
  - configs/config.yaml
related_docs:
  - docs/patchouli/retrieval.md
  - docs/contracts/mtp.md
  - docs/contracts/subsystem-contracts.md
last_reviewed: 2026-10-12
---

# 检索分数阈值与候选截取收敛

> **文档状态**：Archived（2026-10-12）。已在分支 `fix/mypy-typing-baseline` 实施并验收；当前事实见 [Patchouli 检索](../../patchouli/retrieval.md)第 4、5、9 节。下文“问题与证据”记录的是实施前的状态。

## 问题与证据

重排序是 `RetrievalEngine`（[`engines/retrieval/engine.py`](../../../src/hivememory/engines/retrieval/engine.py)）中召回之后的独立阶段：引擎按重排器声明的候选数（`CrossEncoderReranker` 为 `max(top_k, reranker.top_k)`）向任意检索器取候选，交给重排器后按本次请求的 `top_k` 截取；`retrieval.reranker` 关闭或缺少重排服务时由 `NoopReranker` 顶替，候选数即 `top_k`。请求的 `top_k` 来自 Gateway 检索计划（prepare）或 `RetrievalRequest.top_k`（MTP SEARCH 与能力层语义检索）。以下事实于 2026-10-12 核对。

### 分数阈值

召回、融合与重排各阶段的分数尺度互不相同：cosine 相似度、BM25 原始分、融合分、sigmoid 后的 cross-encoder 分。同一个阈值数字只能对其中一种有意义。

- `RetrievalEngine.retrieve` 的默认阈值是 0.75，原样交给检索器；`RetrievalFamiliar.retrieve` 不传阈值，src 中也没有其他调用方传值；
- Dense 把阈值交给存储召回，按 cosine 相似度过滤；
- Sparse 也把阈值交给存储，但 `VectorStore.search_memories`（[`vector_store.py`](../../../src/hivememory/infrastructure/storage/vector_store.py)）在 sparse 模式下把阈值固定为 `None`，因此 `sparse.score_threshold` 从未生效；
- Hybrid 忽略调用方阈值，其内部两路使用各自配置的阈值；Hybrid 顶层的 `score_threshold` 配置字段不起任何作用；
- 融合分数也不适合设阈值：RRF 的分数是 `Σ w/(k + rank)`，只反映名次，无意义查询的第 1 名与相关查询的第 1 名得分相同（默认 `rrf_k=60`、权重 1 时，只在一路排第 1 约为 0.0164，两路都排第 1 约为 0.0328），且尺度随 `rrf_k`、权重和关闭的路径变化；加权融合直接相加 cosine 与未归一化的 BM25 原始分，尺度随查询变化；
- 重排后的分数不再过滤；`CrossEncoderReranker._normalize_score` 的说明写明 sigmoid 映射“以便于阈值过滤”，但没有任何阶段按重排分数过滤。

结果是阈值参数的含义随检索器类型变化：Dense 时 0.75 生效，Sparse 与 Hybrid 时被忽略。默认配置（`configs/config.yaml`）使用 Hybrid 召回并启用 reranker（`BAAI/bge-reranker-base`），Dense 与 Sparse 的阈值均为 0.0，因此生产检索从不应用分数阈值。

e2e 测试 `tests/e2e/component/test_retrieval_e2e.py::TestEndToEndFlow::test_empty_result_handling` 暴露了这一点：以无意义查询和 0.9 阈值检索，期望空结果，实际返回 5 条，最高分约 0.348。该条件自 2026-01 的混合检索实现起就存在；2026-10-04 修复该 e2e 文件的事件循环问题后，测试才运行到这一断言。

### 候选截取

RRF 与加权融合（[`fusion.py`](../../../src/hivememory/engines/retrieval/fusion.py)）在全部融合入口（`fuse`、`fuse_multi`、`fuse_with_intent`）内按 `fusion.final_top_k` 截取，`configs/config.yaml` 中为 5。默认配置下，引擎为重排向 Hybrid 请求 20 条候选，Hybrid 融合后最多只交出 5 条，reranker 只能在 RRF 的前 5 名内重排，重排器声明的候选池不起作用。master 上 Hybrid 内部同样是“融合（截取到 5）→ 重排”的顺序，该问题早于重排序改为独立阶段。

## 决定

owner 于 2026-10-12 确定：

- 分数阈值只作用于 Dense 召回，按原始 cosine 相似度粗筛；交给 reranker 的候选即粗筛后的结果，重排分数不套用阈值；
- Sparse 不设阈值；
- Dense 的默认阈值为 0.5；
- 融合后不设分数阈值；
- 最终截取只在整个检索流程的最后、重排阶段之后发生：融合不再自行截取，Hybrid 按引擎请求的候选数交出融合结果；
- 引擎保留一个用户可配置的引擎级 `top_k`，只在请求未给出 `top_k` 时作为本次结果条数；最终截断仍只在重排阶段之后按本次结果条数进行一次；
- 重排阶段在编排中的位置固定：重排关闭或缺少重排服务时由 `NoopReranker` 透传，最终截断始终位于这一步之后（`NoopReranker` 与 `create_reranker` 的替换已实现，本 Todo 保持该结构）。

## 影响范围

- 生产检索开始过滤：Hybrid 下 cosine 低于 0.5 的 Dense 候选不再进入融合与重排；prepare 的检索、MTP SEARCH 与能力层的语义检索都经过它；
- Sparse 不受阈值约束，弱相关查询的结果可能更多来自 BM25 匹配；无意义查询在 Hybrid 下不保证返回空结果；
- 默认配置下 reranker 的候选从最多 5 条恢复为最多 20 条，最终结果可能改变，CPU 上 cross-encoder 的重排耗时随候选数增加；
- 未给出 `top_k` 的请求改由引擎级配置决定条数，默认 5 与现值一致。当前即 MTP SEARCH：它提交的 `RetrieveRequest` 不传 `top_k`，由契约默认值 5 填充。prepare 与被动链路的条数仍由 Gateway 检索计划给出（`gateway.default_top_k`，默认 5），不受引擎级配置影响；
- Hybrid 下请求的 `top_k` 大于 5 时，此前被融合截到 5，改后按请求条数返回；
- 召回数量与提示上下文会变化，需要以真实模型（BGE-M3 与 bge-reranker-base）和真实数据核对 0.5 不会滤掉相关记忆，并记录重排耗时。

不在本 Todo 范围：加权融合直接相加 cosine 与未归一化 BM25 分数的问题；默认配置使用 RRF，不受影响。

## 完成条件

阈值：

- 阈值只由 Dense 召回的配置决定：`DenseRetrieverConfig.score_threshold` 默认 0.5，`configs/config.yaml` 中 Dense 路径同为 0.5；`RetrievalEngine.retrieve` 与 `BaseMemoryRetriever.retrieve` 不再接收调用方阈值，删除引擎的 0.75 默认值；
- `SparseRetrieverConfig` 与 `HybridRetrieverConfig` 不再有 `score_threshold`；
- `_normalize_score` 的说明不再声称用于阈值过滤。

截取：

- `retrieval` 配置段新增引擎级 `default_top_k`（默认 5，`ge=1`，`configs/config.yaml` 中显式给出，供用户配置），由 `RetrievalEngine` 持有；引擎以请求的 `top_k`（未给出时为 `default_top_k`）作为本次结果条数，据此向重排器取候选数，并在重排阶段之后截取一次；
- “未给出”须贯通整条链路：`RetrieveRequest.top_k`（workspace contracts）、`RetrievalRequest.top_k`（core protocol）、Patchouli 本地 route `MEMORY_RETRIEVE`、`RetrievalFamiliar.retrieve` / `retrieve_async` 与 `RetrievalEngine.retrieve` 的 `top_k` 改为可缺省（`None`），删除各处硬编码的默认 5；`top_k=0` 的现有语义不变；
- RRF 与加权融合的配置不再有 `final_top_k`，全部融合入口返回完整的融合结果；Hybrid 按引擎请求的候选数截取后交给重排。

配置与测试：

- 上述被删除的配置键（Sparse 与 Hybrid 的 `score_threshold`、融合的 `final_top_k`）所在配置段均为 `extra="ignore"`，直接删除字段会静默忽略旧键，因此配置中出现这些键时启动显式报错（与 reranker 旧位置的处理一致；融合的 `final_top_k` 提示由 `retrieval.default_top_k` 取代），`configs/config.yaml` 同步修改；
- 单元测试覆盖：Dense 按配置阈值召回；引擎不向检索器传阈值；重排后不按分数过滤；启用重排且候选多于结果条数时，Hybrid 向重排交出引擎请求数量的候选；请求给出 `top_k` 时按它截取、未给出时按 `default_top_k` 截取，`NoopReranker` 下同样在重排位置之后截取；旧配置键被拒绝；现有向引擎或检索器传阈值、按融合 `final_top_k` 截取的单元与 e2e 用例按新语义改写；
- `test_empty_result_handling` 改为验证 Dense 召回阈值：低于阈值的候选不进入重排，全部低于阈值时结果为空。检索 e2e 以真实 Qdrant 与模型通过，并记录 0.5 阈值下的召回数量与重排耗时；
- [Patchouli 检索](../../patchouli/retrieval.md)第 5、9 节按最终代码更新阈值语义、候选截取与限制；[MTP 契约](../../contracts/mtp.md)与[子系统公共契约](../../contracts/subsystem-contracts.md)中 `RetrieveRequest` 的 `top_k=5` 默认值同步更新。

## 追踪

- 2026-10-04：修复检索 e2e 的事件循环问题后发现（分支 `refactor/identity-access-batch-2`）；未绑定版本或 Issue。
- 2026-10-12：重排序改为独立阶段后重新核对证据（分支 `fix/mypy-typing-baseline`）；同日 owner 确定阈值只作用于 Dense 召回、Sparse 不设阈值、Dense 默认 0.5。核对中发现存储层对 Sparse 阈值固定传 `None`，以及融合按 `final_top_k` 提前截取、使重排候选池失效；owner 决定融合后不设分数阈值、截取只在流程最后进行，并入本 Todo；随后决定引擎保留用户可配置的 `top_k`，只在请求未给出 `top_k` 时使用，重排阶段位置保持固定。

## 实施记录（2026-10-12）

- 按“完成条件”实施：Dense 阈值默认 0.5 且只由配置决定；Sparse、Hybrid 顶层与融合的阈值及融合 `final_top_k` 删除，旧键在配置加载时显式失败；融合返回全部结果；`retrieval.default_top_k`（默认 5）在请求未给出 `top_k` 时生效，`RetrieveRequest` / `RetrievalRequest` / 本地 route / Familiar / 引擎的 `top_k` 均可缺省；引擎只在重排之后截取一次。
- 单元测试覆盖完成条件所列行为；在副本中逐项回退实现（融合截到 5、忽略 `default_top_k`、Dense 阈值写死、Sparse 传阈值、不拒绝旧键、截取移到重排前），每项都有测试失败。
- 真实环境（本地 Qdrant 1.16、BGE-M3、bge-reranker-base、真实 LLM）下 e2e 与 real_infra 共 44 个全部通过；会话使用隔离 collection，开发数据未改动。
- 0.5 阈值的核对：检索 e2e 的召回条数为语义召回 4、关键词召回 2、混合冲突 1、精排 1、无关查询 0，检索指标 Recall@5 100%、MRR 0.833；无意义查询在不设阈值时召回 11 条，最高 cosine 0.3978，按阈值召回为空。实测内容贴切的记忆与相关查询的 cosine 为 0.63–0.70、与无意义查询为 0.32；以“测试用记忆”式标题与摘要写入的同一内容只有 0.35–0.39，低于阈值，pipeline e2e 的预埋记忆因此改为按生成记忆的写法填写标题与摘要。
- 耗时：e2e 中端到端检索（含重排，候选池 20）约 127 ms，其中混合召回约 34–46 ms。
- “不在本 Todo 范围”的加权融合尺度问题转入 [Patchouli 检索](../../patchouli/retrieval.md)第 9 节的当前限制。
