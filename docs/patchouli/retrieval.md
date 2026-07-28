---
title: Patchouli Retrieval
status: current
owner: patchouli
scope: memory-recall-ranking-and-read-side-effects
code_paths:
  - src/hivememory/engines/retrieval/
  - src/hivememory/patchouli/services/retrieval.py
  - src/hivememory/infrastructure/storage/vector_store.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/mtp.md
  - docs/architecture/boundaries.md
last_reviewed: 2026-07-28
---

# 记忆检索

Retrieval 的职责是从当前可检索书库中找出候选记忆，而不是把候选直接写进 prompt，也不是替 Agent 判断哪条事实最终正确。它返回 `MemoryAtom[]` 与搜索元信息；MemoryCompiler 再根据主动 chat、MTP READ、子 Agent 共享或 embedding 等用途编译文本视图。

这个分离修正了旧设计中的一个根本混淆：检索器应该优化“找什么、排多高”，编译器应该决定“给谁看、展示多少、用何种语言”。若 Retrieval 同时持有 renderer，任何 prompt 变化都会污染排序接口，其他消费者也只能重复实现一套记忆格式。

## 1. 当前链路

```text
RetrievalRequest
  -> RetrievalFamiliar
       -> QueryFilters(identity + business filters)
       -> RetrievalQuery
  -> RetrievalEngine
  -> Dense / Sparse / Hybrid Retriever
       -> Qdrant MidTermMemoryStore.search
       -> Fusion
       -> optional Reranker
  -> RetrievalResult
  -> RetrievalResponse(MemoryAtom[])
  -> caller-owned MemoryCompiler
```

`RetrievalEngine` 是很薄的执行层，只调用 retriever 并统计 latency/count。业务过滤、话题读取、Agent Profile 读取和生命周期副作用由 RetrievalFamiliar 承担。

## 2. 三种读取面

### 2.1 短期话题读取

RetrievalFamiliar 通过 MemoryLibrary.short_term 返回不可变 `TopicData` 和 `TopicSnapshot`。Gateway 使用按最近访问降序、默认排除空话题的 snapshot 进行路由；Patchouli prepare 和前端可以请求包含空话题的完整池。

读取话题可以选择 `touch`，控制是否刷新最近访问与 last-active；纯展示或内部构造应显式避免无意义 touch，防止观察行为改变 LRU 顺序。

### 2.2 中期记忆读取

中期读取包括 UUID、alias、scroll/list 和相关性搜索。Agent Profile 也以 `MemoryType.AGENT_PROFILE` 的 MemoryAtom 保存；别名缺失、类型不符或读取异常时，当前回退到内置 `OMNI_DOLL_PROFILE`。

### 2.3 长期归档读取

Familiar 可查询 archive records 或检查 `is_archived`，但普通检索只搜索中期 Qdrant。长期冷藏记忆不会自动混入召回，也不会因搜索 miss 自动 revive。

## 3. 身份与可见性过滤

RetrievalFamiliar 总是从 `RetrievalRequest.identity` 构造安全基线，调用方 filters 只能补充 memory type、tags 与 min confidence，不能替换 identity。

Qdrant 当前转换为：

```text
must user_id == current user
and (
  visibility == PUBLIC
  or visibility == WORKSPACE and team_id == current team
  or visibility == PRIVATE and source_agent_id == current agent
)
and optional memory_type
and optional min_confidence
```

这是一种“同一用户内的同心圆可见性”模型：PUBLIC 并不表示跨所有用户公开，user_id 仍是硬过滤。Workspace 依赖 team identity，Private 依赖 source agent identity。

当前 converter 尚未把 `tags` 与 `time_range` 转为 Qdrant 条件；它们出现在模型中，但不是已经生效的过滤能力。文档和 API 不能仅因字段存在就宣称完整支持。

## 4. Dense 与 Sparse 召回

DenseRetriever 使用语义向量相似度，并可应用：

- 基于 `updated_at` 与 half-life 配置的时间衰减惩罚；
- 基于 confidence 的轻量加成。

SparseRetriever 使用 Qdrant 稀疏向量，目标是保留函数名、错误码和专有实体等词汇级信号，不应用时间衰减。

默认 HybridRetriever 并行执行两路召回；任一路普通异常会退化为空结果，但 `StorageOfflineError` / `StorageReadError` 会重新抛出，使上层能够区分存储不可用与“确实没有相关记忆”。

`RetrievalQuery.keywords` 当前没有进入 `get_search_text()`，两路实际都使用 `semantic_query` 生成向量。换言之，Gateway 已能给出 keywords，但它们尚未形成独立 sparse query 输入，这是当前分析契约与检索实现之间的缺口。

## 5. Fusion 与 Rerank

Hybrid 先融合 Dense/Sparse `SearchResults`，再可选重排，最后截取 top-k。

当前 Fusion 有两种：

- Reciprocal Rank Fusion：按两路名次合并，适合不直接比较异构原始分数；
- Adaptive Weighted Fusion：按 debug/concept/timeline/brainstorm 模式组合 dense、sparse、time、confidence 和 vitality 因素。

默认配置使用 RRF。Reranker 可注入 cross-encoder service；未提供 service 时即使配置开启也会记录 warning 并关闭重排。CrossEncoderReranker 通过 MemoryCompiler 的 `DENSE_EMBEDDING` target 构造候选文本，避免维护另一套排序文本拼接。

分数阈值的语义当前并不完全统一：Dense/Sparse 在存储召回时接收阈值，Hybrid 融合后又根据 reranker 类型选择是否过滤；RRF score 与 cosine score 不能直接共享同一阈值。调用方应把 top-k 视为稳定控制，把跨策略 threshold 视为仍需收敛的实现细节。

## 6. 与 MemoryCompiler 的边界

Retrieval 只返回 atoms。当前主要调用者分别编译：

- Patchouli prepare：`RETRIEVAL_CONTEXT`；
- Passive Ingress：`RETRIEVAL_CONTEXT`；
- Koakuma MTP SEARCH/READ：retrieval envelope 或 `MTP_READ`；
- 子 Agent orchestration：`SHARED_CONTEXT_INJECTION`；
- Qdrant 与 reranker：dense/sparse embedding targets。

详细 target 与预算策略见[MemoryCompiler](./memory-compiler.md)。Retrieval 不应再次引入 full/cascade/compact renderer。

## 7. 访问与生命力副作用

检索结果产生与“记忆确实被使用”不是同一事实：

- `retrieve_async()` 会对返回 atoms 重新计算内存中的 vitality，当前 `persist=False`；
- 主动 chat finalize 对 prepare 阶段实际注入的去重 memory ids best-effort 记录 HIT；
- MTP 明确引用可走 citation route，触发生命周期强化；
- `update_access_stats()` 提供批量访问统计入口，但不是 RetrievalEngine 内部隐式副作用。

将副作用放在 Familiar/finalize 而非底层 retriever 中，可以避免预览、重试或内部排序无条件增加访问次数。

## 8. 失败与降级

Familiar 对 storage offline/read error 保持结构化异常；其他未知异常记录日志并返回空 response。这个边界让系统可以对“书库离线”禁用记忆能力，而对局部模型、过滤或排序异常保守地继续主用例。

空 response 因而有两种可能：确实无结果，或某个被 Familiar 吸收的普通异常。RuntimeEvent/日志仍是定位后者的必要证据，业务层不能把空列表当作完整健康证明。

## 9. 当前限制

- `tags`、`time_range` 尚未转换为 Qdrant filters；
- Gateway keywords 尚未进入独立 sparse query；
- Hybrid 子 retriever 虽有 enabled 开关，但关闭单路后当前 Hybrid 调用仍假定对象存在，非默认组合需要补齐 NoOp/分支处理；
- 跨 Dense、Sparse、RRF 与 reranker 的 threshold 口径未统一；
- 普通异常可能被投影为空结果，调用方只能通过观测区分；
- 普通检索不搜索长期 archive，也不自动 revive；
- 当前 retrieval response 主要暴露 atoms，`SearchResult.match_reason` 等解释元信息没有完整进入公共响应；
- 可见性基线已实现，但 Generation 的 dedup search 尚未复用同一 identity filter。

修复这些缺口时应优先保持身份硬过滤和 Retrieval/Compiler 解耦，不能为了快速接入一个新字段而把 prompt 或跨系统状态重新塞回 retriever。
