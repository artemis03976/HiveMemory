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
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-10-18
---

# 记忆检索

Retrieval 的职责是从当前可检索书库中找出候选记忆，而不是把候选直接写进 prompt，也不是替 Agent 判断哪条事实最终正确。它返回 `MemoryAtom[]` 与搜索元信息；MemoryCompiler 再根据主动 chat、MTP READ、子 Agent 共享或 embedding 等用途编译文本视图。

每个公共 `RetrievalRequest` 必须携带授权点组装的 `IdentityScope`。Patchouli 在公共应用服务边界把它转换为内部 `RetrievalQuery`，独立保存 `belong_to: WorkspaceIdentity` 与 `from_actor: ActorIdentity`；Familiar、Engine、Retriever 和 MemoryLibrary 不再传递 scope。读取先限定请求 Workspace 的资源归属，再应用记忆自身的 actor read policy；身份坐标与资源寻址以[Workspace 架构](../architecture/workspace.md)为准。

这个分离修正了旧设计中的一个根本混淆：检索器应该优化“找什么、排多高”，编译器应该决定“给谁看、展示多少、用何种语言”。若 Retrieval 同时持有 renderer，任何 prompt 变化都会污染排序接口，其他消费者也只能重复实现一套记忆格式。

## 1. 当前链路

```text
RetrievalRequest
  -> MemoryManagementService（公共 owner 边界拆分 scope）
       -> RetrievalQuery(belong_to, from_actor, business filters)
  -> RetrievalFamiliar
  -> RetrievalEngine
       -> 召回：Dense / Sparse / Hybrid Retriever
            -> Qdrant MidTermMemoryStore.search
            -> Fusion（仅 Hybrid）
       -> 重排：Reranker（CrossEncoderReranker；关闭时 NoopReranker）
       -> 截取：请求的 top_k，缺省为 retrieval.default_top_k（只在这里进行一次）
  -> RetrievalResult
  -> RetrievalResponse(MemoryAtom[])
  -> caller-owned MemoryCompiler
```

`RetrievalEngine` 是很薄的执行层：编排召回与重排两个阶段，在重排之后做唯一一次截取，并统计 latency/count。业务过滤、话题读取、Agent Profile 读取和生命周期副作用由 RetrievalFamiliar 承担。

## 2. 三种读取面

### 2.1 短期话题读取

RetrievalFamiliar 通过 MemoryLibrary.short_term 返回不可变 `TopicData` 和 `TopicSnapshot`。Gateway 使用按最近访问降序、默认排除空话题的 snapshot 进行路由；Patchouli prepare 和前端可以请求包含空话题的完整池。

读取话题是纯读操作（无 touch 语义）：访问追踪与 LRU 顺序由 PerceptionFamiliar 编排的 `TopicWorkingSet` 维护，Store 快照不承载访问时间。手动结算必须指定 Topic，Patchouli 不保存“最近活跃话题”。

### 2.2 中期记忆读取

中期读取包括 UUID、alias、scroll/list 和相关性搜索，内部方法显式接收归属与发起者。Agent Profile 也以 `MemoryType.AGENT_PROFILE` 的 MemoryAtom 保存。只有未指定 Profile 时才允许使用内置 `OMNI_DOLL_PROFILE` fallback；`default` / `omni_doll` 是对该内置 Profile 的直接选择。显式自定义 alias 在公共边界必须携带 `IdentityScope`，拆分后先通过 Workspace ownership hard filter，再执行 Workspace 内的 PUBLIC / TEAM / PRIVATE actor policy；缺失、越权、类型不符、配置无效或存储失败都显式返回对应结构化错误。

### 2.3 长期归档读取

Familiar 可查询 archive records 或检查 `is_archived`，但普通检索只搜索中期 Qdrant。长期冷藏记忆不会自动混入召回，也不会因搜索 miss 自动 revive。

## 3. 身份与可见性过滤

安全基线在公共 owner 边界从 `RetrievalRequest.identity_scope` 拆出；prepare 等内部编排直接用已经拆分的归属与发起者构造查询。调用方 filters 只能补充 memory type、source agent 与 min confidence 等业务维度，不能替换身份或 Workspace 边界。

Qdrant 当前先要求记忆的 `workspace_identity` 与请求 Workspace 完全匹配；归属只由 canonical owner/workspace 投影字段表达，对旧 main-workspace 记录的受控兼容分支已随存量迁移完成删除（不再存在 `meta.user_id` OR 分支与 legacy visibility 分支）。通过 ownership hard filter 后，再按记忆的 actor read policy 选择：`PUBLIC` 对 Workspace 内所有已获准执行者可读，`TEAM` 仅目标 team 可读，`PRIVATE` 仅目标 agent 可读。因而 `PUBLIC` 也不表示跨 Workspace 或跨用户公开。

QdrantStorageAdapter 对 UUID、alias、scroll 与 search 返回的每个原子重验归属与读取策略，不能仅信任向量服务的过滤结果。`enforce_actor_visibility=False` 只用于已经授权的管理读取，预过滤与返回校验均跳过 actor policy，Workspace hard boundary 和业务过滤仍生效；Agent 检索与 Profile 解析不使用该开关。来源字段和贡献者筛选只是业务过滤，不构成读取授权。

当前 converter 尚未把 `tags` 与 `time_range` 转为 Qdrant 条件；它们出现在模型中，但不是已经生效的过滤能力。文档和 API 不能仅因字段存在就宣称完整支持。

## 4. Dense 与 Sparse 召回

DenseRetriever 使用语义向量相似度，并可应用：

- 按 `score_threshold`（默认 0.5）在存储召回时过滤原始 cosine 相似度，这是检索链路中唯一的分数阈值；
- 基于 `decay_anchor_at`（与生命力评分共用的衰减基准）与 half-life 配置的时间衰减惩罚；
- 基于 confidence 的轻量加成。

时间影响当前只由 DenseRetriever 的 `enable_time_decay` 与 `time_decay_days` 解释。Fusion 不再暴露没有消费者的独立 `time_weight`，以免把未来的复习调度或时间线排序误写成当前能力。

SparseRetriever 使用 Qdrant 稀疏向量，目标是保留函数名、错误码和专有实体等词汇级信号，不应用时间衰减，也不设分数阈值：BM25 原始分没有上界，阈值会依赖语料。

默认 HybridRetriever 并行执行两路召回；任一路普通异常会退化为空结果，但 `StorageOfflineError` / `StorageReadError` 会重新抛出，使上层能够区分存储不可用与“确实没有相关记忆”。取消等非 `Exception` 的 `BaseException` 同样向上传播，不会被当作某一路的检索结果交给融合。

Hybrid 的 dense/sparse 两路可以分别在配置中关闭；关闭的一路由 `NoOpRetriever`（总是返回空结果）顶替，Hybrid 始终持有两个检索器并照常融合，不在调用处判空。两路同时关闭没有召回来源，配置加载时显式失败。

`RetrievalQuery.keywords` 当前没有进入 `get_search_text()`，两路实际都使用 `semantic_query` 生成向量。换言之，Gateway 已能给出 keywords，但它们尚未形成独立 sparse query 输入，这是当前分析契约与检索实现之间的缺口。

## 5. Fusion 与 Rerank

Hybrid 融合 Dense/Sparse `SearchResults` 后，按调用方请求的候选数截取；融合本身返回全部结果，既不截取，也不设分数阈值。

当前 Fusion 有两种：

- Reciprocal Rank Fusion：按两路名次合并，适合不直接比较异构原始分数；
- Adaptive Weighted Fusion：按 debug/concept/timeline/brainstorm 模式组合 dense、sparse、confidence 和 vitality 因素。

默认配置使用 RRF。

重排序不属于任何检索器，而是 RetrievalEngine 在召回之后的独立阶段：引擎先按重排器的 `candidate_count(top_k)` 向检索器索取候选，交给重排器后再截取 top-k。重排能独立于召回策略启用，dense、sparse 与 hybrid 都接收同一个重排阶段；检索器只负责“找什么”，不感知后续是否精排。配置位于 `patchouli.retrieval.reranker`，与 `retriever` 并列；写在旧位置 `retrieval.retriever.reranker` 时配置加载显式失败并提示迁移，而不是被静默忽略。

重排阶段始终存在：`retrieval.reranker` 关闭时由 `NoopReranker` 顶替，候选数等于 top-k、结果原样返回；配置开启但未装配重排服务时同样退化为 `NoopReranker` 并记录 warning。CrossEncoderReranker 以配置的 `top_k`（默认 20）作为候选池，通过 MemoryCompiler 的 `DENSE_EMBEDDING` target 构造候选文本，避免维护另一套排序文本拼接。

召回、融合与重排的分数尺度互不相同，同一个阈值数字只能对其中一种有意义，因此分数阈值只作用于 Dense 召回，交给重排的候选即粗筛后的结果：

- Sparse 不设阈值（见第 4 节）；
- 融合分数不设阈值：RRF 只反映名次，无意义查询的第 1 名与相关查询的第 1 名得分相同；加权融合直接相加 cosine 与 BM25 原始分，尺度随查询变化；
- 重排分数不设阈值：CrossEncoderReranker 用 sigmoid 把分数映射到 0–1，只用于排序与展示。

阈值不由调用方传入，引擎与检索器接口都只接收条数。本次结果条数是请求的 `top_k`；未给出时（当前为 MTP SEARCH）使用 `patchouli.retrieval.default_top_k`（默认 5）。prepare 与被动链路的条数来自 Gateway 检索计划（`gateway.default_top_k`），不受该配置影响。引擎按结果条数向重排器索取候选池，并且只在重排之后截取一次。

已删除的配置键在出现时使配置加载显式失败，而不是被静默忽略：`retrieval.retriever.sparse.score_threshold`（以及 Sparse 检索器的 `score_threshold`）、Hybrid 顶层的 `score_threshold`、融合配置的 `final_top_k`（由 `retrieval.default_top_k` 取代）。

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

普通检索传播中立的存储故障、未知程序异常与取消；dense/sparse/hybrid 的失败不伪装为空成功。Profile 缺失/不可见使用 `ResourceNotFoundError`，已找到但类型或配置无效使用 `ProfileResolutionError`，由 CALL 或 CPU 分配边界各自处理，不回退 Omni-Doll。这个边界让调用方区分书库离线、配置损坏与真正无命中；分类见[错误模型](../contracts/error-model.md)。

空 response 表示成功读取后没有命中，或调用策略显式跳过检索；健康判断使用独立的 health 契约，不能从一次空结果推断依赖状态。

## 9. 当前限制

- `tags`、`time_range` 尚未转换为 Qdrant filters；
- Gateway keywords 尚未进入独立 sparse query；
- Hybrid 下 Sparse 不受阈值约束，弱相关查询的结果可能主要来自 BM25 命中，无意义查询不保证返回空结果；
- 加权融合直接相加 cosine 与未归一化的 BM25 原始分，尺度随查询变化（默认配置使用 RRF，不受影响）；
- 普通异常可能被投影为空结果，调用方只能通过观测区分；
- 普通检索不搜索长期 archive，也不自动 revive；
- 当前 retrieval response 主要暴露 atoms，`SearchResult.match_reason` 等解释元信息没有完整进入公共响应；
- Workspace ownership hard filter 与 actor policy 已由 Memory store/adapter 统一执行；Generation 的 dedup search 同样独立传入归属与发起者，SETTLE 使用无 Team 的保留 system actor，只能查重本域 PUBLIC 记忆（legacy record 兼容已随存量迁移删除）。

修复这些缺口时应优先保持身份硬过滤和 Retrieval/Compiler 解耦，不能为了快速接入一个新字段而把 prompt 或跨系统状态重新塞回 retriever。
