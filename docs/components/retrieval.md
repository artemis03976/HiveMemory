# 7 核心功能 IV：记忆检索与共享 (The Retrieval Engine)

> **\[归属分身：检索使魔 (Retrieval Familiar)]**
>
> 对应实现：`src/hivememory/engines/retrieval/`

本章定义 Worker Agent 如何访问 HiveMemory。Retrieval Engine 是一个纯粹的执行单元，不负责意图判断，专注于高效、精准的检索执行与上下文渲染。

***

## 7.0 模块概览 (Module Overview)

### 7.0.1 目录结构

```text
src/hivememory/engines/retrieval/
│  __init__.py          # 模块入口
│  engine.py            # RetrievalEngine — 编排器
│  retriever.py         # DenseRetriever / SparseRetriever / HybridRetriever / CachedRetriever
│  fusion.py            # ReciprocalRankFusion / AdaptiveWeightedFusion
│  reranker.py          # NoopReranker / CrossEncoderReranker
│  renderer.py          # FullContextRenderer / CascadeContextRenderer / CompactContextRenderer
│  filter_adapter.py    # QdrantFilterConverter — 过滤条件转换
│  interfaces.py        # 抽象接口层
│  models.py            # 数据模型 (RetrievalQuery, SearchResult, RetrievalResult, ...)
```

### 7.0.2 整体数据流

```
Gateway (intent=RAG)
    │
    ▼
RetrievalFamiliar.retrieve(RetrievalQuery)
    │
    ▼
RetrievalEngine
    ├── HybridRetriever.retrieve()
    │       ├── DenseRetriever  ──┐
    │       └── SparseRetriever ──┤ 并行召回
    │                             ▼
    │                    Fusion (RRF / AdaptiveWeighted)
    │                             │
    │                    Reranker (Noop / CrossEncoder)
    │                             │
    │                    SearchResults (Top-K)
    │
    └── ContextRenderer.render()
            ├── FullContextRenderer
            ├── CascadeContextRenderer
            └── CompactContextRenderer
                    │
                    ▼
            RetrievalResult { memories, rendered_context }
```

***

## 7.1 检索触发与输入 (Retrieval Trigger)

### 7.1.1 上游输入 (Input from Gateway)

检索动作由 **Global Gateway** 触发。当 `Gateway` 输出 `intent: "RAG"` 时，Retrieval Engine 收到检索请求并启动，基本输入封装为 `RetrievalQuery`：

```python
class RetrievalQuery(BaseModel):
    semantic_query: str          # 经指代消解的完整查询（用于向量检索）
    keywords: List[str]          # 提取的关键词（用于稀疏检索）
    filters: QueryFilters        # 结构化过滤条件
```

`QueryFilters` 支持以下过滤维度：

| 字段                | 类型                          | 说明                          |
| :---------------- | :-------------------------- | :-------------------------- |
| `memory_type`     | `MemoryType`                | 按记忆类型过滤（如仅查 `CODE_SNIPPET`） |
| `time_range`      | `Tuple[datetime, datetime]` | 时间窗口过滤                      |
| `tags`            | `List[str]`                 | 标签过滤                        |
| `source_agent_id` | `str`                       | 按来源 Agent 过滤                |
| `user_id`         | `str`                       | 用户隔离（多租户）                   |
| `min_confidence`  | `float`                     | 最低置信度阈值                     |

### 7.1.2 记忆工具化 (Memory as a Tool) — 主动查阅

除响应 Gateway 的被动检索外，Worker Agent 在执行任务过程中也可**主动调用工具**获取记忆，适用于复杂的多步推理场景。

- **工具定义**：`search_memory(query: str, filters: dict)`
- **场景**：Worker 收到 Gateway 注入的上下文后，发现缺少具体的密码加密库文档，主动调用 `search_memory(query="password encryption library")` 补充信息。

***

## 7.2 混合检索策略 (Hybrid Search Strategy)

HiveMemory 采用 **"双路并行召回 + 融合 + 可选精排"** 的混合检索策略，应对多维度的查询需求。

### 7.2.1 检索维度 I：语义向量检索 (Dense Retrieval)

`DenseRetriever` 利用 Index Layer（Title/Tags/Summary）的稠密向量进行语义匹配，捕获模糊语义关联。

- **机制**：Cosine Similarity（通过 Qdrant `mode="dense"`）
- **时间衰减**（可选）：$\text{score} = \text{vector\_score} \times (1 - \text{decay} \times 0.1)$，衰减函数为指数衰减 $e^{-\lambda t}$，半衰期可配置（默认 `time_decay_days`）。
- **置信度加权**（可选）：$\text{score} += \text{confidence} \times 0.05$，轻微提升高置信度记忆的排名。

### 7.2.2 检索维度 II：稀疏向量检索 (Sparse Retrieval)

`SparseRetriever` 使用 BGE-M3 的稀疏向量进行词汇级精准匹配，解决"特定函数名"或"错误码"无法通过向量精准召回的问题。

- **机制**：Sparse Vector Score（通过 Qdrant `mode="sparse"`）
- **场景**：用户查询 `"Fix KeyError in utils.py"`，稀疏检索能精准锁定包含 `KeyError` 和 `utils.py` 字符串的记忆原子。
- **注意**：稀疏检索不应用时间衰减，直接使用原始分数。

### 7.2.3 并行召回 (Parallel Recall)

`HybridRetriever` 默认通过 `ThreadPoolExecutor(max_workers=2)` 并行执行 Dense 和 Sparse 两路检索，任一路失败时以空结果兜底，不影响整体流程。可通过 `enable_parallel=False` 切换为顺序执行。

### 7.2.4 结构化过滤 (Structured Filtering)

过滤条件作为向量检索的 **Pre-filter（前置过滤）** 步骤，由 `QdrantFilterConverter` 将 `QueryFilters` 转换为 Qdrant 原生过滤器。

- **时间窗口**：`WHERE created_at BETWEEN X AND Y`
- **来源与类型**：`WHERE source_agent_id = 'coder'` / `WHERE type = 'CODE_SNIPPET'`

### 7.2.5 缓存检索器 (CachedRetriever)

`CachedRetriever` 是一个装饰器，为任意检索器添加内存缓存能力，减少重复检索开销。

- **缓存键**：`{semantic_query}_{top_k}_{score_threshold}`
- **过期策略**：TTL（默认 60 秒）+ FIFO 淘汰（默认最大 100 条）

***

## 7.3 融合算法 (Fusion)

融合层负责将多路检索结果合并为统一的排序列表。系统提供两种融合器，通过配置切换。

### 7.3.1 倒数排名融合 (RRF)

`ReciprocalRankFusion` 是默认融合器，对分数分布不敏感，能稳健地融合不同检索方式的结果。

**公式**：

$$\text{score}(d) = \sum\_i \frac{w\_i}{k + \text{rank}\_i(d)}$$

其中 $k=60$（常数），$w\_{dense}$ 和 $w\_{sparse}$ 为各路权重（可配置）。

**流程**：

1. Dense 路返回 Top-50，Sparse 路返回 Top-50。
2. RRF 按排名计算综合得分，取两路结果的并集。
3. 按 RRF 分数降序，截取 `final_top_k`（默认 20）。

同时提供 `fuse_multi()` 通用接口，支持任意多路结果的加权融合。

### 7.3.2 自适应加权融合 (AdaptiveWeightedFusion)

`AdaptiveWeightedFusion` 是进阶融合器，实现了"**相关性 × 质量**"的双因子排序，将记忆的置信度和生命力纳入最终评分。

**核心公式**：

$$S\_{final} = \underbrace{\sum\_i (w\_i \cdot S\_i)}_{\text{动态相关性}} \times \underbrace{\mathcal{M}(C, V)}_{\text{固有质量乘数}}$$

**左侧——动态相关性**：根据检索模式动态分配 Dense/Sparse 权重。

**右侧——固有质量乘数** $\mathcal{M}(C, V) = \text{Factor}_{conf}(C) \times \text{Factor}_{vit}(V)$：

| 因子                      | 条件                | 系数       | 说明         |
| :---------------------- | :---------------- | :------- | :--------- |
| $\text{Factor}\_{conf}$ | $C \ge 0.9$（用户验证） | 1.0      | 不惩罚        |
| $\text{Factor}\_{conf}$ | $C < 0.6$（LLM 推理） | 0.5（可配置） | 大幅降权，防幻觉污染 |
| $\text{Factor}\_{vit}$  | $V > 80$（高生命力）    | 1.2（可配置） | 小幅提权       |
| $\text{Factor}\_{vit}$  | $V < 30$（低生命力）    | 0.8（可配置） | 轻微降权       |

#### 四种预设检索模式 (Preset Retrieval Modes)

| 模式             | $w\_{dense}$ | $w\_{sparse}$ | 置信度惩罚     | 典型场景                                 |
| :------------- | :----------: | :-----------: | :-------- | :----------------------------------- |
| **debug**      |      0.3     |      0.9      | 强惩罚（严禁幻觉） | `"Fix KeyError in utils.py"`         |
| **concept**    |      0.8     |      0.2      | 弱惩罚       | `"How does the auth system work?"`   |
| **timeline**   |      0.4     |      0.3      | 中等        | `"What did we discuss last Friday?"` |
| **brainstorm** |      0.6     |      0.1      | 无惩罚（鼓励发散） | `"Any ideas for optimization?"`      |

模式通过 `fuse(mode="debug")` 显式指定，或通过 `fuse_with_intent(query_intent)` 由意图关键词自动推断（`fix/error → debug`，`explain/how → concept`，`when/history → timeline`，`idea → brainstorm`）。

> **当前状态**：`AdaptiveWeightedFusion` 已完整实现，包括四种预设模式和质量乘数计算。意图自动推断目前为规则映射，预留了接入 LLM 或可训练 MoE 模型的扩展接口（`fuse_with_intent`）。

***

## 7.4 精排 (Reranking)

融合后可选地对 Top-K 结果进行精排，进一步提升相关性。

### 7.4.1 NoopReranker

透传实现，不做任何处理，直接返回融合结果。默认使用。

### 7.4.2 CrossEncoderReranker

使用 Infrastructure 层提供的 `BaseRerankService`（默认为 `FlagReranker`，基于 BGE-Reranker 模型）对融合结果进行精排。

**流程**：

1. 截取 Top-K 候选（性能优化，避免对全量结果精排）。
2. 构建 `[query, passage]` 对，`passage` 为记忆的稠密嵌入文本表示。
3. 批量调用 `service.compute_score(pairs)` 获取原始分数。
4. 通过 Sigmoid 函数将原始分数（约 -10 到 +10）标准化到 0-1（可配置关闭）。
5. 按新分数降序重排，更新 `match_reason` 记录原始分数与精排分数。

***

## 7.5 上下文注入策略 (Context Injection Strategy)

检索结果不直接 Dump 为 JSON 注入 Prompt，而是通过 **ContextRenderer** 转换为 LLM 易读的、Token 经济的格式。

### 7.5.1 渲染格式

支持两种输出格式，通过 `RenderFormat` 枚举切换：

- **XML**（推荐）：使用 `<system_memory_context>` / `<memory_block>` / `<memory_ref>` 标签包裹，Claude/GPT-4 表现最佳。
- **Markdown**：使用 `##` 标题和 `---` 分隔符，适合 Markdown 友好的模型。

两种格式均包含头部说明（告知 Agent 这是历史记忆）和尾部指令（提示验证旧/未验证记忆）。

### 7.5.2 三种渲染策略

#### 策略 A：全量加载 (FullContextRenderer)

按相关性顺序强制注入每条记忆的完整 Payload，达到 `max_tokens` 字符上限时硬截断。

- **适用场景**：Token 预算充足，或检索结果较少（< 3 条）的简单场景。
- **配置**：`max_tokens`（默认字符上限）、`max_content_length`（单条内容截断长度）、`stale_days`（超过多少天标记为 Old）。

#### 策略 B：瀑布式降级 (CascadeContextRenderer)

在信息完整性与 Token 预算之间取得平衡，是最推荐的生产策略。

**算法逻辑**：

1. 设定 `max_memory_tokens` Token 预算水位线。
2. **Top-N**（`full_payload_count`，默认 1）：强制注入完整 Payload，确保最相关的信息不丢失。
3. **其余结果**：尝试注入完整 Payload；预算紧张时自动**降级为 Index 视图**（仅摘要 + 标签）。
4. **预算耗尽**：停止注入。

**懒加载支持**（`enable_lazy_loading=True`）：降级为 Index 视图时，自动附加工具提示：

```xml
<memory_ref id="2" type="FACT">
    [标签]: #legal #disclaimer
    [摘要]: Standard disclaimer text for EU clients.
    [提示]: 使用 read_memory("mem_2") 获取完整内容
</memory_ref>
```

Agent 浏览摘要后，可主动调用 `read_memory(id)` 按需加载完整内容，避免一次性注入过多 Token。

#### 策略 C：紧凑模式 (CompactContextRenderer)

默认**仅渲染 Index 视图**（摘要 + 标签），不渲染任何完整 Payload，配合懒加载工具使用，达成渐进式披露的效果。

- **适用场景**：Token 极其昂贵的长上下文场景，或复杂多步推理任务。
- **优势**：Agent 先浏览"菜单"，再按需点"菜"，天然防止因模糊记忆导致的幻觉（不确定时发起 Tool Call，而非胡编）。

### 7.5.3 渲染器工厂

通过 `create_renderer(config)` 工厂函数根据配置类型自动创建对应渲染器：

```python
# FullRendererConfig    → FullContextRenderer
# CascadeRendererConfig → CascadeContextRenderer
# CompactRendererConfig → CompactContextRenderer
renderer = create_renderer(config)
```

### 7.5.4 元数据的语义化翻译

渲染时将复杂元数据翻译为自然语言提示，降低 Agent 的理解成本：

- **置信度**：`confidence < 0.6` → 渲染为 `[Unverified]` 标记。
- **时效性**：超过 `stale_days` 天未更新 → 渲染为 `(Warning: Old)` 标记。
- **版本历史**：若包含 `history_summary`，渲染为 `"Current State (Upgraded from v1 on 2025-05-20)"` 形式。

***

## 7.6 权限与隔离 (Visibility & Scopes)

### 7.6.1 同心圆作用域 (Concentric Scopes)

所有记忆原子在写入时由 Patchouli 自动打上 Scope 标签，检索时强制过滤：

1. **PRIVATE（私有层）**：仅当前 Agent 可见。用于中间思考过程、草稿、失败尝试，防止"脏数据"污染团队视野。
2. **WORKSPACE（领域层）**：共享给特定职能小组或项目空间。用于 API 接口定义（前后端共享）、世界观设定（写作组共享）等。
3. **GLOBAL（全局层）**：该用户下所有 Agent 全员可见。用于用户偏好、项目全局配置、最终交付的高置信度成果。

### 7.6.2 透视逻辑 (The Perspective Logic)

Worker Agent 检索时看到**多个作用域的并集**：

```sql
WHERE
    (visibility = 'GLOBAL')
    OR (visibility = 'WORKSPACE' AND workspace_id = current_agent.workspace_id)
    OR (visibility = 'PRIVATE' AND source_agent_id = current_agent.id)
```

### 7.6.3 知识晋升机制 (Knowledge Promotion)

记忆权限不是一成不变的，而是流动的：

- **默认**：新生成的记忆默认为 `PRIVATE` 或 `WORKSPACE`。
- **晋升触发**：当 Librarian 发现某条 `PRIVATE` 记忆被成功运行且多次引用时，自动将其 Visibility 升级为 `GLOBAL`。
- **隐喻**：个人的经验（Private）经过验证后，变成了团队的标准作业程序（Global）。

### 7.6.4 MVP 阶段实施方案

暂缓实现 L2 Workspace 层，仅保留 L1 和 L3：

- **Public by Default**：绝大多数"结论性"记忆（代码、事实）直接设为 `GLOBAL`。
- **Private for Noise**：仅将思考链和报错日志设为 `PRIVATE`。
- **实现方式**：在 Qdrant Payload 中增加 `visibility` 字段即可，无需建立多张表。

***

## 7.7 数据模型参考 (Data Models)

### RetrievalQuery

```python
class RetrievalQuery(BaseModel):
    semantic_query: str          # 语义查询文本（用于向量检索）
    keywords: List[str]          # 关键词列表（用于稀疏检索）
    filters: QueryFilters        # 结构化过滤条件
```

### SearchResult / SearchResults

```python
class SearchResult(BaseModel):
    memory: MemoryAtom
    score: float                 # 最终分数（RRF / AdaptiveWeighted / Rerank）
    match_reason: str            # 匹配原因（用于可解释性）
    vector_score: float          # 原始向量相似度
    boost_applied: float         # 应用的加权值

class SearchResults(BaseModel):
    results: List[SearchResult]
    total_candidates: int        # 初始候选数量
    latency_ms: float            # 检索耗时
```

### RetrievalResult

`RetrievalEngine` 的统一输出，面向上层业务（`RetrievalFamiliar`）：

```python
class RetrievalResult(BaseModel):
    memories: List[MemoryAtom]   # 便于后续业务处理
    rendered_context: str        # 可直接注入 Prompt 的上下文字符串
    latency_ms: float
    memories_count: int
    search_results: Optional[SearchResults]  # 原始检索结果（用于调试）
```

***

## 7.8 配置参考 (Configuration)

| 配置项                             | 默认值           | 说明                                   |
| :------------------------------ | :------------ | :----------------------------------- |
| `dense.enabled`                 | `true`        | 是否启用稠密检索                             |
| `dense.top_k`                   | `20`          | 稠密检索召回数量                             |
| `dense.score_threshold`         | `0.5`         | 稠密检索相似度阈值                            |
| `dense.enable_time_decay`       | `false`       | 是否启用时间衰减                             |
| `dense.enable_confidence_boost` | `false`       | 是否启用置信度加权                            |
| `sparse.enabled`                | `true`        | 是否启用稀疏检索                             |
| `sparse.top_k`                  | `20`          | 稀疏检索召回数量                             |
| `fusion.type`                   | `rrf`         | 融合算法（`rrf` / `adaptive_weighted`）    |
| `fusion.rrf_k`                  | `60`          | RRF 常数                               |
| `fusion.final_top_k`            | `20`          | 融合后保留数量                              |
| `reranker.enabled`              | `false`       | 是否启用精排                               |
| `reranker.top_k`                | `10`          | 精排候选数量                               |
| `reranker.normalize_scores`     | `true`        | 是否 Sigmoid 标准化精排分数                   |
| `renderer.type`                 | `cascade`     | 渲染策略（`full` / `cascade` / `compact`） |
| `renderer.max_memory_tokens`    | `2000`        | Token 预算上限                           |
| `renderer.full_payload_count`   | `1`           | 强制完整渲染的 Top-N 数量                     |
| `renderer.enable_lazy_loading`  | `false`       | 是否启用懒加载工具提示                          |
| `renderer.lazy_load_tool_name`  | `read_memory` | 懒加载工具名称                              |

