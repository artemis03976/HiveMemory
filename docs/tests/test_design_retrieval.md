# Retrieval Module Test Design

## 1. 测试目标与范围
本测试设计旨在验证 **Retrieval (检索引擎)** 从海量记忆中精准召回相关信息并以合适格式呈现的能力。
*   **测试范围**：
    *   **Hybrid Search (混合检索)**：验证结合 Dense Vector (语义) 和 Sparse Vector (关键词) 的检索效果，确保兼顾语义匹配和精确匹配。
    *   **Reranking (重排序)**：验证 Cross-Encoder 对粗排结果的精排能力，确保最相关的结果排在 Top-1。
    *   **Context Rendering (上下文渲染)**：验证不同渲染策略（Full, Cascade, Compact）输出格式的正确性。
*   **不包含**：
    *   Embedding 和 Rerank 模型的具体训练/微调效果（使用预训练模型）。
    *   Gateway 的查询预处理。

## 2. 测试环境与前置条件
*   **运行环境**：
    *   Python 3.10+
    *   `pytest`
*   **外部依赖**：
    *   **Qdrant Service**: 必须运行真实的 Qdrant 实例（Docker），预注入测试数据。
    *   **Embedding Model**: 本地 BGE-M3。
    *   **Rerank Model**: 本地 BGE-Reranker。
*   **数据准备**：
    *   `fixtures/golden_memories.json`: 一组覆盖不同领域、包含关键词冲突的标准记忆库。

## 3. 测试用例设计

### 3.1 混合检索测试 (Hybrid Search)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RET-HYB-001 | 纯语义召回 | 1. 搜索语义相关但无关键词重叠的 Query。<br>2. 验证 Top-K 结果。 | Query: "水果"<br>DB: "苹果", "香蕉", "汽车" | 召回 "苹果", "香蕉" | P0 |
| RET-HYB-002 | 纯关键词召回 | 1. 搜索包含特定专有名词（如错误拼写或生僻词）的 Query。<br>2. 验证 Top-K 结果。 | Query: "X-1024 参数"<br>DB: "X-1024 配置单", "X-1025" | 优先召回 "X-1024 配置单" | P0 |
| RET-HYB-003 | 混合冲突处理 | 1. 构造语义相关但关键词不匹配，与关键词匹配但语义无关的场景。<br>2. 验证 RRF 融合效果。 | Query: "苹果公司的股价" | 优先召回 "Apple Stock" (语义+词)<br>其次 "水果苹果" (仅词) | P1 |

### 3.2 重排序测试 (Reranking)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RET-RNK-001 | 精排优化 | 1. 构造粗排结果中 Top-1 并非最优的场景。<br>2. 执行 Rerank。<br>3. 验证排序变化。 | Candidates: [1. 弱相关, 2. 强相关]<br>Query: "..." | Result: [1. 强相关, 2. 弱相关] | P0 |
| RET-RNK-002 | 阈值过滤 | 1. 搜索无关 Query。<br>2. 执行 Rerank 并应用 `score_threshold`。<br>3. 验证结果数量。 | Query: "外星人"<br>DB: 只有地球数据 | Result: Empty list (所有结果分数均低于阈值) | P1 |

### 3.3 渲染测试 (Rendering)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RET-RND-001 | XML 格式渲染 | 1. 设置 Format=XML。<br>2. 渲染记忆列表。 | Memories: [M1, M2] | Output: `<system_memory_context>...<memory id="M1">...</memory>...</system_memory_context>` | P0 |
| RET-RND-002 | Markdown 格式渲染 | 1. 设置 Format=Markdown。<br>2. 渲染记忆列表。 | Memories: [M1] | Output: `## 相关记忆上下文\n\n### Title...` | P1 |
| RET-RND-003 | 瀑布式渲染 (Cascade) | 1. 设置 Max Token 限制。<br>2. 输入大量记忆。<br>3. 验证渲染层级。 | Memories: [M1(High Score), M2(Low Score)] | M1 渲染 Payload (全文)<br>M2 渲染 Index (仅摘要) | P1 |

## 4. 关键验证点
1.  **Qdrant 连通性**：验证 Client 能否正确连接、创建 Collection、并在测试结束后清理数据。
2.  **RRF 算法实现**：验证 Reciprocal Rank Fusion 的计算逻辑是否符合预期 `1 / (k + rank)`。
3.  **Token 计数准确性**：渲染器在截断或降级渲染时，计算的 Token 数量应接近真实 Tokenizer 的结果（误差 < 10%）。

## 5. 通过/失败标准
*   **P0 用例通过率**：100%
*   **召回率 (Recall@5)**：在 Golden Dataset 上 > 80%（语义+关键词）。
*   **MRR (Mean Reciprocal Rank)**：> 0.7。

## 6. 风险与假设
*   **假设**：本地环境有足够的内存运行 BGE-M3 和 Reranker 模型（约需 4-8GB VRAM/RAM）。
*   **风险**：如果测试机配置过低，需回退到 Mock Embedding/Rerank，但这会削弱 Retrieve 模块测试的真实性。建议在 CI/CD 中配置 GPU Runner 或使用轻量级模型 (BGE-Small) 进行 CI 测试。
