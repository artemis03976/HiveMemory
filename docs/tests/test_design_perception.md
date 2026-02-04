# Perception Module Test Design

## 1. 测试目标与范围
本测试设计旨在验证 **Perception (感知层)** 将连续的碎片化对话流转化为结构化语义块的核心能力。
*   **测试范围**：
    *   **LogicalBlock Construction (逻辑块构建)**：验证系统能否正确将 User -> Thought -> Tool -> Observation -> Assistant 的完整链路封装为一个 `LogicalBlock`。
    *   **Semantic Splitting (语义切分)**：验证 `SemanticBoundaryAdsorber` 根据语义相似度（High, Low, Grey）的决策逻辑。
    *   **Buffer Management (缓冲区管理)**：验证 `SemanticBufferManager` 对活跃/非活跃 Buffer 的维护，以及空闲超时机制。
    *   **Relay Mechanism (接力机制)**：验证当 Buffer 溢出或切分时，是否生成了包含摘要的 `Relay Context` 传递给下一阶段。
    *   **Workflow Integration (工作流集成)**：验证 Chatbot 和 Agent 模式下的完整消息流转。
*   **不包含**：
    *   上游 Gateway 的重写逻辑。
    *   下游 Generation 的记忆提取逻辑（仅验证触发时机）。

## 2. 测试环境与前置条件
*   **运行环境**：
    *   Python 3.10+
    *   `pytest`
*   **外部依赖**：
    *   **Embedding Model**: 使用本地轻量级模型 `all-MiniLM-L6-v2` 或 `BGE-M3` 计算语义向量。
    *   **Reranker**: 本地 BGE-Reranker (Mocked in Unit Tests)。
    *   **LLM Service**: Mock Summary 生成接口。
*   **数据准备**：
    *   `fixtures/patchouli_test_data.py`: 包含多领域对话（数据科学、Web开发、游戏开发、烹饪）、工具调用场景、长文本溢出场景等。

## 3. 测试用例设计

### 3.1 逻辑块构建与工作流测试 (Block Construction & Workflows)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PER-WFL-001 | Chatbot 基本对话流 | 1. 依次发送 User -> Assistant 消息。<br>2. 检查生成的 Block 结构。 | User: "Python列表推导式"<br>Assistant: "解释..." | Block 包含 1 User, 1 Assistant, 状态为 Closed | P0 |
| PER-WFL-002 | Agent 工具调用流 | 1. 依次发送 User -> Thought -> Tool -> Obs -> Assistant 消息。<br>2. 检查 Block 内部结构。 | User: "查天气"<br>Thought: "调用API"<br>Tool: `weather()`<br>Obs: "晴"<br>Assistant: "今天晴" | Block 包含完整的 `execution_chain` (Triplet 结构) | P0 |
| PER-WFL-003 | 多工具调用生成触发 | 1. 模拟 User -> (Call -> Obs -> Assistant) * 3。<br>2. 验证是否触发生成。 | 多轮工具调用场景 | 每一轮对话后 Buffer 更新，最终触发 Generation | P1 |

### 3.2 语义吸附测试 (Similarity >= 0.75)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PER-ADS-001 | 高相似度吸附 | 1. 初始化 Buffer (数据科学)。<br>2. 发送强相关新消息。<br>3. 验证是否吸附。 | Buffer: ["机器学习"]<br>New: "深度学习" (Sim > 0.8) | Action: `ADSORB` (吸附)，无 Flush | P0 |
| PER-ADS-002 | 边界值吸附 (0.75) | 1. 构造相似度恰好为 0.75 的 Case。<br>2. 验证是否吸附。 | Test Pair: "boundary_high" | Action: `ADSORB` | P1 |
| PER-ADS-003 | 连续同话题 | 1. 发送 6 条连续同话题消息。<br>2. 验证是否无漂移。 | DATA_SCIENCE_CONVERSATION | 无 Semantic Drift Flush，Block 正确累积 | P1 |
| PER-ADS-004 | 短文本强吸附 | 1. 发送短文本或语气词。<br>2. 验证是否忽略语义强制吸附。 | New: "好的" / "继续" | Action: `ADSORB` (忽略语义距离) | P1 |

### 3.3 语义漂移测试 (Similarity < 0.40)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PER-DRT-001 | 低相似度漂移 | 1. 初始化 Buffer (数据科学)。<br>2. 发送完全无关消息 (烹饪)。<br>3. 验证 FlushEvent。 | Buffer: ["机器学习"]<br>New: "红烧肉做法" (Sim < 0.2) | 触发 `FlushReason.SEMANTIC_DRIFT` | P0 |
| PER-DRT-002 | 边界值漂移 (0.39) | 1. 构造相似度为 0.39 的 Case。<br>2. 验证是否漂移。 | Test Pair: "low_similarity" | 触发 `FlushReason.SEMANTIC_DRIFT` | P1 |
| PER-DRT-003 | Flush 内容验证 | 1. 触发漂移。<br>2. 检查 `FlushEvent.blocks_to_flush`。 | 漂移场景 | FlushEvent 包含之前累积的所有 Blocks | P1 |
| PER-DRT-004 | 生成触发验证 | 1. 触发漂移。<br>2. 检查 MockGenerationEngine 调用。 | 漂移场景 | GenerationEngine.process 被调用 | P0 |

### 3.4 灰色区仲裁测试 (0.40 <= Similarity < 0.75)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PER-GRY-001 | 仲裁决定继续 | 1. 构造 Sim=0.55 场景。<br>2. Mock Arbiter 返回 True (相关)。 | Test Pair: "grey_area" | Action: `ADSORB`，无 Flush | P1 |
| PER-GRY-002 | 仲裁决定切分 | 1. 构造 Sim=0.55 场景。<br>2. Mock Arbiter 返回 False (不相关)。 | 数据科学 -> Web开发 | 触发 `FlushReason.SEMANTIC_DRIFT` | P1 |
| PER-GRY-003 | 灰色区边界测试 | 1. 测试 0.41 和 0.74 两个边界值。<br>2. 验证是否进入仲裁流程。 | Test Pair: "grey_boundary" | 都应调用 Arbiter 进行判断 | P2 |

### 3.5 缓冲区与接力测试 (Buffer & Relay)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PER-BUF-001 | 缓冲区 Flush 触发 | 1. 向 Buffer 写入消息直到达到 `max_tokens`。<br>2. 验证 FlushEvent。 | COMPACT_OVERFLOW_CONVERSATION | 触发 `FlushReason.TOKEN_OVERFLOW` | P0 |
| PER-BUF-002 | 接力摘要生成 | 1. 触发 Token Overflow Flush。<br>2. 检查 FlushEvent。 | 同上 | `relay_summary` 字段非空 | P0 |
| PER-BUF-003 | 溢出后继续 | 1. 触发溢出 Flush。<br>2. 继续发送新消息。<br>3. 验证 Buffer 状态。 | 溢出后追加关于 "异步编程" 的对话 | Buffer 重置但 Identity 保持，新消息被正确吸附 | P1 |

## 4. 关键验证点
1.  **状态机完整性**：在网络波动或乱序情况下，BlockBuilder 不应崩溃，应能处理孤立的 Observation。
2.  **Triplet 绑定**：Tool Call 和 Observation 必须严格一一对应，不能错位。
3.  **Topic Kernel 更新**：每次吸附新 Block 后，Buffer 的 `topic_kernel_vector` 应更新（通常采用滑动平均或加权更新），以反映话题的漂移。
4.  **阈值精度**：必须精确验证 0.40 和 0.75 两个临界值的行为，确保没有 Off-by-one error。

## 5. 通过/失败标准
*   **P0 用例通过率**：100%
*   **逻辑正确性**：生成的 `LogicalBlock` 必须符合 Schema 定义，无字段缺失。
*   **Mock 覆盖率**：Perception 层的所有外部依赖（Generation, Lifecycle, Arbiter）都应通过 Mock 验证其调用行为。

## 6. 风险与假设
*   **假设**：测试数据中的 Similarity Pair 经过人工校验，确实符合预期的相似度区间。
*   **风险**：Local Embedding (BGE-M3) 的计算结果可能随版本微调而变化，导致硬编码的阈值测试失败。建议在 `conftest.py` 中使用 `SimilarityInjector` 来 Mock 相似度计算结果，而非依赖真实 Embedding 模型，以保证测试的确定性。
