# Generation Module Test Design

## 1. 测试目标与范围
本测试设计旨在验证 **Generation (生成引擎)** 从原始对话流中提取高价值记忆并维护记忆库一致性的能力。
*   **测试范围**：
    *   **Memory Extraction (记忆提取)**：验证 LLM 能否从 `LogicalBlock` 中提取出符合 Schema 的 `MemoryAtom` 草稿，并过滤无意义信息。
    *   **Deduplication & Evolution (去重与演化)**：验证 `MemoryDeduplicator` 能否正确执行 TOUCH、UPDATE、CREATE 决策，并正确合并新旧记忆。
    *   **Schema Validation (格式校验)**：验证生成的 JSON 数据结构是否合法，关键字段是否存在。
*   **不包含**：
    *   Perception 层的分块逻辑。
    *   Storage 层的具体存储实现（Qdrant/File）。

## 2. 测试环境与前置条件
*   **运行环境**：
    *   Python 3.10+
    *   `pytest`
*   **外部依赖**：
    *   **LLM Service**: 使用真实的 `LiteLLMService` 调用 LLM 模型得到记忆提取结果。
    *   **Vector Store**: 必须运行真实的 Qdrant 实例（Docker），用于存入记忆原子
*   **数据准备**：
    *   `fixtures/generation_blocks.json`: 待处理的 LogicalBlock 列表。
    *   `fixtures/existing_memories.json`: 预置在库中的标准记忆原子。

## 3. 测试用例设计

### 3.1 记忆提取测试 (Extraction)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GEN-EXT-001 | 标准信息提取 | 1. 输入包含事实性信息的对话 Block。<br>2. 调用 `extract` 接口。 | User: "我的API Key是 sk-123"<br>AI: "已记录" | Draft: 包含 "API Key" 相关 Title, Content, Summary | P0 |
| GEN-EXT-002 | 噪音过滤 (Denoising) | 1. 输入无营养的闲聊 Block。<br>2. 调用 `extract` 接口。 | User: "好的，谢谢"<br>AI: "不客气" | Draft: `None` 或 `has_value=False` | P0 |
| GEN-EXT-003 | 复杂结构提取 | 1. 输入包含代码片段的 Block。<br>2. 验证提取内容的完整性。 | User: "Python 冒泡排序怎么写？"<br>AI: [Code Block] | Draft: Content 中应完整包含代码块，Tags 包含 "Python", "Algorithm" | P1 |

### 3.2 去重与决策逻辑测试 (Deduplication Logic)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GEN-DED-001 | 决策：CREATE (新记忆) | 1. Mock 检索结果为空或相似度极低 (<0.75)。<br>2. 输入新 Draft。<br>3. 检查决策结果。 | Draft: "Rust 教程" | Action: `DeduplicationAction.CREATE` | P0 |
| GEN-DED-002 | 决策：TOUCH (仅更新时间) | 1. Mock 检索到相似度 > 0.95 且内容几乎一致的记忆。<br>2. 输入 Draft。<br>3. 检查决策结果。 | Existing: "PyTorch 安装"<br>Draft: "PyTorch 安装" (完全一致) | Action: `DeduplicationAction.TOUCH`<br>Target ID: Existing ID | P1 |
| GEN-DED-003 | 决策：UPDATE (知识演化) | 1. Mock 检索到相似度 > 0.95 但内容有增量，或相似度在 0.75-0.95 之间。<br>2. 输入 Draft。<br>3. 检查决策结果。 | Existing: "会议时间 10点"<br>Draft: "会议时间改到 11点" | Action: `DeduplicationAction.UPDATE`<br>Target ID: Existing ID | P0 |

### 3.3 记忆合并测试 (Merger)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GEN-MRG-001 | 内容追加合并 | 1. 执行 UPDATE 操作。<br>2. 验证合并后的 Content。 | Old: "A"<br>New: "B" | Merged Content: "A\n\n## 更新 [Date]\nB" | P0 |
| GEN-MRG-002 | 标签并集 | 1. 执行 UPDATE 操作。<br>2. 验证合并后的 Tags。 | Old: ["AI", "Tech"]<br>New: ["Tech", "Code"] | Merged Tags: ["AI", "Tech", "Code"] | P1 |
| GEN-MRG-003 | 摘要更新 | 1. 执行 UPDATE 操作。<br>2. 验证 Summary 选择策略（通常保留较长或较新的）。 | Old: "Short"<br>New: "Long Description..." | Merged Summary: "Long Description..." | P2 |

## 4. 关键验证点
1.  **JSON Schema 鲁棒性**：LLM 可能返回字段缺失的 JSON，Extractor 必须有校验和修复机制（或重试）。
2.  **死循环避免**：在 Merge 过程中，确保不会因为不断追加内容导致单条记忆无限膨胀（应有最大长度限制或触发切分，虽然当前版本可能未实现，需作为 Risk 记录）。
3.  **置信度加权**：验证合并后的 Confidence Score 是否按权重更新（如 `0.6 * Old + 0.4 * New`）。

## 5. 通过/失败标准
*   **P0 用例通过率**：100%
*   **数据一致性**：所有生成的 MemoryAtom 必须包含合法的 `vector` (虽由 Embedding 模块生成，但 Generation 需确保调用链路通畅)。

## 6. 风险与假设
*   **假设**：Prompt 能够有效指导 LLM 进行去噪，不会误删重要信息。
*   **风险**：对于“相似但不相同”的微妙差异（如版本号 v1.0 vs v2.0），Embedding 相似度可能极高导致误判为 TOUCH 而非 UPDATE，导致信息丢失。需通过 Rerank 或 LLM Double Check 来缓解（当前版本依赖阈值）。
