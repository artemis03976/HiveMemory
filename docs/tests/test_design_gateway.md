# Gateway Module Test Design

## 1. 测试目标与范围
本测试设计旨在验证 **Gateway (网关层)** 作为系统流量入口的核心能力。
*   **测试范围**：
    *   **Intent Classification (意图识别)**：验证系统能否准确区分 RAG、CHAT、TOOL 和 SYSTEM 意图。
    *   **Query Rewriting (查询重写)**：验证系统能否结合上下文进行指代消解，生成语义完整的 `rewritten_query`。
    *   **Keyword Extraction (关键词提取)**：验证系统能否从查询中提取出用于稀疏检索的核心名词。
    *   **Interceptor Logic (拦截器逻辑)**：验证 L1 正则拦截器的优先处理机制。
*   **不包含**：
    *   下游 Perception、Retrieval 的具体处理逻辑。

## 2. 测试环境与前置条件
*   **运行环境**：
    *   Python 3.10+
    *   `pytest` 测试框架
*   **外部依赖**：
    *   **LLM Service**: 使用真实的 `LiteLLMService` 调用 LLM 模型得到 Gateway 信息。
*   **数据准备**：
    *   `fixtures/gateway_cases.json`: 包含多组 `(query, context) -> (expected_intent, expected_rewritten)` 的测试数据集。

## 3. 测试用例设计

### 3.1 意图识别测试 (Intent Classification)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GW-INT-001 | 显式检索意图识别 | 1. 构造包含明确询问 factual 信息的 Query。<br>2. 调用 Gateway `process` 接口。 | Query: "Rust 的所有权机制是什么？"<br>Context: [] | Intent: `GatewayIntent.RAG`<br>Rewritten: 包含 "Rust", "所有权" | P0 |
| GW-INT-002 | 闲聊意图识别 | 1. 构造简单的打招呼或情感表达 Query。<br>2. 调用 Gateway `process` 接口。 | Query: "今天天气不错"<br>Context: [] | Intent: `GatewayIntent.CHAT` | P1 |
| GW-INT-003 | 系统指令识别 | 1. 构造符合系统指令格式的 Query。<br>2. 调用 Gateway `process` 接口。 | Query: "/clear"<br>Context: [Any] | Intent: `GatewayIntent.SYSTEM` | P0 |
| GW-INT-004 | 模糊意图处理 | 1. 构造既像闲聊又像询问的 Query。<br>2. 验证系统倾向性。 | Query: "你觉得 Python 怎么样？"<br>Context: [] | Intent: `GatewayIntent.RAG` (倾向于检索观点) 或 `CHAT` (取决于 Prompt 调优，需固定预期) | P2 |

### 3.2 查询重写与指代消解测试 (Query Rewriting)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GW-RW-001 | 单轮指代消解 | 1. 构造上下文包含主体的对话。<br>2. 发送带有代词的 Query。<br>3. 检查重写结果。 | Context: [User: "介绍下 Docker"], [Assistant: "..."]<br>Query: "它怎么安装？" | Rewritten: "Docker 怎么安装？"<br>Intent: `RAG` | P0 |
| GW-RW-002 | 跨多轮指代消解 | 1. 构造多轮对话上下文。<br>2. 发送带有模糊指代的 Query。 | Context: [User: "A项目"], [AI: ...], [User: "B项目"], [AI: ...] <br>Query: "前者是用什么语言写的？" | Rewritten: "A项目是用什么语言写的？" | P1 |
| GW-RW-003 | 无需重写保持原样 | 1. 发送语义完整、无上下文依赖的 Query。 | Query: "介绍一下 Kubernetes" | Rewritten: "介绍一下 Kubernetes" (保持原意) | P1 |

### 3.3 关键词提取测试 (Keyword Extraction)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GW-KW-001 | 英文技术名词提取 | 1. 发送包含特定技术栈的 Query。 | Query: "如何在 FastAPI 中使用 Pydantic？" | Keywords: ["FastAPI", "Pydantic"] | P1 |
| GW-KW-002 | 中文实体提取 | 1. 发送包含具体人名或地名的 Query。 | Query: "鲁迅的《狂人日记》讲了什么？" | Keywords: ["鲁迅", "狂人日记"] | P1 |

### 3.4 拦截器测试 (Interceptors)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GW-L1-001 | 正则拦截优先于 LLM | 1. 配置 `/clear` 正则拦截器。<br>2. 发送 `/clear`。<br>3. 监控 Mock LLM 调用次数。 | Query: "/clear" | Intent: `SYSTEM`<br>LLM Call Count: 0 | P0 |

## 4. 关键验证点
1.  **Context 注入正确性**：验证传给 LLM 的 Context 是否按时间倒序排列，且格式符合 Prompt Template 要求。
2.  **JSON 解析鲁棒性**：当 LLM 返回非标准 JSON 或包含 Markdown 代码块时，Parser 是否能正确解析。
3.  **Fallback 机制**：当 LLM 调用失败或超时，Gateway 是否能返回默认的安全结果（如 Intent=CHAT, Rewritten=Original Query）。

## 5. 通过/失败标准
*   **P0 用例通过率**：100%
*   **整体通过率**：> 90%
*   **性能指标**：L1 拦截处理时间 < 10ms (不含网络开销)。

## 6. 风险与假设
*   **风险**：真实 LLM (如 GPT-4 vs GPT-3.5) 对复杂指代消解的能力差异可能导致 E2E 测试通过但线上效果不佳。需在 Stage 3 系统测试中引入真实 LLM 验证。
