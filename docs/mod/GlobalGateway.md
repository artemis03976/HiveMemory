# 进阶模块开发计划：全局智能网关设计
**Advanced Development: Global Intelligent Gateway Design**

## 1. 背景与现状 (Background)

### 1.1 当前架构的痛点
在 MVP 及初期规划中，对用户 Query 的处理逻辑分散在多个独立的环节：
1.  **检索前 (Pre-Retrieval)**：Router 判断是否检索；QueryRewriter 进行重写方便检索。 -> **潜在的两次 LLM 调用，高延迟**。
2.  **生成前 (Pre-Generation)**：Extractor 内部或前置 Gating 判断是否值得存入记忆。 -> **重复计算，浪费 Token**。
3.  **感知层 (Perception)**：使用 Raw Query 进行话题连续性检测。 -> **准确度受指代不明的影响**。

### 1.2 升级目标
构建 **Global Gateway (全局网关)**，作为一个统一的入口处理单元。
实现 **“一次计算，多处复用” (Compute Once, Use Everywhere)**：通过单次 LLM 调用，同时产出意图、重写后的 Query、检索关键词以及记忆价值判定，显著降低系统延迟与运营成本。

## 2. 核心架构设计 (Core Architecture)

由于网关需要处理从 `Worker Agent` 发出的 `User Input` 到后台 `Librarian` 之间的两个独立异步路径，即：
- 记忆检索路径：为当次对话提供需要的相关记忆
- 记忆感知与生成路径：记录对话消息并在合适的时候进行生成处理
因此，网关应该作为一个独立的、顶层的“核心模块”存在，级别相当于当前项目中的Generation模块，Retrieval模块等，而不应该归属于感知层（Perception Layer）。

网关采用 **“漏斗式 (Funnel)”** 两级处理机制。

### 2.1 L1: 规则拦截器 (The Fast Pass)
*   **机制**：基于正则 (Regex) 和字符串匹配的零开销拦截。
*   **任务**：处理系统指令和极短的无效文本。
*   **逻辑**：
    *   命中 `/clear`, `/reset` -> **Intent: SYSTEM**。
    *   命中 "Hi", "你好", "谢谢" (Length < 5) -> **Intent: CHAT** (且标记为无记忆价值)。
*   **收益**：拦截约 20-30% 的无效 LLM 调用。

### 2.2 L2: 语义分析核心 (The Semantic Core)
*   **机制**：调用高响应速度的通用 LLM（Gateway Model）。
*   **任务**：在一个 Prompt 中同时完成以下三个子任务：
    1.  **意图分类 (Intent Classification)**：RAG / CHAT / TOOL。
    2.  **指代消解与重写 (Coreference Resolution)**：生成 `standalone_query`。
    3.  **元数据提取 (Extraction)**：生成检索关键词 (`keywords`) 和 记忆价值信号 (`save_signal`)。

## 3. 统一输出协议 (Unified Output Schema)

Gateway 不再输出简单的文本，而是输出一个**结构化指令包**。这将成为系统内部流转的标准协议。

```json
{
  "intent": "RAG" | "CHAT" | "TOOL",
  
  // 用于检索层 (Retrieval) 和 感知层 (Perception)
  "content_payload": {
    "rewritten_query": "如何将贪吃蛇游戏代码部署到 Ubuntu 服务器", // 指代消解后的完整句
    "search_keywords": ["贪吃蛇", "部署", "Ubuntu", "Server"], // 用于 Sparse Search
    "target_filters": { "type": "CODE_SNIPPET" } // (可选) 启发式过滤
  },
  
  // 用于记忆生成层 (Generation)
  "memory_signal": {
    "worth_saving": boolean, // 替代原本的 Gating 模块
    "reason": "用户询问具体技术部署方案，具有长期参考价值" // 解释
  }
}
```

## 4. 数据流转与复用 (Data Flow & Reuse)

Gateway 的输出将被下游模块“压榨”至干，没有任何字段是多余的。

### 4.1 Hot Path: 检索复用 (Retrieval Optimization)
*   **条件**：当 `intent == "RAG"`。
*   **动作**：
    *   直接使用 `content_payload.rewritten_query` 进行 **Dense Vector Search**。
    *   直接使用 `content_payload.search_keywords` 进行 **Sparse Vector Search**。
    *   **效果**：Worker Agent 获得极其精准的上下文，且无需再次等待重写模型。

### 4.2 Cold Path: 感知复用 (Perception Optimization)
*   **条件**：所有 Intent。
*   **动作**：将 `content_payload.rewritten_query` 异步推送到 Librarian 的缓冲区。
*   **效果**：
    *   感知层利用这个已重写的 Query 作为 **Semantic Anchor** (详见文档 DiscourseContinuity.md)。
    *   **解决痛点**：无需在感知层重新跑一遍指代消解，直接获得了精准的话题特征。

### 4.3 Cold Path: 生成复用 (Generation Optimization)
*   **条件**：异步处理缓冲区时。
*   **动作**：检查 `memory_signal.worth_saving`。
    *   `False` (如闲聊) -> **直接丢弃**，不调用昂贵的 Extractor 模型。
    *   `True` -> 进入 Extractor 流程。
*   **效果**：极大减少 Extractor (通常是高成本大模型) 的无效调用，显著降低 Token 账单。

## 5. 技术栈与服务选型 (Tech Stack)

Gateway 处于用户交互的阻塞路径上（Blocking Path），对 **延迟 (Latency)** 极其敏感，同时对 **指令遵循 (Instruction Following)** 能力要求较高（必须严格输出 JSON）。

| 维度 | 推荐方案 A (Cloud) | 推荐方案 B (Local) | 决策依据 |
| :--- | :--- | :--- | :--- |
| **模型** | **DeepSeek-V3** <br> **GPT-4o-mini** | **Qwen2.5-7B-Instruct** | **必须具备强指令遵循能力**。小于 7B 的模型很难稳定地在一个 Prompt 中完成多任务 JSON 输出。 |
| **部署** | LiteLLM Proxy | vLLM / Ollama | 需要高并发吞吐支持。 |
| **Prompt策略** | JSON Mode / Function Calling | Structured Output (Grammar) | 强制约束输出格式，防止解析失败。 |
| **预算预估** | 低 (~$0.15 / 1M tokens) | 中 (需 GPU 显存 > 8GB) | 根据隐私需求和硬件条件选择。 |

## 6. 开发优先级 (Implementation Priority)

1.  **Prompt 工程**：在 Playground 中反复调试 Gateway 的 System Prompt，确保它在处理模糊指代（"它是什么"）时能准确结合上下文（Last 3 Messages）进行重写。
2.  **网关模块封装**：为网关创建一个独立的模块目录 `gateway`，实现 `GlobalGateway` 类封装 L1 正则逻辑和 L2 LLM 调用逻辑。
3.  **下游适配**：
    *   **Retrieval 端**：移除旧的 `Router` 和 `QueryRewriter` 代码，改为接收 Gateway 的 `content_payload`。
    *   **Perception 端**：修改 `LogicalBlock` 的输入接口，接收 `rewritten_query`。
    *   **Generation 端**：移除旧的 `Gating` 模块，改为检查 `memory_signal`。

## 7. 架构图示 (Architecture Diagram)

```mermaid
graph TD
    User[用户输入] --> L1{L1: 规则拦截?}
    History[对话历史] -.-> Gateway
    
    L1 -- "Hi/Clear" --> Direct[透传 Worker]
    L1 -- Other --> Gateway[Global Gateway \n (LLM)]
    
    Gateway -->|JSON Output| Dispatcher[分发逻辑]
    
    %% 复用路径 1: 检索
    Dispatcher -- "Intent: RAG" --> Retrieval[检索层]
    Retrieval -->|使用 Rewritten + Keywords| VectorDB
    VectorDB --> WorkerAgent
    
    Dispatcher -- "Intent: CHAT" --> WorkerAgent
    
    %% 复用路径 2: 感知 (异步)
    Dispatcher -.->|Rewritten Query| LibrarianBuffer[Librarian 缓冲区]
    LibrarianBuffer -->|作为 Anchor| DriftCheck[连续性检测 (DiscourseContinuity.md)]
    
    %% 复用路径 3: 生成 (异步)
    Dispatcher -.->|Worth Saving?| GenerationGate{价值判断}
    GenerationGate -- Yes --> Extractor[记忆提取 (大模型)]
    GenerationGate -- No --> Discard[丢弃]
```