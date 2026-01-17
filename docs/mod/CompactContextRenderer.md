# 进阶模块开发计划：自适应上下文注入与优化系统
**Module Plan: Adaptive Context Injection & Optimization System**

## 1. 模块概览 (Overview)

### 1.1 背景与痛点
在传统的 RAG 系统中，检索到的 Top-K 文档往往存在两个极端问题：
1.  **冗余挤占 (Redundancy)**：检索出三段内容高度相似的代码片段（如 V1, V2, V3 版本），导致 Token 浪费且信息量低。
2.  **上下文溢出 (Overflow)**：为了覆盖更多信息，检索了过多的 Full Payload，导致挤占了 Agent 的推理空间（Reasoning Space），甚至冲刷掉了 System Prompt 中的核心指令。

### 1.2 核心理念
拒绝引入昂贵且高延迟的“专用压缩 LLM”。利用 HiveMemory 独特的 **“冰山模型 (Iceberg Model)”** 数据结构，采取以下策略：
*   **数学去重 (Mathematical De-duplication)**：利用 MMR 算法在向量层面解决冗余。
*   **分级渲染 (Tiered Rendering)**：利用 `Index` 层作为天然的“压缩视图”，利用 `Payload` 层作为“详情视图”。
*   **按需加载 (Lazy Loading)**：默认只给“菜单”，Agent 需要时再点“菜”。

---

## 2. 核心策略 (Core Strategies)

### 2.1 策略 A: MMR 多样性重排 (Maximal Marginal Relevance)
**目标**：解决“检索结果同质化”问题。
**机制**：在向量检索阶段，不再单纯依据“与 Query 的相似度”排序，而是综合考虑“与 Query 的相似度”和“与已选结果的差异性”。

*   **公式**：
    $$ \text{Score} = \lambda \cdot \text{Sim}(Query, Doc) - (1-\lambda) \cdot \max \text{Sim}(Doc, SelectedDocs) $$
*   **配置**：
    *   推荐 $\lambda = 0.7$（兼顾相关性与多样性）。
    *   **效果**：如果 Top-1 是 `parse_date_v2.py`，Top-2 绝不会是 `parse_date_v1.py`（因为太像了），而可能是 `legal_disclaimer.txt`（差异大）。

### 2.2 策略 B: 动态预算截断 (Dynamic Budgeting)
**目标**：解决“上下文溢出”问题，适用于 MVP 阶段。
**机制**：设定 Token 预算水位线，对记忆进行“瀑布式”降级渲染。

*   **算法逻辑**：
    1.  设定 `Max_Memory_Tokens = 2000`。
    2.  按相关性排序检索结果 $[M_1, M_2, ... M_n]$。
    3.  **Top-1 (最相关)**：强制注入完整 `Payload` (Content)。
    4.  **其余 ($M_2 \dots M_n$)**：
        *   尝试注入 `Payload`。
        *   如果超出预算 $\rightarrow$ **降级为注入 `Index` (Summary + Tags)**。
        *   如果连 Summary 都塞不下 $\rightarrow$ 停止注入。

### 2.3 策略 C: 懒加载与引用 (Lazy Loading / Skill-Style)
**目标**：解决“复杂任务的信息过载”问题，适用于高级 Agent。
**机制**：默认仅提供“记忆索引视图”，赋予 Agent 主动查阅的工具。

*   **注入形态 (Prompt View)**：
    ```markdown
    <memory_index>
    I have found references that might help. To read full content, use tool `read_memory(id)`.
    
    1. [ID: mem_01] "Python Date Utils" (Tags: #code #datetime)
       > Summary: Standard implementation of ISO8601 parsing.
    2. [ID: mem_05] "System Auth Protocols" (Tags: #security)
       > Summary: Updated OAuth2 flows for 2025.
    </memory_index>
    ```
*   **配套工具**: `read_memory(id: str)`
    *   Agent 思考: *"用户想写鉴权代码，mem_01 没用，mem_05 看起来很关键，我要读一下。"*
    *   Action: `read_memory("mem_05")`
    *   System: 下一轮对话注入 `mem_05` 的完整 Payload。

---

## 3. 渲染器工作流 (Renderer Workflow)

本模块将作为当前记忆检索模块的 ContextRenderer 的进阶实现而存在。

```mermaid
graph TD
    A[Raw Retrieval Results (Top-50)] --> B{Apply MMR Filter}
    B -- Reduced to Top-10 (Diverse) --> C[Budget Check Loop]
    
    C --> D{Is Top-1 Result?}
    D -- Yes --> E[Render FULL Payload]
    D -- No --> F{Check Token Budget}
    
    F -- Budget Available --> E
    F -- Budget Low --> G[Render INDEX Only (Summary+Tags)]
    F -- Budget Empty --> H[Stop Injection]
    
    E --> I[Assemble System Prompt]
    G --> I
    
    I --> J[Inject Definition of 'read_memory' Tool]
```

---

## 4. 开发计划与分阶段实施

由于本模块仅影响到Renderer的行为，与其余模块没有交互逻辑，因此在 Renderer 已有完善逻辑的情况下即可进一步实现。

### 实现计划
*   **任务 1**: 在 Qdrant Client 中启用 MMR 搜索模式。
    *   *Action*: 将 `search_type="similarity"` 改为 `search_type="mmr"`, `lambda_mult=0.7`。
*   **任务 2**: 实现 **策略 B (动态预算)**。
    *   *Action*: 为全局编写能够计算 token 数量的工具类/函数，实现瀑布式降级逻辑。确保至少有一条完整记忆被注入。
*   **任务 3**: 实现 **策略 C (懒加载)**。
    *   *Action*:
        1.  修改 Prompt 模板，支持仅渲染列表模式。
        2.  在 Worker Agent 的工具集中注册 `read_memory_artifact` 函数。
        3.  调试 Agent 的 System Prompt，教会它：“看到 Summary 觉得不够时，不要瞎编，去调用 `read_memory`”。

---

## 5. 优势总结 (Why this works)

1.  **零延迟 (Zero Latency)**: MMR 和字符串截断都是毫秒级操作，没有任何额外的 LLM 推理开销。
2.  **天然防幻觉 (Anti-Hallucination)**: 通过 `Index` (摘要) 先行，Agent 知道“有什么”，但不知道“具体是什么”，迫使它在不确定时发起 Tool Call，而不是根据模糊的记忆胡编。
3.  **架构复用 (Architecture Reuse)**: 完美利用了 3.1 节设计的 `Index` (轻量) vs `Payload` (重量) 分离存储结构，无需修改数据库 Schema。

这份计划与主文档 PROJECT.md 的 **5.2 上下文注入策略** 相辅相成，是其更进阶的工程落地指南。