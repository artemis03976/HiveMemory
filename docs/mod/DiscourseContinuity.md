# 进阶模块开发计划：感知层升级——增强型话题连续性检测
**Advanced Perception Layer: Enhanced Topic Continuity Detection**

## 1. 背景与现状 (Background)

### 1.1 当前问题：语义漂移误判
在 MVP 阶段的测试中，系统采用简单的余弦相似度（Cosine Similarity）来判断对话流是否应该切分。这导致了 **“逻辑连贯但语义不连贯”** 场景下的误判：
*   **False Break (错误切分)**：用户从“编写贪吃蛇代码”转向“部署服务器”。虽然任务逻辑是连续的（开发 -> 运维），但由于 Embedding 向量空间距离较远，系统将其误判为新话题，导致记忆割裂。
*   **False Merge (错误合并)**：单纯基于关键词或短文本的匹配，容易将无关的短语（如“好的”、“继续”）错误吸附到当前话题，引入噪音。

### 1.2 升级目标
构建一套**“基于意图与指代 (Intent & Reference based)”** 的检测机制，作为单纯的文本相似度计算的增强附属模块。目标是实现像人类一样理解“任务流”的连续性，而非“词汇”的相似性。

## 2. 核心架构变动 (Core Architecture Changes)

本次升级不改变 `LogicalBlock` 的物理结构，而是改变计算相似度时的**“输入特征”**和**“判定逻辑”**。

### 2.1 引入“语义锚点” (Semantic Anchor)
不再使用整个 Block 的 `content_text`（包含冗长的 LLM 回复）进行 Embedding，而是构建精简的锚点。

*   **输入依赖**：依赖上游（Global Gateway）传入的 `Rewritten_Query`（已完成指代消解）。
*   **上下文桥接 (Context Bridge)**：为了解决跨域任务（Dev -> Ops）的连接问题，Anchor 可选包含上一轮的上下文摘要。
*   **构建公式**：
    ```python
    # Buffer 中上一轮 Agent 回复的简短摘要 (由 Extractor 或 简单截断生成)
    last_context = Buffer.last_block.summary or Buffer.last_block.content[:100]
    
    # 组合锚点
    anchor_text = f"Context: {last_context}\nQuery: {rewritten_query}"
    ```

### 2.2 引入“灰度仲裁”机制 (The Grey Area Arbiter)
摒弃单一的相似度阈值判定，采用 **“双阈值筛选 + 模型仲裁”** 的漏斗机制。利用 System 2 的异步特性，用少量的计算成本换取极高的切分准确率。

## 3. 详细处理流程 (Processing Workflow)

当一个新的 `LogicalBlock` 到达感知层缓冲区时，执行以下三级流水线：

### Step 1: 启发式强吸附 (Heuristic Filtering)
*   **逻辑**：检测“非信息性”短文本。
*   **规则**：
    *   如果 `rewritten_query` 长度 < 5 token（如“继续”、“是的”）。
    *   或者属于预定义的 Stop Words 列表（“不对”、“报错了”）。
*   **动作**：直接 **FORCE_ADSORB (强制吸附)**，跳过后续计算。并不更新当前 Buffer 的 `Topic_Kernel_Vector`（防止噪音污染）。

### Step 2: 向量初筛 (Vector Screening)
*   **计算**：
    *   $V_{new} = \text{Embedding}(anchor\_text)$
    *   $V_{kernel} = \text{Buffer.current\_topic\_vector}$
    *   $Score = \text{Cosine}(V_{new}, V_{kernel})$
*   **判定**：
    *   **Case High ($Score > 0.75$)**: 语义强相关。 -> **ADSORB (吸附)**。
    *   **Case Low ($Score < 0.40$)**: 语义完全无关。 -> **SPLIT (切分)**。
    *   **Case Grey ($0.40 \le Score \le 0.75$)**: 模糊地带（如“写代码”转“部署”）。 -> **进入 Step 3**。

### Step 3: 智能仲裁 (Intelligent Arbitration)
*   **触发条件**：仅在 Step 2 命中 Grey Area 时触发。
*   **执行者**：**Local SLM (本地小模型)** 或 **Cross-Encoder**（优先）。
*   **任务**：二分类逻辑判断。
*   **Prompt / Input 逻辑**：
    > "判断以下两个意图是否属于同一个任务流？
    > 上文任务: {Buffer_Summary}
    > 新请求: {rewritten_query}
    > 只要存在因果、递进、指代关系，即视为同一任务。
    > 输出: YES / NO"
*   **动作**：
    *   YES -> **ADSORB (吸附)**。
    *   NO -> **SPLIT (切分)**。

## 4. 技术栈与服务选型 (Tech Stack)

鉴于感知层运行在后台（Cold Path），且频率较高，必须控制成本。

| 组件 | 选型建议 | 备注 |
| :--- | :--- | :--- |
| **Embedding 模型** | **BGE-M3** (Local) | 继续复用现有本地模型，开销极低。 |
| **仲裁官 (Arbiter)** | **BGE-Reranker-v2-m3** <br> 或 **Qwen2.5-1.5B** | **新增组件**。推荐优先尝试 Reranker，它本质是计算句子对分数的 Cross-Encoder，比 LLM 更快更准，完全不需要生成能力。 |
| **输入源** | 依赖 Gateway | 必须确保上游传入的 Query 已经过重写，否则本层效果大打折扣。 |

## 5. 数据流图 (Data Flow)

```mermaid
graph TD
    Input[新 LogicalBlock] -->|提取 Rewritten Query| Heuristic{Step 1: 极短文本?}
    
    Heuristic -- Yes --> Adsorb[强制吸附 \n(不更新 Kernel)]
    Heuristic -- No --> AnchorGen[构建 Anchor Text \n(Context + Query)]
    
    AnchorGen --> Embed[Embedding 计算]
    Embed --> VectorCheck{Step 2: 相似度?}
    
    VectorCheck -- "> 0.75 (High)" --> AdsorbUpdate[吸附 & 更新 Kernel]
    VectorCheck -- "< 0.40 (Low)" --> Split[切分 & 归档旧 Buffer]
    
    VectorCheck -- "0.4 - 0.75 (Grey)" --> Arbiter[Step 3: 本地仲裁官]
    
    Arbiter -- "相关 (YES)" --> AdsorbUpdate
    Arbiter -- "无关 (NO)" --> Split
```

## 6. 开发优先级 (Implementation Priority)

1.  **对接上游**：修改 `LogicalBlock` 结构，增加 `rewritten_query` 和 `anchor_text` 字段，确保能接收 Gateway 的输出。
2.  **实现 Context Bridge**：编写简单的逻辑，从 Buffer 中提取上一轮 Assistant 回复的前 100 个字符作为 Context。
3.  **接入 Reranker/SLM**：在本地环境部署一个小型的 Cross-Encoder 模型，写一个简单的 Python 函数 `check_coherence(text_a, text_b)`。
4.  **集成测试**：使用“贪吃蛇 -> 部署”这类跨域对话用例进行测试，验证是否不再发生错误切分。

---

## 7. 现状落地对照

当前代码库已具备“语义流感知层 + 漂移检测”的基础实现，但与本文的 v2.0 升级方案存在差距，建议在文档中显式标注，便于后续迭代对齐：

- **已实现（Baseline）**
  - 语义流感知层：`SemanticFlowPerceptionLayer`，[semantic_flow_perception_layer.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/perception/semantic_flow_perception_layer.py)
  - 漂移判定：`SemanticBoundaryAdsorber.should_adsorb()`（短文本强吸附 + embedding 相似度阈值），[semantic_adsorber.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/perception/semantic_adsorber.py)
  - 锚点文本：当前为 `LogicalBlock.user_block.content`，[models.py](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/perception/models.py)
- **尚未实现（本文重点）**
  - 依赖网关的 `rewritten_query` 作为语义锚点（见 [InternalProtocol_v2.0.md](file:///c:/Users/29305/Projects/HiveMemory/docs/mod/InternalProtocol_v2.0.md)）
  - Context Bridge 公式（Context + Query）与双阈值 Grey Area 仲裁（Step 3 Arbiter）
