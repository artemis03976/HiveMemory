# 进阶模块开发计划：自适应加权检索与质量融合系统
**Module Plan: Adaptive Weighted Retrieval & Quality Fusion System**

## 1. 模块概览 (Overview)

### 1.1 背景与目标
在基础 RAG 系统中，检索通常仅依赖单一的向量相似度（Vector Similarity），容易出现以下问题：
*   **语义漂移**：在需要精确匹配变量名时，向量检索可能返回语义相关但拼写不同的错误结果。
*   **质量忽视**：检索器无法区分“高置信度的即时代码”与“低置信度的错误尝试”，导致幻觉污染。
*   **时效性弱**：无法根据 Query 的时态需求（如“上周的”）动态调整时间权重。

本模块旨在构建一个 **“带价值观的动态检索器”**。它不仅仅匹配**相关性 (Relevance)**，还通过内生变量评估记忆的 **质量 (Quality)**，实现“相关性 $\times$ 质量”的双因子排序。

### 1.2 核心理念
*   **动态路由 (Dynamic Routing)**：借鉴 MoE 思想，根据 Query 意图动态分配不同检索路（语义、关键词、时间、图谱）的权重。
*   **价值观排序 (Value-driven Ranking)**：将记忆的“置信度（Confidence）”和“生命力（Vitality）”作为惩罚或奖励因子，直接影响最终排序。

---

## 2. 核心算法逻辑 (Core Algorithm)

最终评分公式定义如下：

$$S_{final} = \underbrace{\sum (w_i \cdot S_i)}_{\text{Contextual Relevance (动态相关性)}} \times \underbrace{\mathcal{M}(C, V)}_{\text{Intrinsic Quality (固有质量乘数)}}$$

### 2.1 左侧：动态相关性 (Contextual Relevance)
根据 Query 意图生成权重向量 $\mathbf{w} = [w_{dense}, w_{sparse}, w_{time}, w_{graph}]$。

*   **$S_{dense}$ (语义分)**: Dense Vector Cosine Similarity.
*   **$S_{sparse}$ (关键词分)**: BM25 / Sparse Vector Score.
*   **$S_{time}$ (时间分)**: 基于时间窗口的衰减函数或重叠度。
*   **$S_{graph}$ (图谱分)**: (预留) 知识图谱跳数距离。

### 2.2 右侧：固有质量乘数 (Intrinsic Quality Multiplier)
由记忆原子的元数据（Payload）决定，作为硬性约束。

$$ \mathcal{M}(C, V) = \text{Factor}_{conf}(C) \times \text{Factor}_{vit}(V) $$

*   **$\text{Factor}_{conf}$ (置信度惩罚)**:
    *   若 $C \ge 0.9$ (User Verified): 系数 $= 1.0$
    *   若 $C < 0.6$ (LLM Inferred): 系数 $= 0.5$ (大幅降权)
*   **$\text{Factor}_{vit}$ (生命力增益)**:
    *   若 $V > 80$ (High Vitality): 系数 $= 1.2$ (小幅提权)
    *   若 $V < 30$ (Low Vitality): 系数 $= 0.8$

---

## 3. 架构组件设计 (Component Architecture)

### 3.1 意图权重生成器 (Intent Weight Generator)
*   **依托组件**: Librarian Agent (Patchouli) 或 专门的小模型。
*   **输入**: User Query + Context Summary。
*   **输出**: 权重配置。
*   **Prompt 示例**:
    > "Analyze the user's query. Assign weights (0.0-1.0) to the following retrieval aspects:
    > - Semantic (Concept/Meaning)
    > - Keyword (Exact name/Error code)
    > - Time (Recency/Specific date)
    > Output JSON."

### 3.2 多路检索执行器 (Multi-Path Executor)
*   **依托组件**: Qdrant / Python Middleware。
*   **逻辑**: 并行执行 Dense 和 Sparse 检索，获取候选集 ID 的并集 (Union)。

### 3.3 融合评分引擎 (Fusion Scoring Engine)
*   **依托组件**: Python 后端逻辑 (Scoring Function)。
*   **职责**:
    1.  从 Qdrant Fetch 候选集的 Payload (包含 $C$ 和 $V$ 分数)。
    2.  应用上述 $S_{final}$ 公式进行重算分。
    3.  执行排序并截取 Top-K。

---

## 4. 预设检索模式 (Preset Retrieval Modes)

为了简化决策难度，我们可以预定义几种典型的权重模板 (Profiles)：

| 模式名称 | $w_{dense}$ | $w_{sparse}$ | $w_{time}$ | 惩罚策略 | 典型场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Debug Mode** | 0.3 | **0.9** | 0.1 | **强惩罚** (严禁幻觉) | "Fix KeyError in utils.py" |
| **Concept Mode** | **0.8** | 0.2 | 0.1 | 弱惩罚 | "How does the auth system work?" |
| **Timeline Mode** | 0.4 | 0.3 | **0.8** | 中等 | "What did we discuss last Friday?" |
| **Brainstorm** | **0.6** | 0.1 | 0.0 | **无惩罚** (鼓励发散) | "Any ideas for optimization?" |

---

## 5. 开发路线与实施步骤 (Implementation Roadmap)

本模块建议在 **Phase 3 (生命周期管理)** 完成后开始实施，因为依赖于 `Vitality Score` 的存在。

### Step 1: 数据准备 (Data Readiness)
*   **任务**: 确保 Qdrant 的 Payload Schema 中已正确写入 `confidence` 和 `vitality` 字段。
*   **检查**: 运行脚本检查现有记忆的元数据完整性。

### Step 2: 权重生成器开发 (Weight Generator)
*   **任务**: 调试相关模块的Prompt或是可训练模型，使其能根据 Query 准确分类到上述“预设模式”之一，或输出自定义权重。
*   **验证**: 输入 "Fix error 404"，Router 应输出 `Debug Mode` 或高 $w_{sparse}$。

### Step 3: 融合逻辑编码 (Fusion Logic Coding)
*   **任务**: 在 fusion.py 中编写新的 Python 类 `AdaptiveQualityFusion`。
*   **核心代码**: 实现加权求和与乘法惩罚逻辑。
*   **验证**: 单元测试——构造一条“高语义相关但低置信度”的假数据，验证其在 Debug 模式下是否被正确沉底。

### Step 4: 调优与评估 (Tuning)
*   **任务**: 调整 $\text{Factor}_{conf}$ 的惩罚力度。
*   **测试**: 使用真实对话历史，对比“开启融合评分”与“仅向量检索”的 Top-5 结果质量。

---

## 6. 未来扩展 (Future Expansion)

*   **图谱检索接入 ($S_{graph}$)**: 当知识图谱模块上线后，将其作为第四个加项引入公式。
*   **可学习的 MoE (Trainable MoE)**: 如果积累了足够的用户反馈数据（用户点击了哪条记忆），可以使用机器学习模型（如 XGBoost 或 MLP）来进行意图权重生成，从用户 query 中自动生成动态权重 $\mathbf{w}$。