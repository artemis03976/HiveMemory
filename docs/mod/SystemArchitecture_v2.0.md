# HiveMemory 系统架构演进与未来开发规划 (v2.0)

**版本**: 2.0 
**核心目标**: 解决在 MVP 阶段后的多次模块增补与重构后，造成的系统延迟问题与帕秋莉的职能分配不合理。重新调整系统的逻辑架构，构建高鲁棒性的记忆基础设施。

## 1. 架构重构：帕秋莉体系 (The Patchouli System)

为了解决记忆检索需求的“实时检索”与记忆生成的“异步管理”的本质在工程实现上的冲突，同时保持“帕秋莉作为总管理者”的设计初衷，我们将系统架构重构为 **分布式智能体系**。帕秋莉不再是一个单一的后台进程，而是由多个职能组成的有机整体。

### 1.1 概念模型：三位一体 (The Trinity Aspect)

| 分身名称 | 对应模块实现 | 所在层级 | 核心职责 | 特性 |
| :--- | :--- | :--- | :--- | :--- |
| **真理之眼 (The Eye)** | **Global Gateway** | **交互层 (Interaction)** | 意图识别、查询重写、流量分发 | **同步阻塞**、极低延迟、小模型驱动 |
| **检索使魔 (The Familiar)** | **Retrieval Engine** | **热处理层 (Hot Path)** | 混合检索、重排序、上下文渲染 | **同步阻塞**、高并发、本地计算密集 |
| **大图书馆本体 (The Core)** | **Perception Layer + Memory Generation + Lifecycle** | **冷处理层 (Cold Path)** | 话题感知、记忆生成、生命周期管理 | **异步非阻塞**、高智商、SOTA 模型驱动 |

### 1.2 顶层数据流架构

```mermaid
graph TD
    User[用户输入] --> Eye[真理之眼 (Gateway)]
    
    %% 真理之眼 - 分发
    Eye -->|1. Hot Signal: Rewritten Query| Familiar[检索使魔 (Retrieval)]
    Eye -->|2. Cold Signal: Query Anchor| Core[本体 (Perception)]
    
    %% 检索使魔 - 提供记忆服务
    Familiar -->|读取索引| DB[(Memory Store)]
    Familiar -->|注入 Context| Worker[Worker Agent]
    
    %% 帕秋莉本体 - 对记忆管理
    subgraph "Librarian Core (System 2)"
        Core -->|话题漂移检测| Buffer[逻辑块缓冲]
        Buffer -->|提取与摘要| Generator[记忆生成]
        Generator -->|写入| DB
        Lifecycle[生命周期 Gardener] -->|维护| DB
    end
```

---

## 2. 核心组件详解 (Component Detail)

### 2.1 帕秋莉·真理之眼 (The Eye of Patchouli) —— 对应 Global Gateway
> **定位**：**守门人与感知者**。系统的全局阀门/路由，负责“一次计算，多处复用”。

*   **设定**：帕秋莉通常坐在图书馆深处，但她释放了一个“魔法之眼”悬浮在门口（交互层最前端）。
*   **职能**：
    *   这只眼睛负责**第一时间审视**所有进来的访客（用户消息）。
    *   它速度极快，负责判断：“这个访客是来闲聊的（Chat），还是来查阅禁书的（RAG）？”
    *   它负责将访客模糊的请求翻译成图书馆通用的咒语（Rewriting）。
*   **工程实现**：就是我们设计的 Global Gateway。它是帕秋莉感知世界的**第一触点**。
*   **任务逻辑**：
    1.  **L1 规则拦截**：快速过滤 Hi/Clear 等无意义对话，以及可能的系统指令。
    2.  **L2 智能处理**：调用轻量级 LLM (GPT-4o-mini / Local 7B)。
        *   **意图判断**：Chat vs RAG vs Tool。
        *   **指代消解**：将 "部署它" 重写为 "部署贪吃蛇游戏"。
        *   **关键词提取**：提取用于稀疏检索的 Tokens。
*   **输出**：结构化 JSON，同时供给检索层和感知层。

### 2.2 帕秋莉·检索使魔 (The Familiar of Patchouli) —— 对应 Retrieval Engine
> **定位**：**服务员与执行者**。解决“非对称检索”问题，确保高精度召回。

*   **设定**：当“真理之眼”确认需要查书时，帕秋莉会召唤使魔去书架取书。
*   **职能**：
    *   这是一个**即时响应**的动作（Hot Path）。
    *   它没有复杂的思考，只有精准的执行。它拿着“眼睛”给出的咒语（Keywords/Vector），瞬间抓取对应的书页（Memory Atoms）。
    *   它将书页递给前台的 Worker Agent。
*   **工程实现**：Retrieval 模块。虽然它是代码逻辑，但在概念上，它是帕秋莉**即时能力的体现**。
*   **三段式检索流**：
    1.  **并行召回 (Parallel Recall)**：
        *   **Dense**: 使用 `Rewritten Query` 进行向量检索。
        *   **Sparse**: 使用 `Keywords` 进行 BM25 检索。
        *   **Filter**: 应用 `Type` 或 `Source` 过滤。
    2.  **融合 (Fusion)**：使用 RRF (Reciprocal Rank Fusion) 合并多路结果。
    3.  **精排 (Reranking)**：引入 **Cross-Encoder Reranker** (如 bge-reranker)，对 Top-N 结果进行语义重打分，剔除伪相关内容。
*   **上下文渲染**：将 JSON 转换为 XML/Markdown 格式，并将置信度转化为自然语言提示。

### 2.3 帕秋莉·大图书馆本体 (Librarian Core)—— 对应 Perception & Generation & Lifecycle
> **定位**：**思考者与管理者**。解决“语义漂移”与“上下文割裂”问题。

*   **设定**：这是帕秋莉的**本体**，坐在书桌前，一边喝红茶一边处理堆积如山的借阅记录。
*   **职能**：
    *   **异步思考**：她阅读刚才发生的对话（感知话题漂移）。
    *   **撰写书籍**：她将有价值的知识精炼成册（记忆生成）。
    *   **整理书架**：她决定哪些书太旧了要扔进地下室（生命周期管理）。
*   **工程实现**：异步的 Librarian Loop。这是唯一需要完整 LLM 人格和复杂推理的部分。

---

## 3. 技术栈与服务选型 (Service Strategy)

遵循 **“重本地轻云端，重前端轻后端”** 的成本/性能优化原则。各模块的服务选型详见对应实现文档。

## 4. 进阶开发路线图 (Refined Roadmap)

基于目前的 MVP 实现，接下来的开发将分为多个专项优化阶段。

### Phase 2.1: 交互层重构 (The Eye Upgrade)

！注意，由于 Global Gateway 的模块设计会对当前的系统架构与数据流向产生重大影响，因此 Phase 2.1 必须先于所有其他 Phase 进行。

*   **目标**：实现 Global Gateway，打通“一次计算，多处复用”的链路。
*   **任务**：详见GlobalGateway.md文档

### Phase 2.2: 项目模块目录重构
*   **目标**：让帕秋莉的执行链路更加清晰，同时明确其作为HiveMemory系统的总管理者这一角色。
*   **任务**：在实现 Global Gateway 之后，将项目模块目录结构按照以上新的帕秋莉体系进行重构。

### Phase 2.3: 感知层重构 (The Core Upgrade)
*   **目标**：解决“贪吃蛇部署”被切分的话题连续性检测问题。
*   **任务**：详见DiscourseContinuity.md文档

### Phase 2.3: 检索层增强 (The Phantom Upgrade)
*   **目标**：增强并行检索后对于结果的融合排序，提高检索精度；优化上下文注入，减少重复内容。
*   **任务**：详见AdaptiveWeightedRetrieval.md文档与CompactContextRenderer.md文档

---
