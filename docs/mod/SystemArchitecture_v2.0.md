# HiveMemory 系统架构演进与未来开发规划 (v2.0)

**版本**: 2.0 
**核心目标**: 解决在 MVP 阶段后的多次模块增补与重构后，造成的系统延迟问题与帕秋莉的职能分配不合理。重新调整系统的逻辑架构，构建高鲁棒性的记忆基础设施。

## 1. 架构重构：帕秋莉体系 (The Patchouli System)

为了解决记忆检索需求的“实时检索”与记忆生成的“异步管理”的本质在工程实现上的冲突，同时保持“帕秋莉作为总管理者”的设计初衷，我们将系统架构重构为 **分布式智能体系**。帕秋莉不再是一个单一的后台进程，而是由多个职能组成的有机整体。

### 1.1 概念模型：三位一体 (The Trinity Aspect)

| 分身名称 | 对应模块实现 | 所在层级 | 核心职责 | 特性 |
| :--- | :--- | :--- | :--- | :--- |
| **真理之眼 (The Eye)** | **Global Gateway** | **交互层 (Interaction)** | 意图识别、查询重写、流量分发 | **同步阻塞**、极低延迟、小模型驱动 |
| **检索使魔 (Retrieval Familiar)** | **Retrieval Engine** | **热处理层 (Hot Path)** | 混合检索、重排序、上下文渲染 | **同步阻塞**、高并发、本地计算密集 |
| **大图书馆本体 (Librarian Core)** | **Perception Layer + Memory Generation + Lifecycle** | **冷处理层 (Cold Path)** | 话题感知、记忆生成、生命周期管理 | **异步非阻塞**、高智商、SOTA 模型驱动 |

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
        Perception -->|话题漂移检测| Buffer[逻辑块缓冲]
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

### 2.2 帕秋莉·检索使魔 (The Retrieval Familiar of Patchouli) —— 对应 Retrieval Engine
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

---

## 4. 进阶开发路线图 (Refined Roadmap)

基于目前的 MVP 实现，接下来的开发将分为多个专项优化阶段。

### Phase 1: 交互层重构 (The Eye Upgrade)

！注意，由于 Global Gateway 的模块设计会对当前的系统架构与数据流向产生重大影响，因此 Phase 1 必须先于所有其他 Phase 进行。
！在 Phase 1 中，先不处理关于项目目录重构的逻辑，仅实现 gateway 组件，与perception等模块同级，重点在于跑通新数据流。

*   **目标**：实现 Global Gateway，打通“一次计算，多处复用”的链路。
*   **任务**：详见GlobalGateway.md文档

### Phase 2: 项目模块目录重构

！注意，项目目录重构是一次破坏性更新，可能导致大量文件导入与部分集成测试文件失效。并且，新目录结构依赖于 Global Gateway 的实现。因此请在重构后，及时更新所有相关的导入路径与测试用例，保证系统仍能正常运行后再规划后续开发

*   **目标**：让帕秋莉的执行链路更加清晰，同时明确其作为HiveMemory系统的总管理者这一角色。
*   **任务**：在实现 Global Gateway 之后，将项目模块目录结构按照以上新的帕秋莉体系进行重构，详见下文

### Phase 3: 感知模块增强 (Perception Upgrade)
*   **目标**：解决“贪吃蛇部署”被切分的话题连续性检测问题。
*   **任务**：详见DiscourseContinuity.md文档

### Phase 4: 检索模块增强 (Retrieval Upgrade)
*   **目标**：增强并行检索后对于结果的融合排序，提高检索精度；优化上下文注入，减少重复内容。
*   **任务**：详见AdaptiveWeightedRetrieval.md文档与CompactContextRenderer.md文档

---

## 5. 项目目录重构具体计划

目前的结构中，`agents/patchouli.py` 承担了过多的职责（感知、生成、以及未来的检索入口），这更像是一个单体应用。

为了体现她作为 **“分布式智能系统”** 和 **“全知全能馆长”** 的设计，建议采用 **Facade（外观模式）** + **Component（组件化）** 的架构进行重构。

以下是具体的重构方案：

### 1. 核心重构理念

我们将项目划分为两个层级：
1.  **能力层 (Capabilities / Engines)**：即现有的 `perception`, `generation`, `retrieval` 等模块。它们是帕秋莉的“魔法书”，负责具体的计算。
2.  **人格层 (Personas / Interfaces)**：即帕秋莉的三个分身。它们封装能力层，对外提供统一的业务接口。

### 2. 推荐的目录结构

建议废弃 src 内部的 `agents` 目录（因为帕秋莉不仅仅是一个 agent，她现在是整个 HiveMemory 系统的管理者与代行者），将其提升为顶级的 `patchouli` 包，并在其中定义她的三个分身。原 `agents/chatbot.py` 以及 `agents/session_manager.py` 需要移出 `src/hivememory` 目录，因为本质上他们属于前台 Worker Agent 的 demo 实现，而不是 HiveMemory 系统的核心组件。这两个文件可以与 `chatbot_ui.py` 共同放在一个目录下管理，但目录名不能是现在的 `examples`，不然与实际用处不符。

为了避免模块的过度封装，eye.py 应由原来的 `gateway/gateway.py` 改造而来，familiar.py 则由原来的 `retrieval/engine.py` 改造而来。核心原则是自此以后帕秋莉将是所有业务逻辑的编排者。她负责实例化这些工具，并定义数据如何在工具间流动。而 `engines` 中的模块作为纯粹的工具类，仅实现对应的业务逻辑

```text
src/hivememory
│  __init__.py
│  client.py                   # [NEW] 统一入口 (HiveMemoryClient)
│
├─patchouli                    # [NEW] 人格层：帕秋莉的三位一体分身
│  │  __init__.py              # 导出 Eye, Familiar, Core
│  │  eye.py                   # [REFACTOR] 真理之眼 (GlobalGateway/Router, 原 gateway/gateway.py) - 同步阻塞
│  │  retrieval_familiar.py    # [REFACTOR] 检索使魔 (Retrieval Engine, 原 retrieval/engine.py) - 同步阻塞
│  │  librarian_core.py        # [REFACTOR] 馆长本体 (原 agents/patchouli.py) - 异步后台
│  │  config.py                # [REFACTOR] 帕秋莉的统一配置 (原 config.py)
│
├─engines                      # [RENAME] 能力层 (原各功能模块归档于此，更清晰)
│  │  __init__.py
│  │
│  ├─gateway                   # [NEW] 具体的 Router/Rewriter 实现
│  │
│  ├─perception                # [EXISTING] 话题感知与缓冲区管理
│  │
│  ├─generation                # [EXISTING] 记忆提取与生
│  │
│  ├─retrieval                 # [EXISTING] 向量检索与上下文渲染
│  │
│  └─lifecycle                 # [EXISTING] 生命周期
│
├─infrastructure               # [NEW] 基础设施层 (数据与底层模型)
│  │  storage                  # 数据库服务统一接口
│  │  llm                      # LLM服务统一接口
│  │  embedding                # 嵌入模型服务统一接口
│
└─utils
```

### 3. 代码实现逻辑

#### A. 统一入口 (The System Facade)

不要让用户分别去实例化 Eye 或 Core，提供一个 `PatchouliSystem` 类来管理一切。

```python
# src/hivememory/patchouli/system.py 

class PatchouliSystem:
    """
    帕秋莉体系 - HiveMemory 的完整封装
    
    Attributes:
        eye (TheEye): 真理之眼，负责流量入口和意图判断 (Hot)
        retrieval_familiar (RetrievalFamiliar): 检索使魔，负责上下文检索 (Hot)
        core (LibrarianCore): 馆长本体，负责后台记忆维护 (Cold)
    """
    def __init__(self, config: HiveMemoryConfig):
        # 初始化基础设施
        self.storage = QdrantMemoryStore(config.qdrant)
        
        # 1. 实例化真理之眼 (Gateway)
        self.eye = TheEye(config=config.gateway)
        
        # 2. 实例化检索使魔 (Retrieval)
        self.retrieval_familiar = RetrievalFamiliar(storage=self.storage, config=config.retrieval)
        
        # 3. 实例化馆长本体 (Librarian)
        # 注意：Core 需要引用 Eye 的重写结果，通过队列或回调连接
        self.librarian_core = LibrarianCore(storage=self.storage, config=config.librarian)
        
    async def process_user_query(self, query: str, context: list):
        """
        标准 Hot Path 流程： Eye -> RetrievalFamiliar -> Worker
        """
        # Step 1: Eye 判断与重写
        gateway_result = await self.eye.gaze(query, context)
        
        # Step 2: 异步通知 LibrarianCore (作为 Anchor 进行感知)
        # 这是一个关键的连接点：Eye 的产出喂给了 LibrarianCore
        self.librarian_core.perceive_async(
            anchor=gateway_result.rewritten_query, 
            raw_message=query
        )
        
        # Step 3: 根据 Eye 的判断决定是否检索
        retrieved_context = None
        if gateway_result.intent == "RAG":
            retrieved_context = await self.retrieval_familiar.retrieve(
                query=gateway_result.content_payload.rewritten_query,
                keywords=gateway_result.content_payload.search_keywords,
                filters=gateway_result.content_payload.target_filters
            )
            
        return {
            "intent": gateway_result.intent,
            "rewritten": gateway_result.rewritten_query,
            "memory": retrieved_context
        }
```

#### B. 馆长本体 (The Librarian Core) 的重构

将当前的 `patchouli.py` 瘦身，重命名为 `librarian_core.py`，专注做 **Cold Path** 的工作。

```python
# src/hivememory/patchouli/librarian_core.py

class LibrarianCore:
    """
    帕秋莉·馆长本体 (Librarian Agent)
    
    职责:
        - 接收 Eye 传来的感知信号 (Anchors)
        - 维护 Buffer 和 漂移检测
        - 调用 Generation 引擎写书
        - 调用 Lifecycle 引擎修书
    """
    def __init__(self, storage, config):
        self.perception_layer = create_perception_layer(config.perception)
        self.generation_orchestrator = create_generation_orchestrator(storage, config.generation)
        self.lifecycle_manager = create_lifecycle_manager(storage, config.lifecycle)

    def perceive_async(self, anchor: str, raw_message: str):
        """接收真理之眼的投喂"""
        # 使用 Anchor 进行语义漂移检测
        # 如果漂移 -> 触发 flush -> 调用 generation_orchestrator
        pass
        
    def start_gardening(self):
        """开启定时维护模式"""
        self.lifecycle_manager.run_maintenance()
```

---

### 4. 重构的具体步骤建议

1.  **移动与重命名**：
    *   将 `core/embedding`, `core/llm`, `memory/storage` 移动到 `infrastructure/`。这会让根目录更干净。
    *   将 `gateway`, `perception`, `generation`, `retrieval`, `lifecycle` 移动到 `engines/`。这代表它们是底层的“发动机”。
    *   将 `agents/patchouli.py` 移动到 `patchouli/core.py` 并进行删减（移除还没有做的 retrieval 占位符）。

3.  **构建人格层**：
    *   创建 `patchouli/eye.py`: 封装 Gateway 引擎，提供 `gaze()` 方法。
    *   创建 `patchouli/retrieval_familiar.py`: 封装 Retrieval 引擎，提供 `retrieve()` 方法。
    *   修改 `patchouli/librarian_core.py`: 封装 Perception & Generation & Lifecycle 引擎，提供对记忆"CRUD"的所有操作

4.  **连接**：
    *   编写 `patchouli/system.py`，这是用户（开发者）唯一需要 import 的东西。今后所有 Worker Agent 都将通过这个 System 类获得 HiveMemory系统的所有功能

### 5. 这种结构的好处

*   **全知全能的体现**：用户调用的是 `patchouli.system`，感觉上是在与一个完整的智能体交互，而不是在拼凑 `retriever` 和 `generator`。
*   **分布式友好**：`Eye` 和 `Familiar` 是轻量级、无状态的，可以很容易地部署为 HTTP API (FastAPI)；而 `Core` 是有状态的（Buffer），可以单独作为一个 Worker 进程部署。
*   **认知分离**：
    *   代码逻辑上，你很清楚：`engines` 目录是**干活的工具**（怎么算 Embedding，怎么存 Qdrant）。
    *   `patchouli` 目录是**业务逻辑**（什么时候切分，什么时候归档，怎么重写）。