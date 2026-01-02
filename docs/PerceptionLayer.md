这份文档更新方案旨在将“**统一语义流 (Unified Semantic Stream)**”架构正式纳入项目规划。这将显著提升系统在生产环境中处理复杂消息流（如长执行链、话题快速切换）的鲁棒性。

请按照以下指南修改原文档。主要变动集中在 **第 2 章（架构）**、**第 4 章（写入机制）** 以及 **第 8 章（技术栈）**。

---

## 修改点 1：第 2 章 系统总体架构

**位置**：`2.3 模块划分` -> `2.3.1 感知层 (Interaction Layer)`
**操作**：**[重写]** 该小节。
**理由**：原有的“影子复写”机制较为简单，需要升级为具备智能切分能力的“统一感知引擎”。

#### 更新后的内容：

### 2.3.1 感知层：统一语义流 (Interaction Layer: Unified Semantic Stream)
*   **职责**：作为系统的流量入口与预处理器，负责将非结构化的原始消息流（Raw Stream）转化为语义连贯的**逻辑原子块 (Logical Blocks)**，并决定何时唤醒 Librarian Agent。
*   **核心组件**：
    *   **流式解析器 (Stream Parser)**：抹平 1:1 对话（Bot）与 1:N 执行（Agent）的格式差异，将其封装为标准化的 Logical Block。
    *   **语义吸附器 (Semantic Adsorber)**：基于轻量级本地 Embedding 模型，实时计算新消息与当前 Buffer 内的 Logical Block 的语义相似度，决定是“吸附”还是“切分”。
    *   **接力控制器 (Relay Controller)**：处理 Token 溢出情况，生成“中间态摘要”以维持跨 Block 的上下文连贯性。

---

## 修改点 2：第 4 章 核心功能 I：记忆生成与写入

**位置**：`4.1 触发机制`
**操作**：**[完全替换]** 原 4.1 节内容。
**理由**：废弃原有的简单语义关键词/满额触发，采用新的“逻辑块 + 语义吸附 + 接力棒”三位一体策略。

#### 更新后的内容：

## 4.1 流式处理与触发机制 (Stream Processing & Triggering)

为了应对生产环境中多变的消息流形态（如 Bot 的闲聊跳转、Agent 的长链条执行、Bugfix 的短文本修正等等），系统不单单将消息同步给帕秋莉，而是采用 **“统一语义流 (Unified Semantic Stream)”** 感知层架构。这一方式不再区分 Agent 的对话模式，而是统一基于**逻辑原子块**作为最小的语义单元进行处理。

### 4.1.1 基础单元：逻辑原子块 (Logical Block)
帕秋莉处理的最小单位不再是单条从LLM接收的Message，而是**不可分割的语义单元**。在消息进入 Buffer 之前，先通过一个**预处理器**，将其封装为标准化的**逻辑原子块**。

*   **定义**：一个不可分割的最小语义单元。
*   **结构示例**：
    ```python
    class LogicalBlock:
        def __init__(self):
            # 1. 必须部分：用户意图
            self.user_block: Message = None 
            
            # 2. 可选部分：执行链 (Execution Chain)
            # 必须满足三元组约束: (Thought -> Tool Call -> Observation)
            # 如果中间断了（比如只有 Thought 没有 Tool），则视为不完整的 Block，等待流继续
            self.execution_chain: List[Triplet] = [] 
            
            # 3. 必须部分：最终响应
            self.response_block: Message = None
            
            # 辅助信息
            self.total_tokens: int = 0
            self.timestamp: float = time.time()

            # 2. 语义锚点 (Semantic Anchor) - 用于语义吸附与漂移
            # 仅包含 User Query (以及可能的少量上下文补充)
            self.anchor_text: str = self.user_block.content if self.user_block else ""

        @property
        def is_complete(self):
            """只有当 User 和 Final Response 都存在时，Block 才算闭合"""
            return self.user_block is not None and self.response_block is not None
    ```

**处理消息流的状态机逻辑：**

1.  **State: IDLE**
    *   收到 `User Message` -> 创建新 LogicalBlock，填入 `user_block`。
    *   转入 **State: PROCESSING**。
2.  **State: PROCESSING**
    *   收到 `Thought/Tool Call` -> 暂存入 `execution_chain`。
    *   收到 `Tool Output` -> 匹配并闭合上一个 Triplet。
    *   收到 `Assistant Message` (不带 Tool Call) -> 填入 `response_block`。
    *   **Block 闭合 (Sealed)** -> 推入 Buffer 进行语义判定。
    *   转入 **State: IDLE**。


### 4.1.2 核心算法：语义吸附与漂移 (Semantic Adsorption & Drift)

现在，Buffer 不再是一个“消息队列”，而是一个“**Logical Block 容器**”。我们使用“**语义吸附 (Semantic Adsorption)**” 算法来决定新来的 Block 是放入当前容器，还是触发切割。

由于 LLM 的生成内容（AI Response）通常很长且包含大量解释性废话，拼接后会稀释 User Query 的核心意图。显然话题的转移 90% 是由用户发起的，因此采用“**锚点对齐**”策略，不再计算整个 Buffer 的平均向量，而是维护一个 “**当前话题核心 (Current Topic Kernel)**”。变量定义：

- **Topic_Kernel_Vec**: 当前 Buffer 中所有 anchor_text 的指数移动平均向量（或者仅使用第一条 User Query 作为基准，视策略而定）。

帕秋莉的感知引擎维护一个动态的 Buffer，对新进入的 Block 执行以下判定流程：

1.  **Step 1: Anchor 文本提取与增强**：
    *   短文本强吸附/极短文本补全：若新 Block Token 数极少（如 < 50，典型如“继续”、“报错了”），**强制吸附**进当前 Buffer，或引入上一个block的user query作为Anchor Text。
    *   *目的*：防止因 Embedding 不准导致对上一轮的修正指令被错误切分为新话题。

2. **Step 2: 计算向量距离**
    *   使用本地轻量模型（如 `all-MiniLM-L6-v2`）计算 `New_Block` 的 `anchor_text` 与 `Topic_Kernel_Vec` 的余弦相似度。

2.  **Step 2: 语义相似度判定 (Embedding Similarity)**：
    *   **吸附 (Adsorb)**：相似度 > 阈值（如 0.6）。加入 Buffer，更新 Buffer 平均向量，让 Topic Kernel 缓慢向新的 User Query 移动
    *   **漂移 (Drift)**：相似度 < 阈值。判定为话题切换（语义休止符）。
        *   *Action*：触发 **Flush**（唤醒帕秋莉处理旧 Buffer），以 New_Block 开启新 Buffer并建立新的 Topic Kernel。

3.  **闲置超时 (Idle Timeout)**：
    *   若 Buffer 超过 T 分钟无新 Block 进入，视为自然休止，触发 **Flush**。

4. **用户手动触发（Flush）**：
    *   用户在任何对话结束时，都可以通过发送 `/save` 指令，强制触发当前 Buffer 的处理，即绕过语义吸附，直接发送给帕秋莉。
    *   *目的*：尊重用户的不同工作流习惯，给出人为介入的接口。

### 4.1.3 接力棒机制 (State Relay Mechanism)：处理“上下文割裂问题”
解决“长任务导致的 Token 溢出割裂”问题。

*   **触发条件**：在吸附之前，检查 Current_Buffer_Tokens + New_Block_Tokens 是否超过 Max_Processing_Tokens（如 8k）。
*   **执行流程**：
    1.  **强制切分**：将 `Current_Buffer` 发送给 帕秋莉。
    2.  **摘要生成**： 帕秋莉 在处理该 Batch 时，额外产出一个 **Running Summary**（如“已完成 A 模块代码，正在调试 B 模块”）。
    3.  **状态注入**：系统将该 Summary 自动插入到下一个新 Buffer 的头部（作为虚拟的 Context）。
*   **效果**：即使物理上切分了，帕秋莉在处理下一段时依然拥有前文的“上帝视角”，保证生成的记忆原子不丢失上下文。

#### 场景演示：大执行 + 小 Bugfix

1.  **Block A (大执行)**：用户让写贪吃蛇游戏。Agent 写了 50 轮，Tokens = 6k。
    *   *判定*：Token 即将溢出。
    *   *动作*：**强制切分**。
    *   *接力*：帕秋莉生成摘要 `Summary_A`: "已完成贪吃蛇核心逻辑，包含类 Snake 和 Game。" -> **传入下一个 Buffer**。

2.  **Block B (小 Bugfix)**：用户说：“蛇撞墙没死”。
    *   *判定*：
        *   虽然 Block A 已经被切走了（不在 Buffer 里了），但 Buffer 头部有 `Summary_A`。
        *   计算 `Block B` 与 `Summary_A` 的相似度 -> **高**（都包含 Snake/Game 关键词）。
        *   **吸附成功**。
    *   *结果*：Block B 单独（或与后续对话一起）生成记忆。
    *   *帕秋莉视角*：当她处理 Block B 时，她看到的 Input 是 `[Summary_A, Block_B]`。她能完美理解“蛇撞墙”是指什么，生成的记忆原子会是：“修复了贪吃蛇撞墙判定的 Bug”。

---

### 修改点 3：第 8 章 技术栈选型

**位置**：`8.3 模型服务抽象层`
**操作**：**[新增]** 本地感知模型部分。
**理由**：语义吸附需要极低延迟、零成本的 Embedding 计算，不能依赖调 API。

#### 更新后的内容：

*   **感知层 Embedding (Perception Layer)**:
    *   *用途*: 用于 4.1 节的流式语义吸附与漂移检测。
    *   *选型*: **Sentence-Transformers** (本地运行)。
    *   *模型*: `all-MiniLM-L6-v2` 或 `bge-small-en-v1.5`。
    *   *理由*: 极轻量（< 100MB），CPU 推理仅需毫秒级，无需消耗昂贵的 LLM Token，适合高频实时计算。
