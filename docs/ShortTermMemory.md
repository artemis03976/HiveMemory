# HiveMemory 子系统设计文档
## 主题：短期记忆系统重构与 MMU 化 (STM Refactoring & MMU)

**文档状态**: Approved (已定稿)
**适用阶段**: Phase 4.5 (Kernel Foundation Refactoring)
**核心重构模块**: `engines.gateway` (The Eye), `engines.perception`, `engines.kernel`

---

### 1. 架构动机与核心理念 (Motivation & Core Philosophy)

#### 1.1 现状与痛点 (Current Status & Pain Points)
在传统的 Agent 架构（含 HiveMemory 早期版本）中，**短期记忆 (STM)** 通常等同于 LLM 的 `messages` 数组。这种“线性堆叠、先进先出 (FIFO)”的管理方式在应对复杂、长时间的真实工作流时，暴露出了致命缺陷：
1.  **上下文污染 (Attention Dilution)**：用户在“深度 Coding”和“日常闲聊”之间来回切换时，无关话题会相互堆叠，严重干扰 LLM 的注意力机制，导致逻辑推理能力（Reasoning Performance）直线下降。
2.  **强制的会话隔离 (Session Isolation)**：主流应用依赖用户手动点击“新建对话”来隔离话题。这破坏了“全天候全能助理”的沉浸感，体验极其割裂。
3.  **指代消解的“先有鸡与先有蛋”悖论**：在多话题交织的情况下，纯向量或正则匹配无法准确判断“把它关掉”中的“它”究竟指向后台的哪个特定任务。

#### 1.2 核心理念一：三级记忆管理模型 (The Three-Tier Memory Model)
为了彻底解决上述痛点，系统确立了认知心理学与计算机科学相融合的三级记忆架构。**感知层（Perception Layer）明确不独立，而是作为 Librarian Core（帕秋莉）的前置短期记忆引擎。**
*   **短期记忆 (STM / Perception Layer)**：**工作台 (Workbench)**。即 Librarian 的感知层，负责管理当前正在进行的多个活跃话题（Active Topics），实现即时对话的路由、折叠（Fold）与防爆。
*   **中期记忆 (MTM / Generation Engine)**：**海马体 (Hippocampus)**。即 Librarian 的 GenerationEngine，当工作台上的任务告一段落（Swap-out）时，异步介入，将泥沙俱下的对话提取为高浓度的结构化“记忆原子（Memory Atoms）”，存入高速向量库。
*   **长期记忆 (LTM / Lifecycle Engine)**：**大脑皮层 (Cortex)**。即 Librarian 的 Lifecycle， 负责原子的遗忘、降级归档（冷存储）与知识图谱演化。

#### 1.3 核心理念二：感知层 MMU 化 (Perception as MMU)
我们将操作系统的 **段页式内存管理 (Segmented Paging)** 引入 LLM 上下文管理。
*   **摒弃全局 History**：Kernel 不再持有单一的线性 `history`。短期记忆的控制权全权交由感知层。
*   **单窗口多线程 (Single-Window, Multi-Track)**：用户只需面对一个无边界的聊天窗口。系统在后台自动维护多个独立的 **话题段 (Topic Segments)**。每个话题拥有绝对纯净的上下文。
*   **动态换入换出 (Swap-in/out)**：感知层作为 **内存管理器 (MMU)**，根据用户当前的意图，动态将对应的话题 Buffer 换入 Kernel 的工作区；对于长期不活跃的话题，则将其换出并移交 MTM 进行固化。

#### 1.4 核心理念三：智能体路由 (Agentic Routing)
放弃在短期活跃话题中使用“向量相似度”进行路由。将 **The Eye (Gateway)** 升级为 **进程调度员 (Agentic Dispatcher)**。
*   The Eye 拥有读取当前“活跃话题列表 (Active Process List)”的权限。
*   利用高智商低延迟的小模型（如 GPT-4o-mini），The Eye 能够在一个 Prompt 中同时完成 **“上下文精准匹配”** 与 **“复杂指代消解”**。
*   这不仅彻底解决了路由悖论，更赋予了系统宛如真人般“在多个并发任务间丝滑切换”的注意力控制能力。
---

### 2. 核心数据结构 (Core Data Structures)

为了实现话题的自由切换与流转，类似操作系统的内存分页机制（Segmented Paging），系统废除了单一的线性对话列表，转而构建多层级的容器结构。

#### 2.1 话题管理器 (TopicManager / The MMU)
`TopicManager` 存在于感知层（Perception Layer）中，是短期记忆的中央调度器，类似于操作系统的 MMU（内存管理单元）。

```python
class TopicManager:
    def __init__(self):
        # 活跃话题池 (L1 Cache / 驻留内存)
        # 仅在内存中维护少数几个正在并发交织的活跃话题
        self.active_topics: Dict[str, TopicSegment] = {}
        
        # 最大驻留限制 (超过此数量将触发 LRU Swap-out)
        self.max_resident_topics: int = 5 
```

#### 2.2 话题段 (TopicSegment / The Segment)
代表一个独立的讨论线程或工作区（Workspace）。它是上下文隔离的物理边界。

```python
class TopicSegment:
    def __init__(self, topic_id: str):
        self.topic_id: str = topic_id               # 唯一标识符
        self.title: str = "新建话题"                  # 由 The Eye 或 Kernel 异步生成，用于菜单展示
        
        # --- 页表 (Pages) ---
        # 存储具体的对话轮次，等同于旧架构中的 Buffer
        self.blocks: List[LogicalBlock] = []
        
        # --- 状态摘要 (Page Table Summary) ---
        # 当话题极长发生折叠 (Fold) 时，生成的伪无限上下文基底
        self.state_summary: str = "" 

        # --- 生命周期元数据 (Lifecycle Meta) ---
        self.last_accessed_at: float = time.time()  # 用于闲置超时与 LRU 淘汰
        self.total_tokens: int = 0                  # 监控水位线 (Watermark)
```

#### 2.3 逻辑块 (LogicalBlock / The Page)
最小的数据承载单元，代表一轮完整的交互（包含 MTP 协议的执行轨迹）。
*(注：结构沿用 MTP 清洗管道设计，包含 `user_query`, `clean_response`, `semantic_traces` 等)*。

---

### 3. 智能体路由与调度机制 (Agentic Routing & Dispatching)

本章定义当新的用户输入到达时，系统如何解决“多线程复合意图”与“跨话题指代消解”的难题。我们采用 **LLM 智能体路由 (Agentic Routing)** 替代传统的纯向量路由。

#### 3.1 核心流程：The Eye 调度机制
The Eye (Gateway) 不再是单纯的文本重写器，而是升级为**拥有全局视野的进程调度员**。

**交互时序**：
1.  **感知读取**：The Eye 收到用户输入后，首先向 MMU (`TopicManager`) 索要当前所有活跃话题的“进程菜单”（即 `[Topic_ID: Title]` 列表）。
2.  **智能推理**：The Eye 结合用户输入与活跃菜单，调用高智商/低延迟小模型（如 GPT-4o-mini）进行一次性推理。
3.  **结果输出**：The Eye 严格输出 JSON 格式的路由指令，包含 `target_topic` 和 `rewritten_query`。
4.  **MMU 执行**：MMU 根据 `target_topic` 将对应的 `TopicSegment` 换入（Swap-in）内核工作区。

#### 3.2 The Eye 核心 Prompt 设计示例
```markdown
[System]
你是一个 OS 级别的调度网关。你的任务是分析用户的最新输入，判断它属于哪个后台活跃任务，并补全缺失的指代信息。

【当前活跃任务列表】
{active_topics_menu}  # 例：["T_01: 编写贪吃蛇", "T_02: 晚餐食谱"]

【用户最新输入】
"{user_input}"

【输出要求 (严格输出 JSON)】
1. `target_topic`: 匹配的任务 ID。如果都不匹配，输出 "NEW_TOPIC"。
2. `rewritten_query`: 结合匹配任务的上下文，消除代词（它/这个），生成完整的独立指令。
```

#### 3.3 经典路由场景应对策略

基于上述 Agentic Routing，系统能够完美处理极其复杂的上下文跳跃：

*   **场景 A：连续强指代 (延续当前话题)**
    *   *输入*：(当前在 T_01 贪吃蛇) “把它调快点”
    *   *The Eye 决策*：命中 `T_01`。输出 query: “把贪吃蛇游戏的移动速度调快点”。
*   **场景 B：跨话题跳跃 (在并行任务中切换)**
    *   *输入*：(当前在 T_03 食谱，但之前有 T_02 训练任务) “看看训练情况”
    *   *The Eye 决策*：跨越当前上下文，精准命中 `T_01`。输出 query: “看看 LoRA 脚本的训练情况”。
*   **场景 C：开拓新荒 (全新话题)**
    *   *输入*：“今天天气怎么样”
    *   *The Eye 决策*：输出 `NEW_TOPIC`。

---

### 4. 上下文防爆与页折叠 (Context Compaction & Page Folding)

本章定义了在单一长任务（如长篇 Coding）下，如何防止 Worker Agent 的上下文窗口被撑爆。此过程为**同步阻塞操作**，其目标仅为“压缩上下文（Working Memory）”，不涉及记忆原子的提取。

#### 4.1 双重水位线设计 (Dual Watermarks)
我们在每个 `TopicSegment` 中引入水位线监控：
*   **物理极限 (Physical Limit)**：LLM 所能承受的最大 Context 窗口（如 128k Tokens）。
*   **高水位线 (High Watermark)**：触发折叠的软阈值（建议设为 32k - 64k Tokens，为 MTP 注入与新生成预留充足的 Buffer）。

#### 4.2 折叠算法与执行流 (The Folding Algorithm)
当检测到某话题的 `total_tokens > High_Watermark` 时，系统在 Kernel 发起下一次生成前，强制执行以下流程：

1.  **挂起生成 (Suspend)**：暂停当前的 LLM 续写请求。
2.  **状态提取 (State Extraction)**：
    *   取出当前话题旧的 `state_summary`。
    *   取出该话题下**最旧的 N 个** `LogicalBlock`（留下最近的 M 个以保持短期语境连贯）。
3.  **高速压缩 (Compaction)**：
    *   调用极速、低成本小模型（如 GPT-4o-mini 或 Claude-3.5-Haiku）执行压缩。
    *   *Prompt 示例*: "将以下历史对话提炼为当前状态摘要。重点保留：当前项目的阶段、已确立的规则、环境变量、以及未解决的 Bug 或任务。保持紧凑。"
4.  **状态替换 (State Replacement)**：
    *   将产出的新摘要覆盖旧的 `state_summary`。
    *   从 `blocks` 列表中永久丢弃那 N 个已被压缩的旧块，重置 `total_tokens` 计数。

#### 4.3 组装“伪无限上下文” (Context Hydration)
折叠完成后，Kernel 组装给 Worker Agent 的最终 Prompt 将呈现为“冰山结构”：

```markdown
<system_prompt>
... (基础人设与 MTP 协议说明) ...
<working_memory>
[当前话题状态]: {topic.state_summary}
</working_memory>
</system_prompt>

[这里是保留下来的最近 2-3 轮未折叠的对话，保证 Agent 仍能顺畅接话]
```
**设计收益**：此方案彻底解决了 RelayController 与 Librarian 的职责冲突。帕秋莉无需介入实时压缩，Agent 也不会因为历史过长而“降智”。

---

### 5. 话题生命周期与休眠换出 (Topic Lifecycle & Swap-out)

本章定义了多话题并发时的内存管理策略。这是驱动短期记忆（STM）向中期记忆（MTM）转化的核心动力学。

#### 5.1 换出触发器 (Swap-out Triggers)
感知层的 MMU 维护着有限的活跃话题池（`active_topics`）。当满足以下任一条件时，触发话题换出：

1.  **空闲超时休眠 (Idle Hibernate)**：
    *   *条件*：某话题最后一次访问时间 (`last_accessed_at`) 距今超过 X 分钟。
    *   *场景*：用户聊完“做菜”去忙别的了，该话题自然休眠。
2.  **空间挤压驱逐 (LRU Eviction)**：
    *   *条件*：活跃池已满（5/5），且 The Eye 刚刚路由出了一个 `NEW_TOPIC`（第 6 个）。
    *   *动作*：寻找 `last_accessed_at` 最久远的话题，强制驱逐。
3.  **显式结束 (Explicit Close)**：
    *   *条件*：用户发送系统指令（如 `/close`），或 The Eye 检测到明确的告别语（“这个项目就到这吧”）。

#### 5.2 换出执行流：帕秋莉的收割 (The Librarian's Harvest)
与“折叠 (Fold)”不同，“换出 (Swap-out)”是结束话题活跃状态的行为，必须触发中期记忆的归档操作。

1.  **移出内存 (Evict)**：将目标 `TopicSegment` 从 `active_topics` 字典中移除。
2.  **异步投递 (Async Hand-off)**：
    *   将这整个 `TopicSegment`（包含所有的 `LogicalBlock` 和 `state_summary`）打包，发送给 **Librarian Core (大图书馆本体)**。
    *   *帕秋莉视角*：“这个工作台上的任务结束了，我来将其中的精华提取为结构化的 Memory Atoms 存入向量库。”

#### 5.3 辅助生成：基于图与向量的记忆更新 (RAG for Memory) - *Phase 6 预留*
为了防止帕秋莉每次收割都创建一堆重复的原子（导致碎片化），系统在 MTM 提炼阶段引入**内部检索**：
*   在 Librarian 生成 `MemoryAtom` 草稿后，先拿去向量库里偷偷查一下：“我以前存过相关的记忆吗？”
*   如果命中，Librarian 会倾向于触发 **UPDATE 指令** 覆盖旧知识，而不是 CREATE 新知识。
*   *(注：此为架构前瞻，建议在后续的高级生命周期开发中具体实装。)*

---

### 6. 系统数据流转 (The New System Data Flow)

重构后，一次完整的用户交互生命周期如下：

1.  **用户输入** -> The Eye (清洗、路由、重写) -> 生成 `rewritten_query`，向 Perception Layer 发起 `route(topic_id)` 请求。
2.  **环境换入 (Swap-in)**：Perception Layer 返回对应的 `TopicSegment` 数据。
3.  **长时记忆注入 (RAG)**：Kernel 拿着 `rewritten_query` 向 Retrieval Familiar 请求 MTP 菜单（Menu）。
4.  **组装 Prompt**：Kernel 拼装 `[System + Topic_State_Summary + MTP_Menu + Topic_Recent_Blocks]`。
5.  **生成循环 (Generation Loop)**：Kernel 驱动 Worker Agent，处理 MTP 协议的多次中断与恢复（`⟪ READ ⟫`, `⟪ RUN ⟫`）。
6.  **环境换出与归档 (Swap-out & Archive)**：
    *   生成结束后，Kernel 将最终的 `InteractionPayload` (含 MTP Traces) 发送给 Perception Layer，处理成 `LogicalBlock` 并推送至 `TopicSegment` 中。
    *   Perception Layer 异步唤醒 Librarian，对新增的 Block 进行记忆原子（Memory Atom）的精炼与入库。

---
