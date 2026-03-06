# HiveMemory 技术架构更新文档
## 主题：话题生命周期调度与 Buffer 维护机制重构 (Topic Lifecycle & Buffer Management)

**文档状态**: Draft (草案)
**适用模块**: `engines.perception.topic_manager`
**关联机制**: SystemBus, MTP (WRITE/UPDATE), Page Folding
**当前阶段**: Phase 4.5 (STM / MMU Refactoring)

---

### 1. 痛点与演进动机 (Motivation)

在升级为“段页式话题管理 (MMU)”架构后，感知层（Perception Layer）对对话数据的处理变得高度复杂。原有的单一 `flush` 逻辑已无法满足需求。

**核心矛盾**：对于一个话题 Buffer 的维护，实际上涉及三个完全独立的决策维度：
1.  **数据去向 (Data Destination)**：这批数据是否包含了完整的事件/重要意图，需要发送给帕秋莉（Librarian）生成中期记忆？
2.  **语境延续 (Context Continuation)**：该话题是否还要继续？是否需要提取核心摘要（State Summary）供下一轮对话使用？
3.  **内存生命周期 (Memory Lifecycle)**：处理完毕后，该 Buffer 是继续驻留在活跃内存（Active Pool）中，还是被踢出/销毁？

为了避免代码中出现大量冗余的 `if-else` 和状态混乱，必须对 Buffer 的维护动作进行**解耦与正交化**。

---

### 2. 核心概念：原子操作 (Atomic Operations)

我们废弃“Flush”这一统称，在 `SemanticBufferManager` 中定义三个基础原子操作：

*   **`Archive` (归档 - 异步)**
    *   *动作*：将当前 Buffer 中的 `LogicalBlocks` 打包，通过 `SystemBus` 发射 `librarian.generate_memory` 事件。
    *   *特性*：纯异步（Fire-and-forget），绝对不阻塞前台 Agent 的生成。
*   **`Compact` (压缩 - 同步)**
    *   *动作*：调用高速小模型（如 GPT-4o-mini），将当前的 Blocks 浓缩为新的 `state_summary`，**并清空 Blocks**。
    *   *特性*：同步阻塞。因为 Agent 的下一轮发言必须依赖这个新生成的摘要。
*   **`Evict` (驱逐 - 内存操作)**
    *   *动作*：将该 `SemanticBuffer` 从 `SemanticBufferManager` 的 Buffer 池字典中彻底移除，释放 L1 Cache 空间。

---

### 3. 场景映射矩阵 (The Scenario Decision Matrix)

基于上述三个原子操作，系统中目前所有可能触发 Buffer 结算的场景都可以清晰地映射为一张控制表：

| 触发器 (Trigger Reason) | 业务场景描述 | Action 1: Archive (造记忆) | Action 2: Compact (留摘要) | Action 3: Evict (踢出内存) | 最终 Buffer 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`TOKEN_OVERFLOW`** | 话没聊完，但逼近单话题 Token 水位线 | ❌ 否 *(话题未完，暂不归档)* | ✅ **是** *(必须留摘要以接力)* | ❌ 否 | 存活 *(仅含新摘要)* |
| **`IDLE_TIMEOUT`** | 话题闲置超过 xx 分钟，自然冷却 | ✅ **是** *(完整事件，落库)* | ❌ 否 *(无需摘要，人已走)* | ✅ **是** | 被销毁 |
| **`LRU_EVICTION`** | 活跃话题超限，被新话题硬挤出内存 | ✅ **是** *(完整事件，落库)* | ❌ 否 *(同上)* | ✅ **是** | 被销毁 |
| **`MTP_SIGNAL`** | Agent 主动发出 `WRITE`/`UPDATE` 指令 | ✅ **是** *(带 Focus 打包发走)* | ✅ **是** *(话题未完，必须留摘要)*| ❌ 否 | 存活 *(含新摘要)* |

---

### 4. 架构落地：统一调度器 (The Dispatcher)

在 `SemanticBufferManager` 中实现统一的话题结算中心，通过标志位（Flags）组合来控制流转。

#### 4.1 调度器伪代码实现
```python
# engines/perception/buffer_manager.py

class SemanticBufferManager:
    def __init__(self, bus: SystemBus):
        self.bus = bus
        self.active_topics: Dict[str, TopicSegment] = {}

    async def resolve_topic(self, topic_id: str, trigger_reason: str, mtp_focus=None):
        """
        统一的话题结算调度器
        """
        topic = self.active_topics.get(topic_id)
        if not topic or not topic.blocks:
            return

        # 1. 提取当前数据快照 (防止后续操作污染)
        blocks_snapshot = topic.blocks.copy()

        # 2. 查表：决策开关 (依据第三节的矩阵)
        need_archive = trigger_reason in["IDLE_TIMEOUT", "LRU_EVICTION", "MTP_SIGNAL"]
        need_compact = trigger_reason in ["TOKEN_OVERFLOW", "MTP_SIGNAL"]
        need_evict   = trigger_reason in ["IDLE_TIMEOUT", "LRU_EVICTION"]

        # ================= 执行区 =================
        
        # Action 1: Archive (异步非阻塞)
        if need_archive:
            payload = {
                "topic_id": topic_id,
                "blocks": blocks_snapshot,
                "focus": mtp_focus  # 仅在 MTP_SIGNAL 时有值
            }
            self.bus.emit("librarian.generate_memory", data=payload)

        # Action 2: Compact (同步阻塞)
        if need_compact:
            new_summary = await self._generate_state_summary(
                old_summary=topic.state_summary, 
                blocks=blocks_snapshot
            )
            topic.state_summary = new_summary
            
        # 无论是否 Compact，只要发生结算，旧 Blocks 都必须清空
        # (因为 Archive 拿走了快照，Compact 提炼了摘要)
        topic.blocks.clear() 

        # Action 3: Evict (内存清理)
        if need_evict:
            del self.active_topics[topic_id]
```

---

### 5. 关键附属机制实现 (Supporting Mechanisms)

为了让调度器完美运转，需要补充以下周边机制：

#### 5.1 MTP 触发下的时序优化
在 `MTP_SIGNAL` 场景下，既要 `Archive` 又要 `Compact`。
*   **设计保障**：由于 `Archive` 被设计为向 `SystemBus` 发射异步事件（`bus.emit`），它会立刻返回。随后系统阻塞在 `Compact` 步骤（耗时约 1-2 秒）。
*   **体验保障**：在这 1-2 秒内，前端（The Eye 或 API 返回）可以维持 Loading 状态（如：“*正在整理工作区...*”），确保摘要生成完毕后，再释放给 Kernel 发起下一轮 LLM 推理。

---
