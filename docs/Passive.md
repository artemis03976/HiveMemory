# HiveMemory 技术架构更新文档 (v1.2)
## 主题：外部被动监听模式与兼容性架构 (Passive Observer Mode)

**适用模块**: `engines.gateway` (The Eye), `engines.kernel`, `engines.retrieval`
**设计模式**: 适配器模式 (Adapter Pattern), 旁路中间件 (Sidecar)
**当前阶段**: Phase 4/5 架构完善

---

### 1. 背景与动机 (Motivation)

在先前版本的重构中，系统感知层（Perception Layer）全面转向了“原子化提交（Atomic Commit）”模式，只接收完整闭合的 `InteractionPayload`，废弃了原有的逐条消息拼接逻辑。

这一改动极大地提升了 MTP 协议下的数据纯净度，但也带来了一个兼容性问题：**对于不受 PatchouliKernel 控制的外部系统（如独立的 Discord Bot、微信机器人或其他传统 Agent 框架），它们的消息流依然是离散的、逐条到达的。**

为了保持 HiveMemory 的通用基础设施定位，提供**非侵入性**的记忆支持，我们需要在系统最外层（Gateway）构建一个缓冲与适配机制，使得核心引擎（Kernel/Librarian）能够用统一的逻辑同时处理“主动驱动”与“被动监听”两种数据流。

---

### 2. 核心设计：双模态系统架构 (Dual-Mode Architecture)

通过本次更新，HiveMemory 将正式确立两种运行形态：

1.  **形态 A：AIOS 引擎 (Active Kernel Mode)**
    *   **流程**：Kernel 主动调用 LLM，通过 MTP 协议控制生成流，支持动态工具和记忆菜单。
    *   **适用场景**：基于 HiveMemory 原生构建的 Worker Agent。
2.  **形态 B：记忆中间件 (Passive Observer Mode)**
    *   **流程**：外部系统接管 LLM 生成。Gateway 仅作为“旁路监听者”收集离散消息，打包后异步投递给 Kernel 进行记忆沉淀；在对话前提供全量文本的上下文支持。
    *   **适用场景**：接入已有的外部 Chatbot 或传统 Agent 系统。

---

### 3. Gateway 缓冲与状态机设计 (The Observer Buffer)

Gateway（真理之眼）将承担适配器（Adapter）的职责。如同“等全员到齐才放行的保安”，它负责将外部系统碎片化的流拼接为完整的 Payload。

#### 3.1 SessionBuffer 机制
在 Gateway 内存（或 Redis）中维护一个轻量级的 `SessionBuffer`，按 `session_id` 隔离。

#### 3.2 触发打包的条件 (Flush Triggers)
Gateway 通过以下三种策略判断一轮对话（User -> Assistant）是否结束：
1.  **新用户消息打断 (Next User Turn)**: 当收到同一个 Session 的*下一条* `role: "user"` 消息时，说明上一轮必然已结束。立即打包上一轮的数据，并将新消息作为新一轮的开头。
2.  **闲置超时 (Idle Timeout)**: 收到 Assistant 消息后，启动 $T$ 秒（如 30 秒）的倒计时。若无新消息进入，判定本轮结束，触发打包。
3.  **显式结束符 (Explicit EOF)**: 对外提供特殊标记/接口，允许外部系统主动通知“本轮生成已完成”，实现零延迟打包。

#### 3.3 构建 Payload
触发打包时，Gateway 组装 `InteractionPayload`：
*   `user_message`: 缓冲的用户请求。
*   `assistant_message`: 缓冲的助手回复（拼接后）。
*   `mtp_traces`: **空列表 `[]`**（被动模式无协议指令）。
*   `write_focus`: **None**。

---

### 4. 检索与上下文降级策略 (Retrieval Downgrade)

在被动模式下，外部 Agent 不懂 MTP 协议，如果预检索（Pre-retrieval）向其提供 `<memory_index>`（别名菜单），外部 Agent 将无法使用 `⟪ READ ⟫` 获取详情，导致上下文失效。

因此，**Retrieval Familiar 必须支持“策略降级”**。

#### 5.1 检索模式参数 (`mode`)
API 请求（或 Gateway 路由时）增加 `mode` 参数：`active` 或 `passive`。

#### 5.2 渲染分流逻辑
```python
# engines/retrieval/familiar.py

async def fetch_context(self, query: str, mode: str = "active") -> str:
    # 1. 执行混合向量检索，获取 Top-K Atoms
    atoms = await self._hybrid_search(query)
    
    # 2. 渲染策略分支
    if mode == "active":
        # AIOS 模式：使用 CompactContextRenderer
        # 仅注入 Title + Alias 菜单，引导 Agent 使用 MTP 查阅
        return self.compact_renderer.render(atoms)
        
    elif mode == "passive":
        # 兼容模式：使用 FullContextRenderer
        # 简单粗暴，直接将 Top-K 的完整 Payload 文本拼在一起
        # 注意：强制设定 max_tokens 截断，防止撑爆外部 Bot 的窗口
        return self.full_renderer.render(atoms, max_tokens=2000)
```

---

### 5. 总结 (Conclusion)

通过本次网关缓冲机制的引入与检索策略的降级，HiveMemory 完美保留了其作为“被动海马体”的兼容性。

*   **对内**：核心业务逻辑（感知、提取、存储）保持高度纯洁与统一，数据通道唯一。
*   **对外**：提供了极大的接入灵活性，业务方可以根据自身系统的改造意愿，自由选择作为“寄生”（Passive）还是“共生”（Active）节点接入 PatchouliSystem。