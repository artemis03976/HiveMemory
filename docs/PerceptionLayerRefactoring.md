# HiveMemory 技术架构更新文档
## 主题：感知层重构与逻辑块演进 (Perception Layer Refactoring)

**适用模块**: `engines.perception`, `kernel`
**关联协议**: Memory Tool Protocol (MTP)

---

### 1. 背景与动机 (Motivation)

随着 **PatchouliKernel** 及其递归生成循环（Recursive Loop）的实装，系统的运行模式已从“被动监听流式对话”转变为“内核驱动的主动交互”。原有的感知层设计面临以下核心冲突：

1.  **结构不匹配**：旧的 `execution_chain`（Thought-Tool-Observation 三元组）不再适用。在 Kernel 模式下，Tool 的执行过程已被内化为完整的 Assistant Message，而非碎片化的流。
2.  **噪音干扰**：MTP 协议产生的 `⟪...⟫` 指令和 `<mtp_response>` XML 标签对于记忆生成（Librarian）而言是高密度的“语法噪音”，需要清洗。
3.  **时序错位**：原有的 `WRITE/UPDATE` 采用专用通道触发 Flush，导致当轮对话的 Assistant Message 尚未进入 Block 就被强制截断，造成上下文丢失（Context Loss）。

为了解决上述问题，我们需要对 **LogicalBlock 的数据结构** 及 **感知层的数据摄入管道（Ingestion Pipeline）** 进行重构。

---

### 2. LogicalBlock 结构重构 (Structural Changes)

`LogicalBlock` 的职责从“拼接碎片”转变为 **“语义承载”**。我们废弃原有的 `execution_chain`，引入 **`Semantic Trace`**。

#### 2.1 新版结构定义

```python
class LogicalBlock:
    def __init__(self):
        # --- 基础上下文 ---
        self.user_query: str = ""       # 原始用户问题
        self.rewritten_query: str = ""  # (来自 Gateway) 指代消解后的意图锚点
        
        # --- 语义轨迹 (替代 Execution Chain) ---
        # 存储经过清洗和降维的 MTP 操作摘要
        self.semantic_traces: List[TraceItem] = []
        
        # --- 响应数据 ---
        self.raw_response: str = ""     # 包含 MTP 指令和 XML 的完整原始文本 (用于 Debug/Context)
        self.clean_response: str = ""   # 去除 MTP 噪音后的纯净回复 (用户可见版本)
        
        # --- 控制信号 (The Signals) ---
        self.priority: str = "NORMAL"   # NORMAL | URGENT
        self.write_focus: Optional[WriteFocus] = None   # 携带 WRITE 指令的核心素材
        self.update_focus: Optional[UpdateFocus] = None # 携带 UPDATE 指令的修改意图
```

#### 2.2 Semantic Trace 设计
`TraceItem` 旨在为 Librarian 提供高信噪比的行为摘要，而非执行细节。

| MTP 指令 | 清洗策略 | Trace 示例 |
| :--- | :--- | :--- |
| **READ** | **折叠**。忽略具体读了什么内容，仅记录查阅动作。 | `{"action": "READ", "target": "mem_auth_doc"}` |
| **SEARCH** | **保留**。记录 Agent 的探索意图。 | `{"action": "SEARCH", "query": "docker config"}` |
| **RUN** | **摘要**。记录副作用操作及状态。 | `{"action": "RUN", "tool": "sys_write_file", "status": "success"}` |
| **XML注入包裹** | **丢弃**。具体的 `<mtp_response>` 标签内容不进入 Trace。 | *(N/A)* |

---

### 3. 数据摄入管道重构 (Ingestion Pipeline Refactoring)

为了解决“上下文丢失”问题，我们废弃旁路触发机制，采用 **“消息即载体 (Message as Carrier)”** 的统一提交模式。

#### 3.1 传输对象：InteractionPayload
在 Kernel 完成一轮生成循环后，将所有相关数据封装为一个原子包提交给感知层。

```python
class InteractionPayload:
    """Kernel -> Perception 的原子传输包"""
    user_message: Message
    assistant_message: Message      # 包含 MTP 指令的完整文本
    mtp_traces: List[dict]          # 由 Koakuma 在执行过程中记录的 Trace 列表
    
    # 控制信号 (挂载在 Payload 上，而非独立传输)
    write_focus: Optional[WriteFocus] = None
    update_focus: Optional[UpdateFocus] = None
```

#### 3.2 BlockBuilder 状态机逻辑
感知层不再被动等待，而是根据 Payload 中的信号主动决策。

*   **Step 1: 填充数据**
    *   将 `user_message` 填入 Block。
    *   将 `assistant_message` 解析为 `raw` 和 `clean` 版本填入 Block。
    *   追加 `mtp_traces`。

*   **Step 2: 信号检查 (Signal Check)**
    *   **Check**: Payload 是否携带 `write_focus` 或 `update_focus`？
    *   **YES**: 
        1.  将 Focus 对象挂载到 Block 上。
        2.  将 Block 标记为 `URGENT`。
        3.  **立即触发 Flush** (Reason: MTP_WRITE/MTP_UPDATE)。
    *   **NO**:
        1.  执行常规的语义吸附（Semantic Adsorption）判定与溢出接力判定。
        2.  决定是吸附到当前 Block 还是开启新 Block。

---

### 4. 关键处理流程 (Processing Workflow)

#### 4.1 MTP 日志解析器 (MTPLogParser)
在感知层内部引入一个新的工具类，负责将 Agent 的 `raw_response` 拆解。

*   **输入**: 包含 `⟪...⟫` 和 `<mtp_response>` 的原始 assistant response 文本。
*   **输出**: 
    1.  `clean_text`: 移除了所有协议符号的自然语言。
    2.  `traces`: (如果 Kernel 未传入，可在此处补救解析)。

#### 4.2 Librarian 的视角变化
重构后，Librarian 接收到的 `GenerationRequest` 将包含两种模式的数据：

*   **模式 A: 常规对话 (Normal Batch)**
    *   `Focus`: None
    *   `Block Content`: User Query + Clean Response + Traces.
    *   *行为*: 执行标准的总结与摘要。

*   **模式 B: 主动记忆 (Urgent Batch)**
    *   `Focus`: `WriteFocus(content="...", reason="...")`
    *   `Block Content`: 同上 (包含了触发 WRITE 指令的完整对话上下文)。
    *   *行为*: **以 Focus 内容为主体**，以 Block Content 为背景校验材料，执行高置信度的记忆入库。

---

### 5. 总结 (Conclusion)

通过本次重构，我们实现了：
1.  **数据完整性**：确保 `WRITE/UPDATE` 指令触发时，Agent 的完整上下文（包括为何要 Write 的前言后语）都被正确捕获。
2.  **关注点分离**：Koakuma (Kernel) 负责执行，Librarian.Perception 负责清洗，Librarian.Generation 负责归档。
3.  **系统鲁棒性**：统一了普通对话与指令对话的处理管道，消除了竞态条件。
