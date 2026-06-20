# 5 核心功能 II：记忆感�?(The Perception Layer)

> **[归属分身：大图书馆本�?(Librarian Core)]**

本章定义系统如何作为"感官"实时监听、解析和组织来自不同来源的原始对话流。这�?HiveMemory 的第一道工序，负责将混沌的对话流转化为有序的语义单元，并在适当时机唤醒 Librarian 进行记忆沉淀�?

## 5.0 模块概览

```
src/hivememory/engines/perception/
├── semantic_flow_perception_layer.py  # SemanticFlowPerceptionLayer / MMU 主入口
├── trigger_manager.py                 # TriggerManager / 统一结算调度
├── relay_controller.py                # SimpleRelayController / LLMRelayController / 页折叠摘要
├── context_converter.py               # ContextConverter / Block 到 StreamMessage 转换
├── models.py                          # 核心数据模型（LogicalBlock / SemanticBuffer 等）
└── interfaces.py                      # 抽象接口定义
```

感知层经历了多次重大演进：从最初的简单消息队列，到引入语义吸附的统一语义流架构，再到 MTP 协议适配后的 LogicalBlock 重构，最终升级为完整�?**MMU（内存管理单元）**，承担起整个短期记忆系统的管理职责�?

***

## 5.1 三级记忆架构与感知层定位

感知层在整个系统中扮�?**短期记忆（STM�?* 的角色，是三级记忆架构的第一层：

| 层级 | 名称 | 对应组件 | 类比 |
| :--- | :--- | :--- | :--- |
| **STM** | 短期记忆 | Perception Layer (感知�? | 工作�?(Workbench) |
| **MTM** | 中期记忆 | Generation Engine (生成引擎) | 海马�?(Hippocampus) |
| **LTM** | 长期记忆 | Lifecycle Engine (生命周期引擎) | 大脑皮层 (Cortex) |

感知层明�?*不独�?*，而是作为 Librarian Core（帕秋莉）的前置短期记忆引擎。当工作台上的任务告一段落（Swap-out）时，异步唤�?Generation Engine 将对话提炼为结构化的记忆原子�?

***

## 5.2 核心数据结构

### 5.2.1 逻辑原子�?(LogicalBlock)

`LogicalBlock` 是感知层的最小处理单元，代表一轮完整的交互。其结构经历了两次重大演进：

**v1.0（原始版�?*：基�?Thought-Tool-Observation 三元组的执行链结构�?

**v3.0（当前版，MTP 适配后）**：废�?`execution_chain`，引�?`semantic_traces`，职责从"拼接碎片"转变�?语义承载"�?

```python
class LogicalBlock(BaseModel):
    # --- 基础上下�?---
    user_query: str              # 原始用户问题
    rewritten_query: Optional[str]  # Gateway 指代消解后的意图锚点

    # --- 语义轨迹 (替代 Execution Chain) ---
    semantic_traces: List[TraceItem]  # 经过清洗和降维的 MTP 操作摘要

    # --- 响应数据 ---
    raw_response: str            # 包含 MTP 指令�?XML 的完整原始文�?
    clean_response: str          # 去除 MTP 噪音后的纯净回复（用户可见版本）

    # --- 控制信号 ---
    priority: str                # "NORMAL" | "URGENT"
    write_focus: Optional[WriteFocus]   # 携带 WRITE 指令的核心素�?
    update_focus: Optional[UpdateFocus] # 携带 UPDATE 指令的修改意�?

    # --- 辅助信息 ---
    worth_saving: Optional[bool] # Gateway 记忆价值判�?
    total_tokens: int
    block_id: str
```

**语义锚点优先�?*（`anchor_text` 属性）�?
1. 优先使用 `rewritten_query`（Gateway 指代消解后的查询�?
2. 回退�?`user_query`（原始用户问题）
3. 回退�?`user_block.content`（Legacy 模式�?

#### Semantic Trace 清洗策略

`TraceItem` �?Librarian 提供高信噪比的行为摘要：

| MTP 指令 | 清洗策略 | Trace 示例 |
| :--- | :--- | :--- |
| **READ** | **折叠** �?仅记录查阅动作和目标 | `{"action": "READ", "target": "mem_auth_doc"}` |
| **SEARCH** | **保留** �?记录 Agent 的探索意�?| `{"action": "SEARCH", "query": "docker config"}` |
| **RUN** | **摘要** �?记录副作用操作及状�?| `{"action": "RUN", "tool": "sys_write_file", "status": "success"}` |
| **WRITE/UPDATE** | **不生�?Trace** �?作为控制信号处理 | *(N/A)* |
| **XML 响应** | **丢弃** �?`<mtp_response>` 标签内容不进�?Trace | *(N/A)* |

### 5.2.2 话题�?(SemanticBuffer / TopicSegment)

`SemanticBuffer` 代表一个独立的讨论线程，是上下文隔离的物理边界�?

```python
class SemanticBuffer(BaseModel):
    topic_id: str               # 话题唯一标识（主键）
    identity: Identity          # 归属元数据（权限控制�?
    title: str                  # 话题标题（由 TheEye �?Kernel 异步生成�?
    state_summary: str          # 页折叠后的状态摘要（伪无限上下文基底�?
    blocks: List[LogicalBlock]  # 已闭合的 LogicalBlock 列表（页表）
    state: BufferState          # IDLE / PROCESSING / FLUSHING
    last_update: float          # 最后写入时�?
    last_accessed_at: float     # 最后访问时间（LRU 驱逐依据）
    total_tokens: int           # �?Token 数（水位线监控）
```

### 5.2.3 交互载荷 (InteractionPayload)

`InteractionPayload` �?Kernel �?Perception 的原子传输包，在 Kernel 完成一轮生成循环后提交�?

```python
class InteractionPayload(BaseModel):
    user_message: str           # 原始用户消息
    assistant_message: str      # 包含 MTP 指令的完�?assistant 文本
    mtp_traces: List[TraceItem] # �?Patchouli finalize 从结构化轮次事件归约得到
    write_focus: Optional[WriteFocus]   # WRITE 指令素材（挂载在 Payload 上）
    update_focus: Optional[UpdateFocus] # UPDATE 指令意图
    identity: Identity
    rewritten_query: Optional[str]  # Gateway 重写后的查询
    worth_saving: Optional[bool]    # Gateway 价值判�?
```

这一设计解决了旧�?旁路触发"导致的上下文丢失问题——WRITE/UPDATE 不再通过独立通道触发 Flush，而是随完整的 Payload 一起提交，确保当轮对话的完整上下文被正确捕获�?

***

## 5.3 MMU 架构：多话题并发管理

### 5.3.1 设计理念

感知层采用操作系统的**段页式内存管理（Segmented Paging�?*思想�?

- **摒弃全局 History**：Kernel 不再持有单一的线�?`history`，短期记忆的控制权全权交由感知层�?
- **单窗口多线程（Single-Window, Multi-Track�?*：用户只需面对一个无边界的聊天窗口，系统在后台自动维护多个独立的**话题段（Topic Segments�?*，每个话题拥有绝对纯净的上下文�?
- **动态换入换出（Swap-in/out�?*：感知层作为 MMU，根�?TheEye 的路由决策，动态将对应的话�?Buffer 换入 Kernel 的工作区；对于长期不活跃的话题，则将其换出并移交 Generation Engine 进行固化�?

### 5.3.2 ShortTermMemoryStore（话题状态存储 / MMU）

`ShortTermMemoryStore` 是短期记忆的中央调度器，管理活跃话题池的生命周期。

```python
class ShortTermMemoryStore:
    _buffers: Dict[str, SemanticBuffer]   # 活跃话题池，key �?topic_id
    _user_index: Dict[str, Set[str]]      # 用户索引：user_id:agent_id -> Set[topic_id]
    max_resident_topics: int              # 最大驻留话题数（默�?5�?
    _last_active_topic_id: Optional[str]  # 最后活跃话题（manual_trigger 回退用）
```

核心操作�?

| 方法 | 说明 |
| :--- | :--- |
| `create_buffer(identity, title)` | 创建新话题段，生成唯一 `topic_id` |
| `get_buffer(topic_id)` | 获取话题段，自动更新 `last_accessed_at` |
| `pop_buffer(topic_id)` | 移除并返回话题段（换出） |
| `add_block(topic_id, block)` | 向话题段追加 LogicalBlock |
| `fold_blocks(topic_id, summary, retain_count)` | 原子化页折叠：保留最�?N �?blocks，写入摘�?|
| `get_lru_buffer()` | 获取最久未访问的话题段（LRU 驱逐候选） |
| `needs_eviction()` | 检查活跃池是否已满 |

### 5.3.3 SemanticFlowPerceptionLayer（MMU 主入口）

`SemanticFlowPerceptionLayer` 是感知层的对外接口，编排所有子组件�?

```
route_and_ingest(topic_id, payload)
    �?
    ├── topic_id == "NEW_TOPIC"?
    �?      └── _ensure_topic_slot_and_create()
    �?              └── needs_eviction? �?_evict_lru_topic()
    �?
    ├── ingest_payload(payload, topic_id)
    �?      ├── MTPLogParser.parse() �?clean_text + fallback_traces
    �?      ├── 构建 LogicalBlock
    �?      ├── 信号检�?
    �?      �?      URGENT (write/update_focus) �?add_block �?resolve_topic(MTP_WRITE/UPDATE)
    �?      �?      NORMAL �?add_block
    �?      └── _maybe_fold_pages() �?TOKEN_OVERFLOW 检�?
    �?
    └── short_term_store.set_last_active_topic(topic_id)
```

对外暴露的关键方法：

| 方法 | 说明 |
| :--- | :--- |
| `route_and_ingest(topic_id, payload)` | MMU 核心：路由到指定话题并摄入载�?|
| `get_active_topics_snapshots(identity)` | 获取话题快照列表（含最后一轮对话） |
| `get_topic_context(topic_id)` | 获取话题完整上下文用�?Prompt 组装 |
| `manual_trigger(topic_id)` | 手动触发话题结算（Archive + Compact，不 Evict�?|
| `scan_idle_buffers_now()` | 立即扫描�?flush 超时话题 |

***

## 5.4 数据摄入管道

### 5.4.1 MTP 日志解析器（MTPLogParser�?

在感知层内部，`MTPLogParser` 负责�?Agent �?`raw_response` 拆解�?

- **输入**：包�?`�?..⟫` �?`<mtp_response>` 的原�?assistant response 文本
- **输出**�?
  1. `clean_text`：移除了所有协议符号的自然语言（用户可见版本）
  2. `fallback_traces`：如�?Kernel 未传�?traces，在此处补救解析

### 5.4.2 摄入流程

```
InteractionPayload 到达
        �?
        �?
MTPLogParser.parse(assistant_message)
        �?
        ├── clean_text
        └── fallback_traces
        �?
        �?
构建 LogicalBlock
        �?
        ├── 计算 total_tokens
        �?
        �?
信号检�?(Signal Check)
        �?
        ├── URGENT (write_focus / update_focus 不为 None)
        �?      ├── block.priority = "URGENT"
        �?      ├── add_block(topic_id, block)
        �?      └── resolve_topic(MTP_WRITE / MTP_UPDATE)
        �?
        └── NORMAL
                ├── add_block(topic_id, block)
                └── _maybe_fold_pages() �?TOKEN_OVERFLOW 检�?
```

***


### 5.5.2 停用词列�?

默认停用词（触发强制吸附）：

```python
DEFAULT_SHORT_TEXT_STOP_WORDS = {
    "不对", "报错�?, "错了", "错误", "�?, "�?, "�?,
    "�?, "�?, "可以", "继续", "好的", "然后�?,
    "ok", "okay", "yes", "no", "yeah", "yep", "nope",
    "continue", "go on", "sure", "alright", "next",
}
```

***

## 5.6 页折叠与上下文防爆（Page Folding�?

### 5.6.1 双重水位�?

每个 `SemanticBuffer` 维护 token 水位线监控：

- **高水位线（High Watermark�?*：`fold_token_threshold`（建�?32k-64k tokens），触发页折叠的软阈值，�?MTP 注入与新生成预留充足 Buffer�?
- **物理极限（Physical Limit�?*：LLM 所能承受的最�?Context 窗口（如 128k tokens）�?

### 5.6.2 折叠算法

�?`buffer.total_tokens > fold_token_threshold` 时，触发 `TOKEN_OVERFLOW` 结算�?

1. **挂起生成（Suspend�?*：暂停当�?LLM 续写请求
2. **状态提取（State Extraction�?*：取出旧�?`state_summary` 和全�?最旧的 N �?`LogicalBlock`
3. **高速压缩（Compaction�?*：调�?`RelayController.generate_summary()` 生成新摘�?
4. **状态替换（State Replacement�?*：覆�?`state_summary`，清空已压缩的旧 blocks，更�?`total_tokens`

### 5.6.3 RelayController 实现

| 实现�?| 摘要策略 | 适用场景 |
| :--- | :--- | :--- |
| `SimpleRelayController` | 基于规则统计（请求数、工具列表、最近查询） | 默认，低延迟 |
| `LLMRelayController` | 调用 LLM 生成语义化摘要（预留接口，当前回退�?Simple�?| 高质量摘要需�?|

`SimpleRelayController` 生成的摘要格式：
```
处理�?N 个用户请求；使用了工�? search, read；共 M tokens；最�? <最后一个查询片�?
```

多次折叠时，新摘要与旧摘要以 `---` 分隔累积拼接，形�?*伪无限上下文基底**�?

### 5.6.4 Prompt 组装（冰山结构）

折叠完成后，Kernel 组装�?Worker Agent 的最�?Prompt�?

```markdown
<system_prompt>
... (基础人设�?MTP 协议说明) ...
<working_memory>
[当前话题状态]: {topic.state_summary}
</working_memory>
</system_prompt>

[保留下来的最�?2-3 轮未折叠的对话，保证 Agent 仍能顺畅接话]
```

***

## 5.7 话题生命周期与结算调�?

### 5.7.1 TriggerManager 决策矩阵

`TriggerManager` 是统一的话题结算调度器，废弃了原有的单一 `flush` 逻辑，将 Buffer 维护解耦为三个正交的原子操作：

- **Archive（归档）**：将 blocks 打包发送给 Librarian，异步非阻塞（fire-and-forget�?
- **Compact（压缩）**：生�?`state_summary` 并清�?blocks，同步阻�?
- **Evict（驱逐）**：从活跃池移�?buffer，释放内�?

根据触发原因查表决定执行哪些操作�?

| 触发原因 | 业务场景 | Archive | Compact | Evict | 最终状�?|
| :--- | :--- | :---: | :---: | :---: | :--- |
| `TOKEN_OVERFLOW` | 话没聊完，逼近 Token 水位�?| �?| �?| �?| 存活（含新摘要） |
| `IDLE_TIMEOUT` | 话题闲置超时，自然冷�?| �?| �?| �?| 被销�?|
| `LRU_EVICTION` | 活跃话题超限，被新话题挤�?| �?| �?| �?| 被销�?|
| `MTP_WRITE` | Agent 主动发出 WRITE 指令 | �?| �?| �?| 存活（含新摘要） |
| `MTP_UPDATE` | Agent 主动发出 UPDATE 指令 | �?| �?| �?| 存活（含新摘要） |
| `MANUAL` | 用户手动触发 `/save` | �?| �?| �?| 存活（含新摘要） |

### 5.7.2 Archive �?worth_saving 过滤

Archive 操作在发送前会过滤掉 `worth_saving=False` �?blocks（由 Gateway 标记的无价值对话）�?

```python
blocks_to_archive = [b for b in blocks_snapshot if b.worth_saving is not False]
# worth_saving=None 时保留（Gateway 离线或异常时不影响冷链路�?
```

### 5.7.3 换出触发器（Swap-out Triggers�?

话题从活跃池换出的三种条件：

1. **空闲超时休眠（Idle Hibernate�?*：`last_accessed_at` 距今超过 `idle_timeout_seconds`（默�?900 秒）
2. **空间挤压驱逐（LRU Eviction�?*：活跃池已满（`max_resident_topics`），�?TheEye 路由出新话题
3. **显式结束（Explicit Close�?*：用户发送系统指令，�?TheEye 检测到明确的告别语

***

## 5.8 Prompt 上下文组�?

`get_topic_context()` �?Kernel 提供话题的完整上下文�?

```python
{
    "state_summary": str,          # 折叠摘要（伪无限上下文基底）
    "blocks": List[LogicalBlock],  # 最�?max_recent_blocks �?blocks
    "total_tokens": int,
    "title": str,
}
```

Kernel 使用此数据组�?Prompt：`[System + state_summary + MTP_Menu + recent_blocks]`

***

## 5.9 配置参�?

### SemanticFlowPerceptionConfig

| 配置�?| 类型 | 默认�?| 说明 |
| :--- | :--- | :--- | :--- |
| `max_resident_topics` | `int` | `5` | 活跃话题池最大容�?|
| `idle_timeout_seconds` | `int` | `900` | 话题空闲超时阈值（秒） |
| `scan_interval_seconds` | `int` | `60` | 空闲扫描间隔（秒�?|
| `fold_token_threshold` | `int` | `32768` | 页折叠触发阈值（tokens�?|

