# 5 核心功能 II：记忆感知 (The Perception Layer)

> **[归属分身：大图书馆本体 (Librarian Core)]**

本章定义系统如何作为"感官"实时监听、解析和组织来自不同来源的原始对话流。这是 HiveMemory 的第一道工序，负责将混沌的对话流转化为有序的语义单元，并在适当时机唤醒 Librarian 进行记忆沉淀。

## 5.0 模块概览

```
src/hivememory/engines/perception/
├── semantic_flow_perception_layer.py  # SemanticFlowPerceptionLayer — MMU 主入口
├── buffer_manager.py                  # SemanticBufferManager — 话题池 CRUD + LRU
├── trigger_manager.py                 # TriggerManager — 统一结算调度器
├── relay_controller.py                # SimpleRelayController / LLMRelayController — 页折叠摘要
├── semantic_adsorber.py               # SemanticBoundaryAdsorber — 三级语义吸附判定
├── grey_area_arbiter.py               # GreyAreaArbiter — 灰度区间仲裁器
├── context_converter.py               # ContextConverter — Block → StreamMessage 转换
├── models.py                          # 核心数据模型（LogicalBlock / SemanticBuffer 等）
└── interfaces.py                      # 抽象接口定义
```

感知层经历了多次重大演进：从最初的简单消息队列，到引入语义吸附的统一语义流架构，再到 MTP 协议适配后的 LogicalBlock 重构，最终升级为完整的 **MMU（内存管理单元）**，承担起整个短期记忆系统的管理职责。

***

## 5.1 三级记忆架构与感知层定位

感知层在整个系统中扮演 **短期记忆（STM）** 的角色，是三级记忆架构的第一层：

| 层级 | 名称 | 对应组件 | 类比 |
| :--- | :--- | :--- | :--- |
| **STM** | 短期记忆 | Perception Layer (感知层) | 工作台 (Workbench) |
| **MTM** | 中期记忆 | Generation Engine (生成引擎) | 海马体 (Hippocampus) |
| **LTM** | 长期记忆 | Lifecycle Engine (生命周期引擎) | 大脑皮层 (Cortex) |

感知层明确**不独立**，而是作为 Librarian Core（帕秋莉）的前置短期记忆引擎。当工作台上的任务告一段落（Swap-out）时，异步唤醒 Generation Engine 将对话提炼为结构化的记忆原子。

***

## 5.2 核心数据结构

### 5.2.1 逻辑原子块 (LogicalBlock)

`LogicalBlock` 是感知层的最小处理单元，代表一轮完整的交互。其结构经历了两次重大演进：

**v1.0（原始版）**：基于 Thought-Tool-Observation 三元组的执行链结构。

**v3.0（当前版，MTP 适配后）**：废弃 `execution_chain`，引入 `semantic_traces`，职责从"拼接碎片"转变为"语义承载"：

```python
class LogicalBlock(BaseModel):
    # --- 基础上下文 ---
    user_query: str              # 原始用户问题
    rewritten_query: Optional[str]  # Gateway 指代消解后的意图锚点

    # --- 语义轨迹 (替代 Execution Chain) ---
    semantic_traces: List[TraceItem]  # 经过清洗和降维的 MTP 操作摘要

    # --- 响应数据 ---
    raw_response: str            # 包含 MTP 指令和 XML 的完整原始文本
    clean_response: str          # 去除 MTP 噪音后的纯净回复（用户可见版本）

    # --- 控制信号 ---
    priority: str                # "NORMAL" | "URGENT"
    write_focus: Optional[WriteFocus]   # 携带 WRITE 指令的核心素材
    update_focus: Optional[UpdateFocus] # 携带 UPDATE 指令的修改意图

    # --- 辅助信息 ---
    worth_saving: Optional[bool] # Gateway 记忆价值判断
    total_tokens: int
    block_id: str
```

**语义锚点优先级**（`anchor_text` 属性）：
1. 优先使用 `rewritten_query`（Gateway 指代消解后的查询）
2. 回退到 `user_query`（原始用户问题）
3. 回退到 `user_block.content`（Legacy 模式）

#### Semantic Trace 清洗策略

`TraceItem` 为 Librarian 提供高信噪比的行为摘要：

| MTP 指令 | 清洗策略 | Trace 示例 |
| :--- | :--- | :--- |
| **READ** | **折叠** — 仅记录查阅动作和目标 | `{"action": "READ", "target": "mem_auth_doc"}` |
| **SEARCH** | **保留** — 记录 Agent 的探索意图 | `{"action": "SEARCH", "query": "docker config"}` |
| **RUN** | **摘要** — 记录副作用操作及状态 | `{"action": "RUN", "tool": "sys_write_file", "status": "success"}` |
| **WRITE/UPDATE** | **不生成 Trace** — 作为控制信号处理 | *(N/A)* |
| **XML 响应** | **丢弃** — `<mtp_response>` 标签内容不进入 Trace | *(N/A)* |

### 5.2.2 话题段 (SemanticBuffer / TopicSegment)

`SemanticBuffer` 代表一个独立的讨论线程，是上下文隔离的物理边界：

```python
class SemanticBuffer(BaseModel):
    topic_id: str               # 话题唯一标识（主键）
    identity: Identity          # 归属元数据（权限控制）
    title: str                  # 话题标题（由 TheEye 或 Kernel 异步生成）
    state_summary: str          # 页折叠后的状态摘要（伪无限上下文基底）
    blocks: List[LogicalBlock]  # 已闭合的 LogicalBlock 列表（页表）
    topic_kernel_vector: Optional[List[float]]  # 话题核心向量（语义吸附用）
    state: BufferState          # IDLE / PROCESSING / FLUSHING
    last_update: float          # 最后写入时间
    last_accessed_at: float     # 最后访问时间（LRU 驱逐依据）
    total_tokens: int           # 总 Token 数（水位线监控）
```

### 5.2.3 交互载荷 (InteractionPayload)

`InteractionPayload` 是 Kernel → Perception 的原子传输包，在 Kernel 完成一轮生成循环后提交：

```python
class InteractionPayload(BaseModel):
    user_message: str           # 原始用户消息
    assistant_message: str      # 包含 MTP 指令的完整 assistant 文本
    mtp_traces: List[TraceItem] # 由 Patchouli finalize 从结构化轮次事件归约得到
    write_focus: Optional[WriteFocus]   # WRITE 指令素材（挂载在 Payload 上）
    update_focus: Optional[UpdateFocus] # UPDATE 指令意图
    identity: Identity
    rewritten_query: Optional[str]  # Gateway 重写后的查询
    worth_saving: Optional[bool]    # Gateway 价值判断
```

这一设计解决了旧版"旁路触发"导致的上下文丢失问题——WRITE/UPDATE 不再通过独立通道触发 Flush，而是随完整的 Payload 一起提交，确保当轮对话的完整上下文被正确捕获。

***

## 5.3 MMU 架构：多话题并发管理

### 5.3.1 设计理念

感知层采用操作系统的**段页式内存管理（Segmented Paging）**思想：

- **摒弃全局 History**：Kernel 不再持有单一的线性 `history`，短期记忆的控制权全权交由感知层。
- **单窗口多线程（Single-Window, Multi-Track）**：用户只需面对一个无边界的聊天窗口，系统在后台自动维护多个独立的**话题段（Topic Segments）**，每个话题拥有绝对纯净的上下文。
- **动态换入换出（Swap-in/out）**：感知层作为 MMU，根据 TheEye 的路由决策，动态将对应的话题 Buffer 换入 Kernel 的工作区；对于长期不活跃的话题，则将其换出并移交 Generation Engine 进行固化。

### 5.3.2 SemanticBufferManager（话题管理器 / MMU）

`SemanticBufferManager` 是短期记忆的中央调度器，管理活跃话题池的生命周期：

```python
class SemanticBufferManager:
    _buffers: Dict[str, SemanticBuffer]   # 活跃话题池，key 为 topic_id
    _user_index: Dict[str, Set[str]]      # 用户索引：user_id:agent_id -> Set[topic_id]
    max_resident_topics: int              # 最大驻留话题数（默认 5）
    _last_active_topic_id: Optional[str]  # 最后活跃话题（manual_trigger 回退用）
```

核心操作：

| 方法 | 说明 |
| :--- | :--- |
| `create_buffer(identity, title)` | 创建新话题段，生成唯一 `topic_id` |
| `get_buffer(topic_id)` | 获取话题段，自动更新 `last_accessed_at` |
| `pop_buffer(topic_id)` | 移除并返回话题段（换出） |
| `add_block(topic_id, block)` | 向话题段追加 LogicalBlock |
| `fold_blocks(topic_id, summary, retain_count)` | 原子化页折叠：保留最近 N 个 blocks，写入摘要 |
| `get_lru_buffer()` | 获取最久未访问的话题段（LRU 驱逐候选） |
| `needs_eviction()` | 检查活跃池是否已满 |

### 5.3.3 SemanticFlowPerceptionLayer（MMU 主入口）

`SemanticFlowPerceptionLayer` 是感知层的对外接口，编排所有子组件：

```
route_and_ingest(topic_id, payload)
    │
    ├── topic_id == "NEW_TOPIC"?
    │       └── _ensure_topic_slot_and_create()
    │               └── needs_eviction? → _evict_lru_topic()
    │
    ├── ingest_payload(payload, topic_id)
    │       ├── MTPLogParser.parse() → clean_text + fallback_traces
    │       ├── 构建 LogicalBlock
    │       ├── 信号检查:
    │       │       URGENT (write/update_focus) → add_block → resolve_topic(MTP_WRITE/UPDATE)
    │       │       NORMAL → add_block
    │       └── _maybe_fold_pages() → TOKEN_OVERFLOW 检查
    │
    └── buffer_manager.set_last_active_topic(topic_id)
```

对外暴露的关键方法：

| 方法 | 说明 |
| :--- | :--- |
| `route_and_ingest(topic_id, payload)` | MMU 核心：路由到指定话题并摄入载荷 |
| `get_active_topics_snapshots(identity)` | 获取话题快照列表（含最后一轮对话） |
| `get_topic_context(topic_id)` | 获取话题完整上下文用于 Prompt 组装 |
| `manual_trigger(topic_id)` | 手动触发话题结算（Archive + Compact，不 Evict） |
| `scan_idle_buffers_now()` | 立即扫描并 flush 超时话题 |

***

## 5.4 数据摄入管道

### 5.4.1 MTP 日志解析器（MTPLogParser）

在感知层内部，`MTPLogParser` 负责将 Agent 的 `raw_response` 拆解：

- **输入**：包含 `⟪...⟫` 和 `<mtp_response>` 的原始 assistant response 文本
- **输出**：
  1. `clean_text`：移除了所有协议符号的自然语言（用户可见版本）
  2. `fallback_traces`：如果 Kernel 未传入 traces，在此处补救解析

### 5.4.2 摄入流程

```
InteractionPayload 到达
        │
        ▼
MTPLogParser.parse(assistant_message)
        │
        ├── clean_text
        └── fallback_traces
        │
        ▼
构建 LogicalBlock
        │
        ├── 计算 total_tokens
        │
        ▼
信号检查 (Signal Check)
        │
        ├── URGENT (write_focus / update_focus 不为 None)
        │       ├── block.priority = "URGENT"
        │       ├── add_block(topic_id, block)
        │       └── resolve_topic(MTP_WRITE / MTP_UPDATE)
        │
        └── NORMAL
                ├── add_block(topic_id, block)
                └── _maybe_fold_pages() → TOKEN_OVERFLOW 检查
```

***

## 5.5 语义吸附与话题切分

> **注**：Phase 4.5 重构后，话题路由已由 TheEye（Gateway）通过 Agentic Dispatcher 完成，`SemanticBoundaryAdsorber` 主要用于旧版兼容路径和辅助判定。当前主流程中，`route_and_ingest` 直接接收 TheEye 路由好的 `topic_id`，不再依赖 Adsorber 进行切分决策。

### 5.5.1 三级判定流水线

`SemanticBoundaryAdsorber` 实现三阶段无状态处理管道：

```
新 LogicalBlock 到达
        │
        ▼
Step 1: 启发式强吸附 (Heuristic Filtering)
        │
        ├── anchor_text 极短（< short_text_threshold tokens）
        │   且为停用词（"继续"、"报错了"、"ok" 等）
        │       └── FORCE_ADSORB（不更新 Topic Kernel，防止噪音污染）
        │
        └── 否则 → Step 2
        │
        ▼
Step 2: 向量初筛 (Vector Screening)
        │
        ├── 计算 cosine(V_new, V_kernel)
        │
        ├── Score > high_threshold (0.75) → ADSORB & 更新 Kernel
        ├── Score < low_threshold  (0.40) → SPLIT（触发 FlushEvent）
        └── 模糊地带 → Step 3
        │
        ▼
Step 3: 智能仲裁 (Intelligent Arbitration)
        │
        ├── 调用 GreyAreaArbiter（Cross-Encoder / Local SLM）
        ├── should_continue_topic(prev_context, current_query, similarity)
        │
        ├── YES → ADSORB
        └── NO  → SPLIT（触发 FlushEvent）
```

### 5.5.2 话题核心向量（Topic Kernel）

为避免稀释效应，不计算整个 Buffer 的平均向量，而是维护**指数移动平均（EMA）**：

```
new_kernel = α × V_new + (1 - α) × V_old_kernel
```

其中 `α = ema_alpha`（默认 0.3）。`compute_new_topic_kernel()` 是纯函数，不修改 buffer，由调用方负责通过 `update_metadata()` 写回。

### 5.5.3 停用词列表

默认停用词（触发强制吸附）：

```python
DEFAULT_SHORT_TEXT_STOP_WORDS = {
    "不对", "报错了", "错了", "错误", "嗯", "哦", "啊",
    "好", "行", "可以", "继续", "好的", "然后呢",
    "ok", "okay", "yes", "no", "yeah", "yep", "nope",
    "continue", "go on", "sure", "alright", "next",
}
```

***

## 5.6 页折叠与上下文防爆（Page Folding）

### 5.6.1 双重水位线

每个 `SemanticBuffer` 维护 token 水位线监控：

- **高水位线（High Watermark）**：`fold_token_threshold`（建议 32k-64k tokens），触发页折叠的软阈值，为 MTP 注入与新生成预留充足 Buffer。
- **物理极限（Physical Limit）**：LLM 所能承受的最大 Context 窗口（如 128k tokens）。

### 5.6.2 折叠算法

当 `buffer.total_tokens > fold_token_threshold` 时，触发 `TOKEN_OVERFLOW` 结算：

1. **挂起生成（Suspend）**：暂停当前 LLM 续写请求
2. **状态提取（State Extraction）**：取出旧的 `state_summary` 和全部/最旧的 N 个 `LogicalBlock`
3. **高速压缩（Compaction）**：调用 `RelayController.generate_summary()` 生成新摘要
4. **状态替换（State Replacement）**：覆盖 `state_summary`，清空已压缩的旧 blocks，更新 `total_tokens`

### 5.6.3 RelayController 实现

| 实现类 | 摘要策略 | 适用场景 |
| :--- | :--- | :--- |
| `SimpleRelayController` | 基于规则统计（请求数、工具列表、最近查询） | 默认，低延迟 |
| `LLMRelayController` | 调用 LLM 生成语义化摘要（预留接口，当前回退到 Simple） | 高质量摘要需求 |

`SimpleRelayController` 生成的摘要格式：
```
处理了 N 个用户请求；使用了工具: search, read；共 M tokens；最近: <最后一个查询片段>
```

多次折叠时，新摘要与旧摘要以 `---` 分隔累积拼接，形成**伪无限上下文基底**。

### 5.6.4 Prompt 组装（冰山结构）

折叠完成后，Kernel 组装给 Worker Agent 的最终 Prompt：

```markdown
<system_prompt>
... (基础人设与 MTP 协议说明) ...
<working_memory>
[当前话题状态]: {topic.state_summary}
</working_memory>
</system_prompt>

[保留下来的最近 2-3 轮未折叠的对话，保证 Agent 仍能顺畅接话]
```

***

## 5.7 话题生命周期与结算调度

### 5.7.1 TriggerManager 决策矩阵

`TriggerManager` 是统一的话题结算调度器，废弃了原有的单一 `flush` 逻辑，将 Buffer 维护解耦为三个正交的原子操作：

- **Archive（归档）**：将 blocks 打包发送给 Librarian，异步非阻塞（fire-and-forget）
- **Compact（压缩）**：生成 `state_summary` 并清空 blocks，同步阻塞
- **Evict（驱逐）**：从活跃池移除 buffer，释放内存

根据触发原因查表决定执行哪些操作：

| 触发原因 | 业务场景 | Archive | Compact | Evict | 最终状态 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `TOKEN_OVERFLOW` | 话没聊完，逼近 Token 水位线 | ❌ | ✅ | ❌ | 存活（含新摘要） |
| `IDLE_TIMEOUT` | 话题闲置超时，自然冷却 | ✅ | ❌ | ✅ | 被销毁 |
| `LRU_EVICTION` | 活跃话题超限，被新话题挤出 | ✅ | ❌ | ✅ | 被销毁 |
| `MTP_WRITE` | Agent 主动发出 WRITE 指令 | ✅ | ✅ | ❌ | 存活（含新摘要） |
| `MTP_UPDATE` | Agent 主动发出 UPDATE 指令 | ✅ | ✅ | ❌ | 存活（含新摘要） |
| `MANUAL` | 用户手动触发 `/save` | ✅ | ✅ | ❌ | 存活（含新摘要） |

### 5.7.2 Archive 的 worth_saving 过滤

Archive 操作在发送前会过滤掉 `worth_saving=False` 的 blocks（由 Gateway 标记的无价值对话）：

```python
blocks_to_archive = [b for b in blocks_snapshot if b.worth_saving is not False]
# worth_saving=None 时保留（Gateway 离线或异常时不影响冷链路）
```

### 5.7.3 换出触发器（Swap-out Triggers）

话题从活跃池换出的三种条件：

1. **空闲超时休眠（Idle Hibernate）**：`last_accessed_at` 距今超过 `idle_timeout_seconds`（默认 900 秒）
2. **空间挤压驱逐（LRU Eviction）**：活跃池已满（`max_resident_topics`），且 TheEye 路由出新话题
3. **显式结束（Explicit Close）**：用户发送系统指令，或 TheEye 检测到明确的告别语

***

## 5.8 Prompt 上下文组装

`get_topic_context()` 为 Kernel 提供话题的完整上下文：

```python
{
    "state_summary": str,          # 折叠摘要（伪无限上下文基底）
    "blocks": List[LogicalBlock],  # 最近 max_recent_blocks 个 blocks
    "total_tokens": int,
    "title": str,
}
```

Kernel 使用此数据组装 Prompt：`[System + state_summary + MTP_Menu + recent_blocks]`

***

## 5.9 配置参考

### SemanticFlowPerceptionConfig

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `max_resident_topics` | `int` | `5` | 活跃话题池最大容量 |
| `idle_timeout_seconds` | `int` | `900` | 话题空闲超时阈值（秒） |
| `scan_interval_seconds` | `int` | `60` | 空闲扫描间隔（秒） |
| `fold_token_threshold` | `int` | `32768` | 页折叠触发阈值（tokens） |

### SemanticAdsorberConfig

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `semantic_threshold_high` | `float` | `0.75` | 强相关阈值（直接吸附） |
| `semantic_threshold_low` | `float` | `0.40` | 强无关阈值（直接切分） |
| `short_text_threshold` | `int` | `5` | 短文本强吸附 token 阈值 |
| `ema_alpha` | `float` | `0.3` | 话题核心向量 EMA 系数 |
| `stop_words` | `Set[str]` | 默认列表 | 自定义停用词集合 |
| `arbiter.enabled` | `bool` | `True` | 是否启用灰度仲裁器 |
