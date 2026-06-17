# 8 核心功能 V：记忆生命周期管理 (Lifecycle Management)

> **[归属分身：大图书馆本体 (Librarian Core)]**

本章定义系统如何管理记忆的全生命周期，从诞生、强化、衰减到归档与唤醒。借鉴计算机存储架构（Register → L1/L2 Cache → RAM → Disk），HiveMemory 构建了三级记忆流水线，帕秋莉（Librarian）担任操作系统的 **Memory Management Unit (MMU)** 角色。

## 8.0 模块概览

```
src/hivememory/engines/lifecycle/
├── engine.py                # MemoryLifecycleEngine — 统一协调器
├── vitality.py              # VitalityCalculator — 生命力分数计算
├── reinforcement.py         # DynamicReinforcementEngine — 强化事件处理
├── archiver.py              # FileBasedArchiver — 冷存储管理
├── garbage_collector.py     # PeriodicGarbageCollector — 垃圾回收
├── models.py                # 数据模型（EventType / ReinforcementResult / ArchiveRecord）
└── interfaces.py            # 抽象接口（BaseMemoryArchiver / BaseGarbageCollector）
```

生命周期管理是 HiveMemory 的"自主神经系统"，负责在后台默默维护记忆库的健康状态，防止低价值噪音淹没高质量知识。

***

## 8.1 三级记忆流水线 (The Three-Tier Pipeline)

借鉴计算机存储架构，HiveMemory 将记忆分为三个层级，每层有不同的容量、速度和成本特性：

| 层级 | 名称 | 位置 | 状态 | 容量限制 | 策略 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L1** | Working Context<br>（短期记忆） | Agent 当前 Context Window | 极热 (Hot) | 受 LLM Token 限制<br>（如 128k） | FIFO（先进先出），随对话滚动消失 |
| **L2** | Active Vector Memory<br>（中期记忆/海马体） | 向量数据库<br>（Qdrant/Weaviate）<br>内存/高速索引区 | 温热 (Warm) | 受检索速度和云端成本限制<br>（如 100 万条） | **基于语义价值的 LRU**<br>检索系统的主战场 |
| **L3** | Archival Storage<br>（长期记忆/潜意识） | 低成本冷存储<br>（PostgreSQL / S3 / Blob Storage） | 冷 (Cold) | 无限 | 仅存储，不参与常规向量检索<br>只有通过特定精确指令才能"唤醒" |

### 设计理念

- **成本控制**：向量数据库（尤其是云托管的）通常按存储量和维度计费。将 90% 的不常用记忆放入廉价的 S3/SQL，只把 10% 的"高频高价值"记忆留在向量库，能节省一个数量级的成本。
- **减少幻觉**：过时的、低置信度的信息如果不清理，会成为检索时的"噪音"，误导 Agent。GC 机制本质上是一个**"主动遗忘"**过程，这对保持 Agent 的聪明程度至关重要。

***

## 8.2 记忆生命力模型 (Memory Vitality Model)

为了量化"哪条记忆该留，哪条该走"，我们定义一个核心指标：**记忆生命力分数 (Vitality Score, $V$)**。生命力分数按使用点刷新：检索和记忆 API 在返回前刷新命中的记忆，GC 则由 `MemoryLifecycleEngine.run_garbage_collection()` 在归档判断前刷新全量活跃记忆。

### 8.2.1 评分公式

$$V = (C \times I) \times D(t) \times 100 + A$$

| 组件 | 含义 | 取值范围 | 作用 |
| :--- | :--- | :--- | :--- |
| **$C$** | Confidence（置信度） | 0-1 | 来自 3.3 节。用户输入的 $C=1.0$，模型推理的 $C=0.6$。越真实的信息，生命力越顽强（不易被遗忘） |
| **$I$** | Intrinsic Value（固有价值） | 0-1 | 基于记忆类型。例如：代码 ($I=1.0$) > 事实 ($I=0.9$) > 闲聊 ($I=0.1$)。为记忆赋予相对的重要性 |
| **$D(t)$** | Time Decay（时间衰减函数） | 0-1 | $D(t) = e^{-\lambda \cdot t}$（指数衰减）或简单的 $\frac{1}{1 + \text{days\_elapsed}}$。随着时间推移，记忆自然淡化 |
| **$A$** | Access Boost（访问增强） | 0-max_boost | 每次命中与引用带来的加分（见 8.3）。$A = \min(\text{max\_boost}, \text{access\_count} \times \text{points\_per\_access})$ |

**最终分数范围**：0-100（乘以 100 将基础分数映射到百分制）

### 8.2.2 固有价值权重

`VitalityCalculator` 维护一个类型权重字典：

```python
INTRINSIC_VALUE_WEIGHTS = {
    MemoryType.CODE_SNIPPET: 1.0,
    MemoryType.FACT: 0.9,
    MemoryType.URL_RESOURCE: 0.8,
    MemoryType.REFLECTION: 0.7,
    MemoryType.USER_PROFILE: 0.6,
    MemoryType.WORK_IN_PROGRESS: 0.5,
}
```

### 8.2.3 时间衰减函数

默认使用指数衰减：

$$D(t) = e^{-\lambda \cdot t}$$

- $t=0$ 时：$D(0) = 1.0$（无衰减）
- $\lambda=0.01$，$t=100$ 天时：$D(100) \approx 0.37$

配置项 `decay_lambda`（默认 0.01）控制衰减速度。

***

## 8.3 动态强化算法 (Reinforcement Algorithm)

类似于 Cache 的 **"Hit"** 机制，但更复杂，因为我们要区分"检索到了"和"真的有帮助"。

### 8.3.1 事件类型与效果

`DynamicReinforcementEngine` 处理四种生命周期事件：

| 事件类型 | 触发场景 | 生命力调整 | 置信度调整 | 时间衰减重置 |
| :--- | :--- | :--- | :--- | :--- |
| **HIT** | 被动检索命中并注入 Context | +5 | 无 | 否 |
| **CITATION** | Agent 在回答中明确引用或 Tool 执行成功 | +20 | 无 | **是**（刷新 `updated_at`） |
| **FEEDBACK_POSITIVE** | 用户对 Agent 回答点赞或确认有效 | +50 | 无 | 否 |
| **FEEDBACK_NEGATIVE** | 用户反馈"不对"或"过时了" | -50 | $\times 0.5$ | 否 |

### 8.3.2 强化流程

```
MemoryEvent 到达
        │
        ▼
从存储获取记忆
        │
        ▼
记录当前状态 (previous_vitality, previous_confidence)
        │
        ▼
根据事件类型应用调整
        │
        ├── CITATION → 重置 updated_at（时间衰减归零）
        ├── FEEDBACK_NEGATIVE → confidence *= 0.5
        └── HIT / FEEDBACK_POSITIVE → 简单加成
        │
        ▼
更新访问元信息 (access_count +1, last_accessed_at = now)
        │
        ▼
重新计算生命力分数 (VitalityCalculator.calculate)
        │
        ▼
持久化到存储 (upsert_memory)
        │
        ▼
返回 ReinforcementResult (包含前后对比)
```

### 8.3.3 访问加成计算

```python
A = min(max_access_boost, access_count × points_per_access)
```

- `points_per_access`：默认 1.0（每次访问加 1 分）
- `max_access_boost`：默认 20.0（访问加成上限）

***

## 8.4 垃圾回收与归档策略 (GC & Archiving Strategy)

帕秋莉运行一个异步的 **"Gardening Job" (园艺进程)**，根据 $V$ 分数执行分级处理。

### 8.4.1 阈值定义 (Thresholds)

设定三个水位线：

| 水位线 | 生命力范围 | 状态 | 处理策略 |
| :--- | :--- | :--- | :--- |
| **High Watermark** | $V > 80$ | L2 活跃区 | 保留在向量索引中，随时可查 |
| **Low Watermark** | $20 < V \le 80$ | L2 边缘区 | 保留索引，但在检索时优先级降低（Rerank 降权） |
| **Archive Line** | $V \le 20$ | L3 归档区 | 触发归档流程 |

### 8.4.2 归档流程 (The Archiving Process)

当记忆 $V$ 值跌破 20 分时，`MemoryLifecycleEngine` 触发一次 GC 编排：

```
扫描候选记忆
        │
        ├── 获取所有活跃记忆
        ├── 由 MemoryLifecycleEngine 批量刷新生命力分数
        └── 将已刷新记忆传给 PeriodicGarbageCollector
                └── 筛选 V <= threshold 的记忆
        │
        ▼
批量归档 (限制 batch_size)
        │
        ├── 从 Qdrant 获取记忆
        ├── 序列化为 JSON
        ├── 可选 GZIP 压缩
        ├── 保存到文件系统 (按月份组织)
        ├── 更新归档索引
        └── 从 Qdrant 删除
```

**效果**：
- 普通的模糊语义检索（"找个关于日期的代码"）将不再返回这条记忆
- 数据库体积瘦身，查询速度保持高速

### 8.4.3 记忆唤醒 (Resurrection / Cache Miss Handling)

被归档的记忆并非永久死亡，它可以被"唤醒"（从 L3 搬回 L2）。

**场景**：用户突然问起一年前的一个极冷门的项目代号 "Project Titan"。

**流程**：
1. **L2 Miss**：向量检索无结果
2. **L3 Fallback**：Worker Agent 使用关键词（精确匹配）去冷存储里"打捞"
3. **Resurrection**：
   - 在归档索引中找到了 "Project Titan" 的旧记录
   - Agent 判定该信息当前有用
   - **动作**：帕秋莉重新计算 Embedding，将其重新插入 Qdrant
   - **重置**：$V$ 分数恢复到初始值，`last_accessed_at` 刷新

***

## 8.5 核心组件详解

### 8.5.1 VitalityCalculator（生命力计算器）

**职责**：实现生命力分数计算公式。

**核心方法**：

| 方法 | 说明 |
| :--- | :--- |
| `calculate(memory)` | 计算单个记忆的生命力分数（0-100） |

**配置项**（`VitalityCalculatorConfig`）：

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `code_snippet_weight` | `float` | `1.0` | 代码片段固有价值权重 |
| `fact_weight` | `float` | `0.9` | 事实固有价值权重 |
| `url_resource_weight` | `float` | `0.8` | URL 资源固有价值权重 |
| `reflection_weight` | `float` | `0.7` | 反思固有价值权重 |
| `user_profile_weight` | `float` | `0.6` | 用户配置固有价值权重 |
| `work_in_progress_weight` | `float` | `0.5` | 进行中工作固有价值权重 |
| `default_weight` | `float` | `0.5` | 未知类型默认权重 |
| `decay_lambda` | `float` | `0.01` | 时间衰减系数 λ |
| `points_per_access` | `float` | `1.0` | 每次访问加分 |
| `max_access_boost` | `float` | `20.0` | 访问加成上限 |

### 8.5.2 DynamicReinforcementEngine（动态强化引擎）

**职责**：处理记忆生命周期事件，动态调整生命力分数和置信度。

**核心方法**：

| 方法 | 说明 |
| :--- | :--- |
| `reinforce(memory_id, event)` | 处理强化事件，返回 `ReinforcementResult` |
| `get_event_history(memory_id, limit)` | 获取事件历史（最新的在前） |
| `clear_history()` | 清空事件历史（用于测试或维护） |
| `get_stats()` | 获取统计信息（总事件数、各类型事件计数） |

**配置项**（`ReinforcementEngineConfig`）：

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `enable_event_history` | `bool` | `True` | 是否记录事件历史 |
| `event_history_limit` | `int` | `10000` | 事件历史最大条数 |
| `hit_boost` | `float` | `5.0` | HIT 事件生命力加成 |
| `citation_boost` | `float` | `20.0` | CITATION 事件生命力加成 |
| `positive_feedback_boost` | `float` | `50.0` | 正面反馈生命力加成 |
| `negative_feedback_penalty` | `float` | `-50.0` | 负面反馈生命力惩罚 |
| `negative_confidence_multiplier` | `float` | `0.5` | 负面反馈置信度乘数 |

### 8.5.3 FileBasedArchiver（文件归档器）

**职责**：管理记忆在热存储（Qdrant）和冷存储（文件系统）之间的迁移。

**目录结构**：

```
data/archived/
├── archive_index.json      # 归档索引
└── 2025-01/                # 按月份组织
    ├── {uuid1}.json.gz
    └── {uuid2}.json.gz
```

**核心方法**：

| 方法 | 说明 |
| :--- | :--- |
| `archive(memory_id)` | 归档记忆到冷存储 |
| `resurrect(memory_id)` | 从冷存储唤醒记忆 |
| `is_archived(memory_id)` | 检查记忆是否已归档 |
| `get_archive_record(memory_id)` | 获取归档记录 |
| `list_archived(limit, vitality_threshold)` | 列出已归档的记忆 |

**配置项**（`ArchiverConfig`）：

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `archive_dir` | `str` | `"data/archived"` | 归档目录路径 |
| `compression` | `bool` | `True` | 是否使用 GZIP 压缩 |

### 8.5.4 PeriodicGarbageCollector（垃圾回收器）

**职责**：扫描调用方传入的、已刷新生命力的记忆并批量归档。它不持有 `VitalityCalculator`，也不负责持久化刷新后的生命力分数；这些编排职责属于 `MemoryLifecycleEngine`。

**核心方法**：

| 方法 | 说明 |
| :--- | :--- |
| `scan_candidates(memories, vitality_threshold)` | 从调用方传入的记忆中扫描低于阈值的候选 ID |
| `collect(memories, force, batch_size, vitality_threshold)` | 运行垃圾回收，返回归档数量 |
| `get_stats()` | 获取统计信息（最后运行时间、总扫描数、总归档数） |
| `reset_stats()` | 重置统计信息 |

**配置项**（`GarbageCollectorConfig`）：

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `low_watermark` | `float` | `20.0` | 低水位阈值（触发归档） |
| `batch_size` | `int` | `10` | 每次最多归档数量 |

### 8.5.5 MemoryLifecycleEngine（生命周期管理器）

**职责**：统一协调所有生命周期组件，提供统一操作接口。

**核心方法**：

| 方法 | 说明 |
| :--- | :--- |
| `refresh_vitality(memory, persist=False)` | 刷新单条记忆的生命力分数，可选择是否持久化 |
| `refresh_vitality_batch(memories, persist=False)` | 刷新调用方提供的一组记忆的生命力分数，可选择是否持久化 |
| `record_event(event)` | 记录生命周期事件 |
| `record_hit(memory_id, source)` | 记录检索命中事件（HIT） |
| `record_citation(memory_id, source)` | 记录主动引用事件（CITATION） |
| `record_feedback(memory_id, positive, source)` | 记录用户反馈事件 |
| `await run_garbage_collection(force)` | 获取全量活跃记忆，刷新生命力并异步运行垃圾回收 |
| `await archive_memory(memory_id)` | 异步手动归档指定记忆 |
| `await resurrect_memory(memory_id)` | 异步唤醒归档记忆 |
| `get_low_vitality_memories(threshold, limit)` | 获取低于阈值的记忆列表 |
| `get_event_history(memory_id, limit)` | 获取事件历史 |
| `get_archived_memories(limit, vitality_threshold)` | 获取已归档的记忆列表 |
| `get_stats()` | 获取统计信息 |

***

## 8.6 数据模型参考

### EventType（事件类型枚举）

```python
class EventType(str, Enum):
    HIT = "hit"                          # 被动检索命中
    CITATION = "citation"                # 主动引用
    FEEDBACK_POSITIVE = "feedback_positive"  # 用户正面反馈
    FEEDBACK_NEGATIVE = "feedback_negative"  # 用户负面反馈
```

### MemoryEvent（记忆事件）

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `event_type` | `EventType` | 事件类型 |
| `memory_id` | `UUID` | 目标记忆 ID |
| `timestamp` | `datetime` | 事件发生时间 |
| `source` | `str` | 事件来源（agent_id 或 "system"） |
| `metadata` | `Dict[str, Any]` | 事件额外信息 |

### ReinforcementResult（强化结果）

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `memory_id` | `UUID` | 目标记忆 ID |
| `previous_vitality` | `float` | 强化前的生命力分数（0-100） |
| `new_vitality` | `float` | 强化后的生命力分数（0-100） |
| `previous_confidence` | `float` | 强化前的置信度（0-1） |
| `new_confidence` | `float` | 强化后的置信度（0-1） |
| `event_type` | `EventType` | 触发的事件类型 |
| `timestamp` | `datetime` | 事件时间戳 |

### ArchiveStatus（归档状态枚举）

```python
class ArchiveStatus(str, Enum):
    ACTIVE = "active"                    # 在热存储中（Qdrant）
    ARCHIVED = "archived"                # 已归档到冷存储
    PENDING_ARCHIVE = "pending_archive"  # 标记待归档
    PENDING_RESURRECT = "pending_resurrect"  # 标记待唤醒
```

### ArchiveRecord（归档记录）

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `memory_id` | `UUID` | 记忆 ID |
| `original_vitality` | `float` | 归档时的生命力分数（0-100） |
| `archived_at` | `datetime` | 归档时间 |
| `storage_path` | `str` | 存储路径（文件路径或 S3 key） |
| `compressed_size_bytes` | `Optional[int]` | 压缩后的大小（字节） |

***

## 8.7 配置参考

生命周期管理相关配置统一在 `MemoryLifecycleConfig` 中：

```python
class MemoryLifecycleConfig(BaseModel):
    vitality_calculator: VitalityCalculatorConfig
    reinforcement: ReinforcementEngineConfig
    archiver: ArchiverConfig
    garbage_collector: GarbageCollectorConfig
    high_watermark: float = 80.0  # 高水位阈值
```

详细配置项见 8.5 节各组件的配置表。
