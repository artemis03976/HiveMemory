# MemoryLifeCycleManagement - 记忆生命周期管理模块

## 📖 概述

MemoryLifeCycleManagement 模块负责记忆的动态演化、垃圾回收和冷热数据管理。通过生命力分数（Vitality Score）机制，系统能够自动识别高价值记忆并维持其活跃状态，同时将低价值记忆自动归档，确保系统的高效运行。

对应设计文档: **PROJECT.md 第 6 章**

---

## ✅ 当前状态

**🎉 Stage 3 实现完成**

本模块已完成核心功能开发，包括：
- **生命力计算体系**: 基于置信度、固有价值、时间衰减和访问加成的综合评分模型
- **动态强化引擎**: 支持 HIT、CITATION、FEEDBACK 等事件驱动的分数调整
- **垃圾回收机制**: 支持周期性和定时触发的低价值记忆清理
- **冷热分级存储**: 实现基于文件系统的冷存储归档与唤醒机制
- **统一生命周期管理**: 提供 `MemoryLifecycleEngine` 协调各组件工作

---

## 🎯 核心组件

### 1. `vitality.py` - 生命力计算器

**职责**: 计算记忆的生命力分数，决定记忆的存留。

**核心类**:
- `VitalityCalculator`: 标准计算器，实现 $V = (C \times I) \times D(t) + A$ 公式

**评分模型**:
- **固有价值 (I)**: 代码片段 (1.0) > 事实 (0.9) > URL资源 (0.8) > 反思 (0.7)
- **时间衰减 D(t)**: $e^{-\lambda \times days}$，随时间指数衰减
- **访问加成 (A)**: 每次访问 +2.0，封顶 20.0

### 2. `reinforcement.py` - 动态强化引擎

**职责**: 处理记忆交互事件，动态调整生命力。

**核心类**:
- `DynamicReinforcementEngine`: 处理事件并更新分数

**支持事件**:
- `HIT` (检索命中): +5 生命力
- `CITATION` (主动引用): +20 生命力，并重置时间衰减
- `FEEDBACK_POSITIVE` (正面反馈): +50 生命力
- `FEEDBACK_NEGATIVE` (负面反馈): -50 生命力，置信度减半

### 3. `garbage_collector.py` - 垃圾回收器

**职责**: 扫描低生命力记忆并触发归档。

**核心类**:
- `PeriodicGarbageCollector`: 基础周期性 GC
- `ScheduledGarbageCollector`: 基于 APScheduler 的定时 GC (默认 24h)

**策略**:
- **低水位线**: 生命力 < 20.0 的记忆将被标记为归档候选
- **批量处理**: 每次 GC 限制处理数量，避免阻塞

### 4. `archiver.py` - 冷存储归档器

**职责**: 管理记忆在热存储（Qdrant）和冷存储（文件系统）间的迁移。

**核心类**:
- `FileBasedMemoryArchiver`: 本地文件系统归档 (JSON + GZIP)
- `S3MemoryArchiver`: (TODO) S3 对象存储归档

**目录结构**:
```text
data/archived/
├── archive_index.json      # 归档索引
└── 2025-01/                # 按月份组织
    ├── {uuid}.json.gz
```

### 5. `engine.py` - 生命周期引擎

**职责**: 统一门面，协调所有组件。

**核心类**:
- `MemoryLifecycleEngine`: 提供 `record_event`, `run_garbage_collection` 等统一接口

---

## 🚀 快速使用

### 初始化引擎

```python
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.lifecycle import create_default_lifecycle_engine

# 1. 初始化存储
storage = QdrantMemoryStore()

# 2. 创建生命周期引擎
# 启用定时 GC (每 24 小时运行一次)
lifecycle_engine = create_default_lifecycle_engine(
    storage=storage,
    enable_scheduled_gc=True,
    gc_interval_hours=24
)
```

### 记录事件

```python
from hivememory.lifecycle.types import EventType

# 场景 1: 检索命中 (被动)
lifecycle_engine.record_hit(memory_id="uuid...", source="system")

# 场景 2: 记忆引用 (主动) -> 将重置时间衰减
lifecycle_engine.record_citation(memory_id="uuid...", source="agent_worker")

# 场景 3: 用户反馈
lifecycle_engine.record_feedback(
    memory_id="uuid...", 
    positive=True, 
    source="user"
)
```

### 手动触发垃圾回收

```python
# 强制运行 GC，归档生命力 < 20 的记忆
archived_count = lifecycle_engine.run_garbage_collection(force=True)
print(f"Archived {archived_count} memories")
```

### 记忆唤醒

```python
# 当检索不到时，尝试从冷存储唤醒
try:
    memory = lifecycle_engine.resurrect_memory(memory_id="uuid...")
    print("Memory resurrected from archive")
except ValueError:
    print("Memory not found in archive")
```

---

## 🔧 配置参数

可以通过 `config.yaml` 或初始化参数进行配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `decay_lambda` | 0.01 | 时间衰减系数 |
| `low_watermark` | 20.0 | GC 触发阈值 (生命力 < 20) |
| `gc_batch_size` | 10 | 单次 GC 最大处理数量 |
| `archive_dir` | `data/archived` | 冷存储路径 |
| `compress` | `True` | 是否启用 GZIP 压缩 |

---

## 🛣️ 未来路线图

- [x] 实现 VitalityCalculator (生命力分数公式)
- [x] 实现 ReinforcementEngine (事件驱动强化)
- [x] 实现 GarbageCollector (后台 GC 任务)
- [x] 实现 MemoryArchiver (文件系统冷存储)
- [ ] 实现 S3MemoryArchiver (S3 云存储支持)
- [ ] 增加生命力可视化面板
- [ ] 支持自定义衰减策略插件

---

## 📚 相关文档

- [PROJECT.md 第 6 章](../../docs/PROJECT.md) - 完整设计文档
- [ROADMAP.md Stage 3](../../docs/ROADMAP.md) - 开发路线图

---

**维护者**: HiveMemory Team
**最后更新**: 2025-12-28
**版本**: 0.2.0 (Stage 3 完成)
