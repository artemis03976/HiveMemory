# MemoryLifeCycleManagement - 记忆生命周期管理模块

## 📖 概述

MemoryLifeCycleManagement 模块负责记忆的动态演化、垃圾回收和冷热数据管理。

对应设计文档: **PROJECT.md 第 6 章**

---

## ⚠️ 当前状态

**🚧 骨架接口 - 待 Stage 3 实现**

本模块目前仅包含接口定义，核心功能将在 Stage 3 开发中完成。

---

## 🎯 核心职责 (计划)

1. **访问统计** - 记录 Hit Counter
2. **生命力分数计算** - Vitality Score 公式实现
3. **动态强化** - Hit/Citation 事件驱动加分
4. **时间衰减** - 指数衰减函数
5. **垃圾回收** - 低价值记忆归档
6. **冷热分离** - L1 (Context) → L2 (Qdrant) → L3 (Cold Storage)

---

## 📦 预定义接口

### `interfaces.py`

```python
from abc import ABC, abstractmethod

class VitalityCalculator(ABC):
    """生命力分数计算器"""
    @abstractmethod
    def calculate(self, memory: MemoryAtom) -> float:
        """
        计算公式:
        V = (Confidence × Intrinsic_Value) × Decay(time) + Access_Boost
        """
        pass

class ReinforcementEngine(ABC):
    """动态强化引擎"""
    @abstractmethod
    def reinforce(self, memory_id: UUID, event: Event) -> None:
        """处理 Hit/Citation 事件"""
        pass

class MemoryArchiver(ABC):
    """冷存储管理器"""
    @abstractmethod
    def archive(self, memory_id: UUID) -> None:
        """归档到冷存储 (PostgreSQL/S3)"""
        pass

    @abstractmethod
    def resurrect(self, memory_id: UUID) -> MemoryAtom:
        """从冷存储唤醒"""
        pass
```

---

## 🛣️ 开发计划

**Stage 3 任务清单**:
- [ ] 实现 VitalityCalculator (生命力分数公式)
- [ ] 实现 ReinforcementEngine (事件驱动强化)
- [ ] 实现 DecayFunction (时间衰减)
- [ ] 实现 GarbageCollector (后台 GC 任务)
- [ ] 实现 MemoryArchiver (冷存储机制)
- [ ] 集成 PostgreSQL/SQLite 作为冷存储

---

## 📊 生命力分数模型 (设计)

```python
# 分数公式
V = (C × I) × D(t) + A

# 参数说明:
# C = Confidence Score (置信度, 0.0-1.0)
# I = Intrinsic Value (固有价值, 类型相关)
#     CODE_SNIPPET: 1.2
#     FACT: 1.0
#     REFLECTION: 0.9
# D(t) = Decay Function (时间衰减)
#     D(t) = exp(-λ × days)
#     λ = 0.01 (衰减系数)
# A = Access Boost (访问加成)
#     A = access_count × 5

# 三级阈值:
# V > 80: L2 Active Memory (Qdrant Hot Storage)
# 20 < V < 80: L2 可能被 GC
# V < 20: L3 Cold Storage (PostgreSQL/S3)
```

---

## 📚 相关文档

- [PROJECT.md 第 6 章](../../docs/PROJECT.md) - 完整设计文档
- [ROADMAP.md Stage 3](../../docs/ROADMAP.md) - 开发路线图

---

**维护者**: HiveMemory Team
**最后更新**: 2025-12-23
**版本**: 0.1.0 (骨架)
