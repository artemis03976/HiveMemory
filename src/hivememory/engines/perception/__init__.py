"""
HiveMemory - 帕秋莉感知引擎 (Perception Engine / MMU 算法层)

职责:
    提供无状态的短期记忆摄入纯算法（block 构造、事件归并、token 估算、
    折叠阈值判断），以及 Relay / Page Folding 摘要生成器与触发交接协议模型。

    话题路由、占用（lease）、settle/evict 生命周期与 LRU / idle / shutdown
    维护由 ``PerceptionFamiliar``（Patchouli 服务层）编排；本包不持有
    Store / Journal / Queue，也不导入 ``hivememory.patchouli.*``，保持
    patchouli -> engines 的单向依赖，可被其他 runtime 复用。

核心组件:
    - MemoryPerceptionEngine: 无状态摄入算法引擎（纯函数）
    - TriggerReason / FlushEvent / TopicMaterializeTask: 触发与交接协议模型
    - BaseRelayController: Token 溢出接力控制器 / Page Folding 摘要生成器

参考: ShortTermMemory.md, PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 7.0.0
"""

from hivememory.engines.perception.memory_perception_engine import (
    MemoryPerceptionEngine,
)
from hivememory.engines.perception.models import (
    FlushEvent,
    LogicalBlock,
    TopicMaterializeTask,
    TraceItem,
    TriggerReason,
)
from hivememory.engines.perception.relay_controller import (
    BaseRelayController,
    LLMRelayController,
    NoOpRelayController,
    SimpleRelayController,
    create_relay_controller,
)

__all__ = [
    # 感知引擎（纯算法）
    "MemoryPerceptionEngine",
    # 数据模型
    "TraceItem",
    "LogicalBlock",
    "FlushEvent",
    "TriggerReason",
    "TopicMaterializeTask",
    # 接力控制器 / Page Folding 摘要生成器
    "BaseRelayController",
    "NoOpRelayController",
    "SimpleRelayController",
    "LLMRelayController",
    "create_relay_controller",
]
