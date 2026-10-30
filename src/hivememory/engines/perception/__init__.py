"""
HiveMemory - 帕秋莉感知引擎 (Perception Engine / MMU 算法层)

职责:
    提供无状态的短期记忆摄入纯算法（block 构造、token 估算、折叠阈值判断），
    一轮交互材料的解析与渲染（记忆生成视图、接力摘要与 token 统计共用），
    以及 Relay / Page Folding 摘要生成器与触发交接协议模型。

    话题路由、占用（lease）、settle/evict 生命周期与 LRU / idle / shutdown
    维护由 ``PerceptionFamiliar``（Patchouli 服务层）编排；本包不持有
    Store / Journal / Queue，也不导入 ``hivememory.patchouli.*``，保持
    patchouli -> engines 的单向依赖，可被其他 runtime 复用。

核心组件:
    - MemoryPerceptionEngine: 无状态摄入算法引擎（纯函数）
    - TriggerReason / TopicMaterializeTask: 触发与交接协议模型
    - BaseRelayController: Token 溢出接力控制器 / Page Folding 摘要生成器
"""

from hivememory.engines.perception.memory_perception_engine import (
    MemoryPerceptionEngine,
)
from hivememory.engines.perception.models import (
    LogicalBlock,
    TopicMaterializeTask,
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
    "LogicalBlock",
    "TriggerReason",
    "TopicMaterializeTask",
    # 接力控制器 / Page Folding 摘要生成器
    "BaseRelayController",
    "NoOpRelayController",
    "SimpleRelayController",
    "LLMRelayController",
    "create_relay_controller",
]
