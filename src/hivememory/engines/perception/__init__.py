"""
HiveMemory - 帕秋莉感知层 / MMU (Perception Layer / Memory Management Unit)

职责:
    提供无状态的短期记忆摄入能力（block 构造、事件归并、token 估算），
    以及 Relay / Page Folding 摘要生成器。

    Topic 状态、活跃池、settle/evict 生命周期由
    ``hivememory.patchouli.services.topic_buffer.TopicBufferService`` 唯一拥有；
    本包不再承载任何 Topic 领域状态。

核心组件:
    - BasePerceptionLayer: 感知摄入接口（无状态）
    - SemanticFlowPerceptionLayer: 语义流感知层 / MMU（摄入编排与 retry 幂等）
    - ShortTermMemoryStore: 短期话题 CRUD 与快照边界
    - TriggerReason / FlushEvent / TopicMaterializeTask: 触发与交接协议模型
    - BaseRelayController: Token 溢出接力控制器 / Page Folding 摘要生成器

Note: Topic 状态领域服务 ``TopicBufferService`` 位于
      ``hivememory.patchouli.services.topic_buffer``；本包不导出 Patchouli
      服务，保持 engines -> patchouli 的依赖方向唯一。

参考: ShortTermMemory.md, PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 6.0.0
"""
from hivememory.system.config import (
    MemoryPerceptionConfig,
    SemanticFlowPerceptionConfig,
)
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)

from hivememory.engines.perception.interfaces import (
    BasePerceptionLayer,
)
from hivememory.engines.perception.models import (
    TraceItem,
    LogicalBlock,
    FlushEvent,
    TriggerReason,
    TopicMaterializeTask,
)
from hivememory.core.models import BufferState
from hivememory.engines.perception.relay_controller import (
    BaseRelayController,
    NoOpRelayController,
    SimpleRelayController,
    LLMRelayController,
    create_relay_controller,
)
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
    NullPerceptionLayer,
)

import logging

logger = logging.getLogger(__name__)


def create_perception_layer(
    config: MemoryPerceptionConfig,
    llm_service=None,
    *,
    short_term_store,
    interaction_journal: InMemoryInteractionApplyJournal,
) -> BasePerceptionLayer:
    """
    创建感知层 (MMU) 实例

    Args:
        config: 感知层配置 (MemoryPerceptionConfig)
        llm_service: LLM 服务（用于 LLMRelayController 摘要生成）
        short_term_store: ShortTermMemoryStore 实例，必须由 PatchouliRuntime 从 MemoryLibrary 注入；
            工厂会用它装配 TopicBufferService（Topic 状态唯一所有者）
        interaction_journal: interaction apply 的进程内幂等 journal

    Returns:
        SemanticFlowPerceptionLayer 实例
    """
    impl_config = config.engine

    if not isinstance(impl_config, SemanticFlowPerceptionConfig):
        logger.warning(
            f"配置类型 {type(impl_config).__name__} 不是 SemanticFlowPerceptionConfig，"
            f"将使用默认 SemanticFlowPerceptionConfig"
        )
        impl_config = SemanticFlowPerceptionConfig()

    if not impl_config.enable:
        return NullPerceptionLayer()

    # 延迟导入：避免 engines 包在模块加载期反向依赖 patchouli.services。
    from hivememory.patchouli.services.topic_buffer import TopicBufferService

    relay_config = getattr(config, "relay", None) or impl_config.relay
    relay_controller = create_relay_controller(
        config=relay_config,
        llm_service=llm_service
    )

    topic_buffer = TopicBufferService(
        store=short_term_store,
        relay_controller=relay_controller,
    )

    perception = SemanticFlowPerceptionLayer(
        config=impl_config,
        relay_controller=relay_controller,
        topic_buffer=topic_buffer,
        interaction_journal=interaction_journal,
    )

    return perception


__all__ = [
    # 感知层实现
    "SemanticFlowPerceptionLayer",
    "NullPerceptionLayer",
    # 接口
    "BasePerceptionLayer",
    # 数据模型
    "TraceItem",
    "LogicalBlock",
    "BufferState",
    "FlushEvent",
    "TriggerReason",
    "TopicMaterializeTask",
    # 接力控制器 / Page Folding 摘要生成器
    "BaseRelayController",
    "NoOpRelayController",
    "SimpleRelayController",
    "LLMRelayController",
    "create_relay_controller",
    # 工厂函数
    "create_perception_layer",
]
