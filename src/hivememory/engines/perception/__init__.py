"""
HiveMemory - 帕秋莉感知层 / MMU (Perception Layer / Memory Management Unit)

职责:
    作为短期记忆的 MMU（内存管理单元），管理多话题的生命周期。
    负责话题路由(route)、换入(swap-in)、换出(swap-out)和 LRU 驱逐。

核心组件:
    - BasePerceptionLayer: 感知层基类
    - SemanticFlowPerceptionLayer: 语义流感知层 / MMU（多话题并发管理）
    - SemanticBufferManager: 话题管理器 (TopicManager) - 纯状态管理
    - TriggerManager: 话题结算调度器 - Flush 触发逻辑
    - SemanticBuffer: 话题段 (TopicSegment)
    - LogicalBlock: 页 (Page)
    - BaseRelayController: Token 溢出接力控制器 / Page Folding 摘要生成器

定时调度:
    空闲超时扫描由 SystemAsyncScheduler 统一管理，
    感知层只暴露 scan_idle_buffers_once() 供调度器调用。

参考: ShortTermMemory.md, PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 5.0.0 (Phase S2 — 统一调度器接入)
"""
from hivememory.system.config import (
    MemoryPerceptionConfig,
    SemanticFlowPerceptionConfig,
)

from hivememory.engines.perception.interfaces import (
    BasePerceptionLayer,
)
from hivememory.engines.perception.models import (
    TraceItem,
    LogicalBlock,
    BufferState,
    SemanticBuffer,
    FlushEvent,
    FlushReason,
)
from hivememory.engines.perception.buffer_manager import SemanticBufferManager
from hivememory.engines.perception.trigger_manager import (
    TriggerManager,
    DECISION_MATRIX,
)
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

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_perception_layer(
    config: MemoryPerceptionConfig,
    llm_service=None,
    short_term_store=None,
) -> BasePerceptionLayer:
    """
    创建感知层 (MMU) 实例

    Args:
        config: 感知层配置 (MemoryPerceptionConfig)
        llm_service: LLM 服务（用于 LLMRelayController 摘要生成）
        short_term_store: ShortTermMemoryStore 实例（由 MemoryLibrary 创建后注入）。
            为 None 时自动创建（向后兼容，仅用于测试场景）。

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

    relay_config = getattr(config, "relay", None) or impl_config.relay
    relay_controller = create_relay_controller(
        config=relay_config,
        llm_service=llm_service
    )

    # 若未注入 store，自动创建（测试 / 向后兼容路径）
    if short_term_store is None:
        from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
        short_term_store = ShortTermMemoryStore(
            max_resident_topics=impl_config.max_resident_topics
        )

    perception = SemanticFlowPerceptionLayer(
        config=impl_config,
        relay_controller=relay_controller,
        short_term_store=short_term_store,
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
    "SemanticBuffer",
    "FlushEvent",
    "FlushReason",
    # 缓冲区管理器 — DEPRECATED: 请改用 patchouli.memory_library.ShortTermMemoryStore
    "SemanticBufferManager",
    # 话题结算调度器
    "TriggerManager",
    "DECISION_MATRIX",
    # 接力控制器 / Page Folding 摘要生成器
    "BaseRelayController",
    "NoOpRelayController",
    "SimpleRelayController",
    "LLMRelayController",
    "create_relay_controller",
    # 工厂函数
    "create_perception_layer",
    # Artifact 构建
    "InteractionArtifactBuilder",
]
