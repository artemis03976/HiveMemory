"""
HiveMemory - 帕秋莉感知层 / MMU (Perception Layer / Memory Management Unit)

职责:
    作为短期记忆的 MMU（内存管理单元），管理多话题的生命周期。
    负责话题路由(route)、换入(swap-in)、换出(swap-out)和 LRU 驱逐。

核心组件:
    - BasePerceptionLayer: 感知层基类，提供空闲超时监控功能
    - SemanticFlowPerceptionLayer: 语义流感知层 / MMU（多话题并发管理）
    - SemanticBufferManager: 话题管理器 (TopicManager) - 纯状态管理
    - TriggerManager: 话题结算调度器 - Flush 触发逻辑
    - SemanticBuffer: 话题段 (TopicSegment)
    - LogicalBlock: 页 (Page)
    - BaseRelayController: Token 溢出接力控制器 / Page Folding 摘要生成器

空闲超时监控:
    所有感知层实现都继承了基类的空闲超时监控功能：
    - start_idle_monitor(): 启动监控
    - stop_idle_monitor(): 停止监控
    - scan_idle_buffers_now(): 立即扫描一次

.. deprecated::
    Phase 4.5 重构后，以下组件已废弃：
    - BaseArbiter: 灰度仲裁器接口，已不再被感知层使用
    - SemanticBoundaryAdsorber: 语义吸附器，已不再被感知层使用
    这些组件保留作为参考实现，将在后续版本中移除。

参考: ShortTermMemory.md, PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 4.5.1
"""
from hivememory.patchouli.config import (
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
    SimpleRelayController,
    LLMRelayController,
    create_relay_controller,
)
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_perception_layer(
    config: MemoryPerceptionConfig,
    llm_service=None,
) -> SemanticFlowPerceptionLayer:
    """
    创建感知层 (MMU) 实例

    Phase 4.5 简化版：仅创建 SemanticFlowPerceptionLayer。
    embedding_service / reranker_service 参数保留向后兼容签名但不再使用。

    Args:
        config: 感知层配置 (MemoryPerceptionConfig)
        llm_service: LLM 服务（用于 LLMRelayController 摘要生成）

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

    relay_config = getattr(config, "relay", None) or impl_config.relay
    relay_controller = create_relay_controller(
        config=relay_config,
        llm_service=llm_service
    )

    perception = SemanticFlowPerceptionLayer(
        config=impl_config,
        relay_controller=relay_controller,
    )

    # 启动空闲超时监控
    perception.start_idle_monitor(
        idle_timeout_seconds=impl_config.idle_timeout_seconds,
        scan_interval_seconds=impl_config.scan_interval_seconds,
    )

    return perception


__all__ = [
    # 感知层实现
    "SemanticFlowPerceptionLayer",
    # 接口
    "BasePerceptionLayer",
    # 数据模型
    "TraceItem",
    "LogicalBlock",
    "BufferState",
    "SemanticBuffer",
    "FlushEvent",
    "FlushReason",
    # 缓冲区管理器
    "SemanticBufferManager",
    # 话题结算调度器
    "TriggerManager",
    "DECISION_MATRIX",
    # 接力控制器 / Page Folding 摘要生成器
    "BaseRelayController",
    "SimpleRelayController",
    "LLMRelayController",
    "create_relay_controller",
    # 工厂函数
    "create_perception_layer",
]
