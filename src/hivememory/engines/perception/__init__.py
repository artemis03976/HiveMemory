"""
HiveMemory - 帕秋莉感知层 / MMU (Perception Layer / Memory Management Unit)

职责:
    作为短期记忆的 MMU（内存管理单元），管理多话题的生命周期。
    负责话题路由(route)、换入(swap-in)、换出(swap-out)和 LRU 驱逐。

核心组件:
    - BasePerceptionLayer: 感知层基类，提供空闲超时监控功能
    - SemanticFlowPerceptionLayer: 语义流感知层 / MMU（多话题并发管理）
    - SemanticBufferManager: 话题管理器 (TopicManager)
    - SemanticBuffer: 话题段 (TopicSegment)
    - LogicalBlock: 页 (Page)

空闲超时监控:
    所有感知层实现都继承了基类的空闲超时监控功能：
    - start_idle_monitor(): 启动监控
    - stop_idle_monitor(): 停止监控
    - scan_idle_buffers_now(): 立即扫描一次

Note:
    Phase 4.5 重构：
    - 移除 Adsorber / Relay / TriggerManager / SimplePerceptionLayer 依赖
    - 上述组件源码保留，但不再从 __init__.py 导出或在主流程中使用
    - create_perception_layer() 简化为仅创建 SemanticFlowPerceptionLayer

参考: ShortTermMemory.md, PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 4.5.0
"""
from hivememory.patchouli.config import (
    MemoryPerceptionConfig,
    SemanticFlowPerceptionConfig,
)

from hivememory.engines.perception.interfaces import (
    BasePerceptionLayer,
    BaseArbiter,
)
from hivememory.engines.perception.models import (
    TraceItem,
    InteractionPayload,
    Triplet,
    LogicalBlock,
    BufferState,
    SemanticBuffer,
    FlushEvent,
    FlushReason,
)
from hivememory.engines.perception.buffer_manager import SemanticBufferManager
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_perception_layer(
    config: MemoryPerceptionConfig,
    embedding_service=None,
    reranker_service=None,
    on_flush_callback=None,
) -> SemanticFlowPerceptionLayer:
    """
    创建感知层 (MMU) 实例

    Phase 4.5 简化版：仅创建 SemanticFlowPerceptionLayer。
    embedding_service / reranker_service 参数保留向后兼容签名但不再使用。

    Args:
        config: 感知层配置 (MemoryPerceptionConfig)
        embedding_service: (已弃用) 保留参数兼容性
        reranker_service: (已弃用) 保留参数兼容性
        on_flush_callback: Flush 回调函数

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

    perception = SemanticFlowPerceptionLayer(
        config=impl_config,
        on_flush_callback=on_flush_callback,
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
    "BaseArbiter",
    "BasePerceptionLayer",
    # 数据模型
    "TraceItem",
    "InteractionPayload",
    "Triplet",
    "LogicalBlock",
    "BufferState",
    "SemanticBuffer",
    "FlushEvent",
    "FlushReason",
    # 缓冲区管理器
    "SemanticBufferManager",
    # 工厂函数
    "create_perception_layer",
]
