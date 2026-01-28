"""
HiveMemory - 帕秋莉感知层 (Perception Layer)

职责:
    作为系统的流量入口与预处理器，负责将非结构化的原始消息流（Raw Stream）
    转化为语义连贯的逻辑原子块（Logical Blocks），并决定何时唤醒 Librarian Agent。

核心组件:
    - StreamParser: 流式解析器，抹平不同 Agent 框架的消息格式差异
    - SemanticAdsorber: 语义吸附器，基于 Embedding 实时计算语义相似度
    - RelayController: 接力控制器，处理 Token 溢出并生成中间态摘要
    - BasePerceptionLayer: 感知层基类，提供空闲超时监控功能
    - SimplePerceptionLayer: 简单感知层（三重触发机制）
    - SemanticFlowPerceptionLayer: 语义流感知层（统一语义流架构）

空闲超时监控:
    所有感知层实现都继承了基类的空闲超时监控功能：
    - start_idle_monitor(): 启动监控
    - stop_idle_monitor(): 停止监控
    - scan_idle_buffers_now(): 立即扫描一次

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 2.0.0
"""
from hivememory.patchouli.config import (
    MemoryPerceptionConfig,
    SemanticFlowPerceptionConfig,
    SimplePerceptionConfig,
)

from hivememory.engines.perception.interfaces import (
    TriggerStrategy,
    BasePerceptionLayer,
    BaseArbiter,
)
from hivememory.engines.perception.models import (
    Triplet,
    LogicalBlock,
    BufferState,
    SemanticBuffer,
    SimpleBuffer,
    FlushEvent,
    FlushReason,
)
from hivememory.engines.perception.trigger_strategies import (
    TriggerManager,
    MessageCountTrigger,
    SemanticBoundaryTrigger,
)
from hivememory.engines.perception.stream_parser import UnifiedStreamParser
from hivememory.engines.perception.semantic_adsorber import SemanticBoundaryAdsorber, create_adsorber
from hivememory.engines.perception.relay_controller import RelayController
from hivememory.engines.perception.block_builder import LogicalBlockBuilder
from hivememory.engines.perception.buffer_manager import (
    SemanticBufferManager,
    SimpleBufferManager,
)
from hivememory.engines.perception.context_bridge import ContextBridge
from hivememory.engines.perception.grey_area_arbiter import (
    RerankerArbiter,
    SLMArbiter,
    NoOpArbiter,
)
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.engines.perception.simple_perception_layer import SimplePerceptionLayer

from hivememory.infrastructure.embedding import BaseEmbeddingService
from hivememory.infrastructure.rerank import BaseRerankService

from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def create_perception_layer(
    config: MemoryPerceptionConfig,
    embedding_service: BaseEmbeddingService,
    reranker_service: Optional[BaseRerankService] = None,
    on_flush_callback=None,
) -> Union[SemanticFlowPerceptionLayer, SimplePerceptionLayer]:
    """
    创建默认配置的感知层实例

    根据 config.layer_type 自动选择：
    - "semantic_flow": SemanticFlowPerceptionLayer（默认）
    - "simple": SimplePerceptionLayer

    Args:
        config: 感知层配置（MemoryPerceptionConfig，可选，使用默认配置）
        embedding_service: Embedding 服务实例（必须传入）
        on_flush_callback: Flush 回调函数

    Returns:
        SemanticFlowPerceptionLayer 或 SimplePerceptionLayer 实例

    Examples:
        >>> from hivememory.perception import create_perception_layer
        >>> def on_flush(blocks, reason):
        ...     print(f"Flush: {reason}, Blocks: {len(blocks)}")
        >>> perception = create_perception_layer(config=config, embedding_service=svc, on_flush_callback=on_flush)
    """

    impl_config = config.engine

    if isinstance(impl_config, SimplePerceptionConfig):
        # 创建可配置的 TriggerManager
        strategies = [
            MessageCountTrigger(threshold=impl_config.message_count_threshold),
        ]
        if impl_config.enable_semantic_trigger:
            strategies.append(SemanticBoundaryTrigger())

        trigger_manager = TriggerManager(strategies=strategies)

        perception = SimplePerceptionLayer(
            config=impl_config,
            trigger_manager=trigger_manager,
            on_flush_callback=on_flush_callback,
        )

        # 启动空闲超时监控
        perception.start_idle_monitor(
            idle_timeout_seconds=impl_config.idle_timeout_seconds,
            scan_interval_seconds=impl_config.scan_interval_seconds,
        )

        return perception

    elif isinstance(impl_config, SemanticFlowPerceptionConfig):
        parser = UnifiedStreamParser()

        adsorber = create_adsorber(
            config=impl_config.adsorber,
            embedding_service=embedding_service,
            reranker_service=reranker_service,
        )
        relay_controller = RelayController(
            max_processing_tokens=impl_config.max_processing_tokens,
            enable_smart_summary=impl_config.enable_smart_summary,
        )

        perception = SemanticFlowPerceptionLayer(
            config=impl_config,
            parser=parser,
            adsorber=adsorber,
            relay_controller=relay_controller,
            on_flush_callback=on_flush_callback,
        )
        
        # 启动空闲超时监控
        perception.start_idle_monitor(
            idle_timeout_seconds=impl_config.idle_timeout_seconds,
            scan_interval_seconds=impl_config.scan_interval_seconds,
        )
        
        return perception
    
    else:
        logger.error(f"未知的 layer 类型: {impl_config.layer_type}")
        return None



__all__ = [
    # 感知层实现
    "SimplePerceptionLayer",
    "SemanticFlowPerceptionLayer",
    # 接口
    "TriggerStrategy",
    "BaseArbiter",
    "BasePerceptionLayer",
    # 数据模型
    "Triplet",
    "LogicalBlock",
    "BufferState",
    "SemanticBuffer",
    "SimpleBuffer",
    "FlushEvent",
    "FlushReason",
    # 触发策略
    "TriggerManager",
    "MessageCountTrigger",
    "SemanticBoundaryTrigger",
    # 缓冲区管理器
    "LogicalBlockBuilder",
    "SemanticBufferManager",
    "SimpleBufferManager",
    # 统一消息流分析
    "UnifiedStreamParser",
    # 语义边界判定
    "SemanticBoundaryAdsorber",
    # 溢出接力控制
    "RelayController",
    # 灰度区仲裁器
    "RerankerArbiter",
    "SLMArbiter",
    "NoOpArbiter",
    "ContextBridge",
    "create_perception_layer",
]
