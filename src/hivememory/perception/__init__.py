"""
HiveMemory - 帕秋莉感知层 (Perception Layer)

职责:
    作为系统的流量入口与预处理器，负责将非结构化的原始消息流（Raw Stream）
    转化为语义连贯的逻辑原子块（Logical Blocks），并决定何时唤醒 Librarian Agent。

核心组件:
    - StreamParser: 流式解析器，抹平不同 Agent 框架的消息格式差异
    - SemanticAdsorber: 语义吸附器，基于 Embedding 实时计算语义相似度
    - RelayController: 接力控制器，处理 Token 溢出并生成中间态摘要
    - IdleTimeoutMonitor: 空闲超时监控器，异步监控 Buffer 空闲状态
    - SimplePerceptionLayer: 简单感知层（三重触发机制）
    - SemanticFlowPerceptionLayer: 语义流感知层（统一语义流架构）

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 2.0.0
"""

from hivememory.core.models import FlushReason
from hivememory.perception.models import (
    StreamMessageType,
    StreamMessage,
    Triplet,
    LogicalBlock,
    BufferState,
    SemanticBuffer,
    SimpleBuffer,
)
from hivememory.perception.interfaces import (
    StreamParser,
    SemanticAdsorber,
    RelayController,
    IdleTimeoutMonitor as IdleTimeoutMonitorInterface,
    BasePerceptionLayer,
)
from hivememory.core.embedding import (
    LocalEmbeddingService,
    get_embedding_service,
)
from hivememory.perception.stream_parser import UnifiedStreamParser
from hivememory.perception.semantic_adsorber import SemanticBoundaryAdsorber
from hivememory.perception.relay_controller import TokenOverflowRelayController
from hivememory.perception.idle_timeout_monitor import IdleTimeoutMonitor
from hivememory.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.perception.simple_perception_layer import (
    SimplePerceptionLayer,
)
from hivememory.perception.trigger_strategies import (
    TriggerManager,
    create_default_trigger_manager,
    MessageCountTrigger,
    IdleTimeoutTrigger,
    SemanticBoundaryTrigger,
)
from hivememory.core.config import MemoryPerceptionConfig
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def create_default_perception_layer(
    config: Optional[MemoryPerceptionConfig] = None,
    on_flush_callback=None,
) -> Union[SemanticFlowPerceptionLayer, SimplePerceptionLayer]:
    """
    创建默认配置的感知层实例

    根据 config.layer_type 自动选择：
    - "semantic_flow": SemanticFlowPerceptionLayer（默认）
    - "simple": SimplePerceptionLayer

    Args:
        on_flush_callback: Flush 回调函数
        config: 感知层配置（MemoryPerceptionConfig，可选，使用默认配置）

    Returns:
        SemanticFlowPerceptionLayer 或 SimplePerceptionLayer 实例

    Examples:
        >>> from hivememory.perception import create_default_perception_layer
        >>> def on_flush(blocks, reason):
        ...     print(f"Flush: {reason}, Blocks: {len(blocks)}")
        >>> perception = create_default_perception_layer(on_flush_callback=on_flush)
    """
    if config is None:
        from hivememory.core.config import MemoryPerceptionConfig
        config = MemoryPerceptionConfig()

    if not config.enable:
        logger.warning("感知层未启用 (config.enable=False)")
        return None

    if config.layer_type == "simple":
        # 创建 SimplePerceptionLayer
        logger.info("创建 SimplePerceptionLayer")

        # 读取 simple 子配置
        simple_config = config.simple

        # 创建可配置的 TriggerManager
        strategies = [
            MessageCountTrigger(threshold=simple_config.message_threshold),
            IdleTimeoutTrigger(timeout=simple_config.timeout_seconds),
        ]
        if simple_config.enable_semantic_trigger:
            strategies.append(SemanticBoundaryTrigger())

        trigger_manager = TriggerManager(strategies=strategies)

        return SimplePerceptionLayer(
            trigger_manager=trigger_manager,
            on_flush_callback=on_flush_callback,
        )

    else:  # semantic_flow
        # 创建 SemanticFlowPerceptionLayer
        logger.info("创建 SemanticFlowPerceptionLayer")

        # 读取 semantic_flow 子配置
        sf_config = config.semantic_flow

        # SemanticBoundaryAdsorber 的 embedding 模型参数
        embedding_model = sf_config.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"

        parser = UnifiedStreamParser()
        adsorber = SemanticBoundaryAdsorber(
            semantic_threshold=sf_config.semantic_threshold,
            short_text_threshold=sf_config.short_text_threshold,
            embedding_model=embedding_model,
            ema_alpha=sf_config.ema_alpha,
        )
        relay_controller = TokenOverflowRelayController(
            max_processing_tokens=sf_config.max_processing_tokens,
            enable_smart_summary=sf_config.enable_smart_summary,
        )

        return SemanticFlowPerceptionLayer(
            parser=parser,
            adsorber=adsorber,
            relay_controller=relay_controller,
            on_flush_callback=on_flush_callback,
            idle_timeout_seconds=sf_config.idle_timeout_seconds,
            scan_interval_seconds=sf_config.scan_interval_seconds,
        )


__all__ = [
    # 数据模型
    "StreamMessageType",
    "StreamMessage",
    "Triplet",
    "LogicalBlock",
    "BufferState",
    "SemanticBuffer",
    "SimpleBuffer",
    "FlushReason",  # 从 core.models 导入
    # 接口
    "StreamParser",
    "SemanticAdsorber",
    "RelayController",
    "IdleTimeoutMonitorInterface",
    "BasePerceptionLayer",
    # 感知层实现
    "SimplePerceptionLayer",
    "SemanticFlowPerceptionLayer",
    # 空闲超时监控
    "IdleTimeoutMonitor",
    # 触发策略
    "TriggerManager",
    "create_default_trigger_manager",
    # 具体实现
    "LocalEmbeddingService",
    "get_embedding_service",
    "UnifiedStreamParser",
    "SemanticBoundaryAdsorber",
    "TokenOverflowRelayController",
    # 配置
    "MemoryPerceptionConfig",
    "create_default_perception_config",
    "create_default_perception_layer",
]
