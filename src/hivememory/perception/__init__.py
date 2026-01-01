"""
HiveMemory - 帕秋莉感知层 (Perception Layer)

职责:
    作为系统的流量入口与预处理器，负责将非结构化的原始消息流（Raw Stream）
    转化为语义连贯的逻辑原子块（Logical Blocks），并决定何时唤醒 Librarian Agent。

核心组件:
    - StreamParser: 流式解析器，抹平不同 Agent 框架的消息格式差异
    - SemanticAdsorber: 语义吸附器，基于 Embedding 实时计算语义相似度
    - RelayController: 接力控制器，处理 Token 溢出并生成中间态摘要
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
    BasePerceptionLayer,
)
from hivememory.perception.embedding_service import (
    LocalEmbeddingService,
    get_embedding_service,
)
from hivememory.perception.stream_parser import UnifiedStreamParser
from hivememory.perception.semantic_adsorber import SemanticBoundaryAdsorber
from hivememory.perception.relay_controller import TokenOverflowRelayController
from hivememory.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.perception.simple_perception_layer import (
    SimplePerceptionLayer,
)
from hivememory.perception.trigger_strategies import (
    TriggerManager,
    create_default_trigger_manager,
)
from hivememory.perception.config import PerceptionConfig, create_default_perception_config

# 向后兼容别名
PerceptionLayer = SemanticFlowPerceptionLayer

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
    "BasePerceptionLayer", 
    # 感知层实现
    "SimplePerceptionLayer",  # 新增
    "SemanticFlowPerceptionLayer",  # 新增
    # 触发策略
    "TriggerManager",  # 新增
    "create_default_trigger_manager",  # 新增
    # 具体实现
    "LocalEmbeddingService",
    "get_embedding_service",
    "UnifiedStreamParser",
    "SemanticBoundaryAdsorber",
    "TokenOverflowRelayController",
    # 配置
    "PerceptionConfig",
    "create_default_perception_config",
    "create_default_perception_layer",
]


def create_default_perception_layer(on_flush_callback=None, config=None):
    """
    创建默认配置的语义流感知层实例

    Args:
        on_flush_callback: Flush 回调函数
        config: 感知层配置（可选，使用默认配置）

    Returns:
        SemanticFlowPerceptionLayer: 语义流感知层实例

    Examples:
        >>> from hivememory.perception import create_default_perception_layer
        >>> def on_flush(blocks, reason):
        ...     print(f"Flush: {reason}, Blocks: {len(blocks)}")
        >>> perception = create_default_perception_layer(on_flush_callback=on_flush)
    """
    if config is None:
        config = create_default_perception_config()

    parser = UnifiedStreamParser()
    adsorber = SemanticBoundaryAdsorber(
        semantic_threshold=config.semantic_threshold,
        short_text_threshold=config.short_text_threshold,
        idle_timeout_seconds=config.idle_timeout_seconds,
        embedding_model=config.embedding_model,
    )
    relay_controller = TokenOverflowRelayController(
        max_processing_tokens=config.max_processing_tokens,
    )

    return SemanticFlowPerceptionLayer(
        parser=parser,
        adsorber=adsorber,
        relay_controller=relay_controller,
        on_flush_callback=on_flush_callback,
    )
