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

from hivememory.core.models import FlushReason
from hivememory.engines.perception.models import (
    StreamMessageType,
    StreamMessage,
    Triplet,
    LogicalBlock,
    BufferState,
    SemanticBuffer,
    SimpleBuffer,
)
from hivememory.engines.perception.interfaces import (
    StreamParser,
    SemanticAdsorber,
    RelayController,
    BasePerceptionLayer,
)
from hivememory.infrastructure.embedding import (
    LocalEmbeddingService,
    get_embedding_service,
)
from hivememory.engines.perception.stream_parser import UnifiedStreamParser
from hivememory.engines.perception.semantic_adsorber import SemanticBoundaryAdsorber
from hivememory.engines.perception.relay_controller import TokenOverflowRelayController
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.engines.perception.simple_perception_layer import (
    SimplePerceptionLayer,
)
from hivememory.engines.perception.trigger_strategies import (
    TriggerManager,
    create_default_trigger_manager,
    MessageCountTrigger,
    SemanticBoundaryTrigger,
)
from hivememory.engines.perception.context_bridge import (
    ContextBridge,
    DEFAULT_STOP_WORDS,
)
from hivememory.engines.perception.grey_area_arbiter import (
    GreyAreaArbiter,
    RerankerArbiter,
    SLMArbiter,
    NoOpArbiter,
    DEFAULT_ARBITER_PROMPT,
)
from hivememory.patchouli.config import MemoryPerceptionConfig
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def create_default_perception_layer(
    config: Optional[MemoryPerceptionConfig] = None,
    embedding_service: Optional["BaseEmbeddingService"] = None,
    on_flush_callback=None,
) -> Optional[Union[SemanticFlowPerceptionLayer, SimplePerceptionLayer]]:
    """
    创建默认配置的感知层实例

    根据 config.layer_type 自动选择：
    - "semantic_flow": SemanticFlowPerceptionLayer（默认）
    - "simple": SimplePerceptionLayer

    Args:
        config: 感知层配置（MemoryPerceptionConfig，可选，使用默认配置）
        embedding_service: Embedding 服务实例（推荐通过依赖注入传入）
        on_flush_callback: Flush 回调函数

    Returns:
        SemanticFlowPerceptionLayer 或 SimplePerceptionLayer 实例

    Examples:
        >>> from hivememory.perception import create_default_perception_layer
        >>> def on_flush(blocks, reason):
        ...     print(f"Flush: {reason}, Blocks: {len(blocks)}")
        >>> perception = create_default_perception_layer(on_flush_callback=on_flush)
    """
    if config is None:
        from hivememory.patchouli.config import MemoryPerceptionConfig
        config = MemoryPerceptionConfig()

    if not config.enable:
        logger.warning("感知层未启用 (config.enable=False)")
        return None

    if config.layer_type == "simple":
        # 创建 SimplePerceptionLayer
        logger.info("创建 SimplePerceptionLayer")

        # 读取 simple 子配置
        simple_config = config.simple

        # 创建可配置的 TriggerManager（不包含 IdleTimeoutTrigger）
        strategies = [
            MessageCountTrigger(threshold=simple_config.message_threshold),
        ]
        if simple_config.enable_semantic_trigger:
            strategies.append(SemanticBoundaryTrigger())

        trigger_manager = TriggerManager(strategies=strategies)

        perception = SimplePerceptionLayer(
            trigger_manager=trigger_manager,
            on_flush_callback=on_flush_callback,
        )

        # 启动空闲超时监控（替代 IdleTimeoutTrigger）
        perception.start_idle_monitor(
            idle_timeout_seconds=simple_config.timeout_seconds,
            scan_interval_seconds=30,
        )

        return perception

    else:  # semantic_flow
        # 创建 SemanticFlowPerceptionLayer
        logger.info("创建 SemanticFlowPerceptionLayer")

        # 读取 semantic_flow 子配置
        sf_config = config.semantic_flow

        parser = UnifiedStreamParser()

        # 如果未传入 embedding_service，则创建（向后兼容）
        if embedding_service is None:
            embedding_model = sf_config.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
            embedding_service = LocalEmbeddingService(
                model_name=embedding_model,
                device=sf_config.embedding_device or "cpu",
                cache_dir=sf_config.embedding_cache_dir,
                batch_size=sf_config.embedding_batch_size or 32,
            )
            logger.info("未传入 embedding_service，使用默认配置创建")

        adsorber = SemanticBoundaryAdsorber(
            config=sf_config.adsorber,
            embedding_service=embedding_service,
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
    "BasePerceptionLayer",
    "GreyAreaArbiter",
    # 感知层实现
    "SimplePerceptionLayer",
    "SemanticFlowPerceptionLayer",
    # 触发策略
    "TriggerManager",
    "create_default_trigger_manager",
    "MessageCountTrigger",
    "SemanticBoundaryTrigger",
    # 具体实现
    "LocalEmbeddingService",
    "get_embedding_service",
    "UnifiedStreamParser",
    "SemanticBoundaryAdsorber",
    "TokenOverflowRelayController",
    # Phase 3 新增组件
    "ContextBridge",
    "DEFAULT_STOP_WORDS",
    "RerankerArbiter",
    "SLMArbiter",
    "NoOpArbiter",
    "DEFAULT_ARBITER_PROMPT",
    # 配置
    "MemoryPerceptionConfig",
    "create_default_perception_layer",
]
