"""
HiveMemory - 记忆生成模块 (MemoryGeneration)

该模块负责从对话流中提取、精炼和存储结构化记忆原子。

核心组件:
- ValueGater: 价值评估器，过滤无价值信息
- MemoryExtractor: LLM 驱动的记忆提取器
- Deduplicator: 查重与知识演化管理器
- MemoryGenerationOrchestrator: 编排器，协调所有组件

对应设计文档: PROJECT.md 第 4 章

作者: HiveMemory Team
版本: 0.3.0
"""

import logging
from typing import Optional, TYPE_CHECKING
from hivememory.memory.storage import QdrantMemoryStore

from hivememory.generation.orchestrator import MemoryGenerationOrchestrator
from hivememory.generation.interfaces import (
    ValueGater,
    MemoryExtractor,
    Deduplicator,
)

from hivememory.generation.models import (
    ConversationMessage,
    ExtractedMemoryDraft,
    DuplicateDecision,
)

from hivememory.generation.gating import (
    RuleBasedGater,
    LLMAssistedGater,
    HybridGater,
    create_default_gater,
)

from hivememory.generation.extractor import (
    LLMMemoryExtractor,
    create_default_extractor,
)

from hivememory.generation.deduplicator import (
    MemoryDeduplicator,
    create_default_deduplicator,
)

if TYPE_CHECKING:
    from hivememory.core.config import MemoryGenerationConfig

logger = logging.getLogger(__name__)


def create_default_generation_orchestrator(
    storage: QdrantMemoryStore,
    config: Optional["MemoryGenerationConfig"] = None,
) -> MemoryGenerationOrchestrator:
    """
    创建默认配置的记忆生成编排器

    根据 config 自动创建并配置所有子组件：
    - ValueGater: 价值评估器
    - MemoryExtractor: LLM 记忆提取器
    - Deduplicator: 查重与知识演化管理器

    Args:
        storage: QdrantMemoryStore 实例
        config: 生成配置（可选，使用默认配置）

    Returns:
        MemoryGenerationOrchestrator: 编排器实例

    Examples:
        >>> from hivememory.generation import create_default_generation_orchestrator
        >>> from hivememory.memory.storage import QdrantMemoryStore
        >>> storage = QdrantMemoryStore()
        >>> orchestrator = create_default_generation_orchestrator(storage=storage)
        >>>
        >>> # 使用自定义配置
        >>> from hivememory.core.config import MemoryGenerationConfig
        >>> config = MemoryGenerationConfig()
        >>> config.gater.gater_type = "llm"
        >>> orchestrator = create_default_generation_orchestrator(storage=storage, config=config)
    """
    if config is None:
        from hivememory.core.config import MemoryGenerationConfig
        config = MemoryGenerationConfig()

    # 创建子组件
    gater = create_default_gater(config.gater)
    extractor = create_default_extractor(config.extractor)
    deduplicator = create_default_deduplicator(storage, config.deduplicator)

    orchestrator = MemoryGenerationOrchestrator(
        storage=storage,
        gater=gater,
        extractor=extractor,
        deduplicator=deduplicator,
    )

    logger.info("MemoryGenerationOrchestrator created with default config")
    return orchestrator


__all__ = [
    # 接口
    "ValueGater",
    "MemoryExtractor",
    "Deduplicator",

    # 数据模型
    "ConversationMessage",
    "ExtractedMemoryDraft",
    "DuplicateDecision",

    # 价值评估
    "RuleBasedGater",
    "LLMAssistedGater",
    "HybridGater",
    "create_default_gater",

    # 记忆提取
    "LLMMemoryExtractor",
    "create_default_extractor",

    # 查重与演化
    "MemoryDeduplicator",
    "create_default_deduplicator",

    # 编排器
    "MemoryGenerationOrchestrator",
]
