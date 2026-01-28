"""
HiveMemory - 记忆生成模块 (MemoryGeneration)

该模块负责从对话流中提取、精炼和存储结构化记忆原子。

核心组件:
- MemoryExtractor: LLM 驱动的记忆提取器
- Deduplicator: 查重与知识演化管理器
- MemoryGenerationOrchestrator: 编排器，协调所有组件

对应设计文档: PROJECT.md 第 4 章

作者: HiveMemory Team
版本: 0.3.0
"""

import logging

from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.interfaces import (
    BaseMemoryExtractor,
    BaseDeduplicator,
)

from hivememory.engines.generation.models import (
    ExtractedMemoryDraft,
    DuplicateDecision,
)

from hivememory.engines.generation.extractor import (
    LLMMemoryExtractor,
    NoOpMemoryExtractor,
    create_extractor,
)

from hivememory.engines.generation.deduplicator import (
    MemoryDeduplicator,
    NoOpDeduplicator,
    create_deduplicator,
)

logger = logging.getLogger(__name__)


__all__ = [
    # 接口
    "BaseMemoryExtractor",
    "BaseDeduplicator",
    # 数据模型
    "ExtractedMemoryDraft",
    "DuplicateDecision",
    # 记忆提取
    "LLMMemoryExtractor",
    "NoOpMemoryExtractor",
    "create_extractor",
    # 查重与演化
    "MemoryDeduplicator",
    "NoOpDeduplicator",
    "create_deduplicator",
    # 引擎
    "MemoryGenerationEngine",
]
