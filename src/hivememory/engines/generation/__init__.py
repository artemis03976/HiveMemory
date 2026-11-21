"""
HiveMemory - 记忆生成模块 (MemoryGeneration)

该模块负责从对话流中提取、精炼和存储结构化记忆原子。

核心组件:
- MemoryExtractor: LLM 驱动的记忆提取器
- Librarian: 话题结算的馆长，先查看本话题的写入与库中相关记忆再写
- Deduplicator: 查重与知识演化管理器
- MemoryGenerationOrchestrator: 编排器，协调所有组件

跨域共享类型（``WriteFocus`` / ``UpdateFocus`` / ``PendingAtomSettlement``）
位于 ``hivememory.core.models``，本模块不转出。
"""

import logging

from hivememory.engines.generation.alias import AliasGenerator
from hivememory.engines.generation.deduplicator import (
    MemoryDeduplicator,
    NoOpDeduplicator,
    create_deduplicator,
)
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.extractor import (
    LLMMemoryExtractor,
    NoOpMemoryExtractor,
    create_extractor,
)
from hivememory.engines.generation.interfaces import (
    BaseDeduplicator,
    BaseMemoryExtractor,
)
from hivememory.engines.generation.librarian import Librarian, create_librarian
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationAction,
    GenerationOutcome,
    GenerationRequest,
    MemoryEditSet,
    MemoryProvenance,
)

logger = logging.getLogger(__name__)


__all__ = [
    # 接口
    "BaseMemoryExtractor",
    "BaseDeduplicator",
    # 生成流水线 DTO
    "DuplicateDecision",
    "ExtractedMemoryDraft",
    "GenerationAction",
    "GenerationOutcome",
    "GenerationRequest",
    "MemoryEditSet",
    "MemoryProvenance",
    # 记忆提取
    "LLMMemoryExtractor",
    "NoOpMemoryExtractor",
    "create_extractor",
    # 话题结算的馆长
    "Librarian",
    "create_librarian",
    # 查重与演化
    "MemoryDeduplicator",
    "NoOpDeduplicator",
    "create_deduplicator",
    # 别名生成
    "AliasGenerator",
    # 引擎
    "MemoryGenerationEngine",
]
