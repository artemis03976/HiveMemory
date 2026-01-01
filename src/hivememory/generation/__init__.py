"""
HiveMemory - 记忆生成模块 (MemoryGeneration)

该模块负责从对话流中提取、精炼和存储结构化记忆原子。

核心组件:
- ValueGater: 价值评估器，过滤无价值信息
- MemoryExtractor: LLM 驱动的记忆提取器
- Deduplicator: 查重与知识演化管理器
- MemoryOrchestrator: 编排器，协调所有组件

注意:
    - ConversationBuffer 已迁移到 perception.simple_perception_layer
    - TriggerManager 已迁移到 perception.trigger_strategies

对应设计文档: PROJECT.md 第 4 章

作者: HiveMemory Team
版本: 0.2.0
"""

from hivememory.generation.orchestrator import MemoryOrchestrator
from hivememory.generation.interfaces import (
    ValueGater,
    MemoryExtractor,
    Deduplicator,
)

__all__ = [
    "MemoryOrchestrator",
    "ValueGater",
    "MemoryExtractor",
    "Deduplicator",
]
