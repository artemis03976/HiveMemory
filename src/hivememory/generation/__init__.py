"""
HiveMemory - 记忆生成模块 (MemoryGeneration)

该模块负责从对话流中提取、精炼和存储结构化记忆原子。

核心组件:
- ValueGater: 价值评估器，过滤无价值信息
- MemoryExtractor: LLM 驱动的记忆提取器
- Deduplicator: 查重与知识演化管理器
- TriggerManager: 对话触发策略管理
- ConversationBuffer: 对话缓冲器
- MemoryOrchestrator: 编排器，协调所有组件

对应设计文档: PROJECT.md 第 4 章

作者: HiveMemory Team
版本: 0.1.0
"""

from hivememory.generation.orchestrator import MemoryOrchestrator
from hivememory.generation.buffer import ConversationBuffer
from hivememory.generation.interfaces import (
    ValueGater,
    MemoryExtractor,
    Deduplicator,
    TriggerStrategy,
)

__all__ = [
    "MemoryOrchestrator",
    "ConversationBuffer",
    "ValueGater",
    "MemoryExtractor",
    "Deduplicator",
    "TriggerStrategy",
]
