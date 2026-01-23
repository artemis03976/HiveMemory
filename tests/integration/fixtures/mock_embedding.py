"""
Mock Embedding 服务

用于集成测试中替代真实的 Embedding 模型。
"""

from typing import List
from .mock_llm import MockEmbeddingService

__all__ = ["MockEmbeddingService"]
