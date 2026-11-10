"""
HiveMemory Embedding 模块

暴露 Embedding 服务接口与存储层使用的 BGE-M3 服务。
"""

from hivememory.infrastructure.embedding.base import BaseEmbeddingService
from hivememory.infrastructure.embedding.bge_m3 import (
    BGE_M3_DIMENSION,
    BGEM3EmbeddingService,
    get_bge_m3_service,
)

__all__ = [
    "BaseEmbeddingService",
    "BGE_M3_DIMENSION",
    "BGEM3EmbeddingService",
    "get_bge_m3_service",
]
