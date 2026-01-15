"""
HiveMemory Embedding 模块

暴露 Embedding 服务接口和工厂函数。
"""

from hivememory.core.embedding.base import BaseEmbeddingService
from hivememory.core.embedding.local import (
    LocalEmbeddingService,
    get_embedding_service,
)
from hivememory.core.embedding.bge_m3 import (
    BGEM3EmbeddingService,
    get_bge_m3_service
)

__all__ = [
    "BaseEmbeddingService",
    "LocalEmbeddingService",
    "BGEM3EmbeddingService",
    "get_embedding_service",
    "get_bge_m3_service",
]
