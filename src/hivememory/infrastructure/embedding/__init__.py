"""
HiveMemory Embedding 模块

暴露 Embedding 服务接口和工厂函数。
"""

from hivememory.config.shared import EmbeddingConfig
from hivememory.infrastructure.embedding.base import BaseEmbeddingService
from hivememory.infrastructure.embedding.bge_m3 import BGEM3EmbeddingService, get_bge_m3_service


def get_embedding_service(config: EmbeddingConfig) -> BaseEmbeddingService:
    """
    通用 Embedding 服务工厂函数（配置由调用方显式提供）
    """
    return BGEM3EmbeddingService(config=config)


def get_default_embedding_service(config: EmbeddingConfig) -> BaseEmbeddingService:
    """
    获取默认/存储层 Embedding 服务
    """
    return get_embedding_service(config)


def get_perception_embedding_service(config: EmbeddingConfig) -> BaseEmbeddingService:
    """
    获取 perception Embedding 服务（已与 default 配置合并）
    """
    return get_embedding_service(config)


__all__ = [
    "BaseEmbeddingService",
    "BGEM3EmbeddingService",
    "get_bge_m3_service",
    "get_embedding_service",
    "get_default_embedding_service",
    "get_perception_embedding_service",
]
