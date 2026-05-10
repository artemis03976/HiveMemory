"""
HiveMemory Embedding 模块

暴露 Embedding 服务接口和工厂函数。
"""

from typing import Optional

from hivememory.infrastructure.embedding.base import BaseEmbeddingService
from hivememory.infrastructure.embedding.bge_m3 import BGEM3EmbeddingService, get_bge_m3_service
from hivememory.patchouli.config import EmbeddingConfig, load_app_config


def get_embedding_service(config: Optional[EmbeddingConfig] = None) -> BaseEmbeddingService:
    """
    通用 Embedding 服务工厂函数
    """
    if config is None:
        config = load_app_config().embedding.default

    return BGEM3EmbeddingService(config=config)


def get_default_embedding_service(config: Optional[EmbeddingConfig] = None) -> BaseEmbeddingService:
    """
    获取默认/存储层 Embedding 服务
    """
    if config is None:
        config = load_app_config().embedding.default

    return get_embedding_service(config)


def get_perception_embedding_service(config: Optional[EmbeddingConfig] = None) -> BaseEmbeddingService:
    """
    获取 perception Embedding 服务（已与 default 配置合并）
    """
    if config is None:
        config = load_app_config().embedding.default

    return get_embedding_service(config)


__all__ = [
    "BaseEmbeddingService",
    "BGEM3EmbeddingService",
    "get_bge_m3_service",
    "get_embedding_service",
    "get_default_embedding_service",
    "get_perception_embedding_service",
]
