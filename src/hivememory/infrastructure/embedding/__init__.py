"""
HiveMemory Embedding 模块

暴露 Embedding 服务接口和工厂函数。
"""

from typing import Optional

from hivememory.infrastructure.embedding.base import BaseEmbeddingService
from hivememory.infrastructure.embedding.local import LocalEmbeddingService
from hivememory.infrastructure.embedding.bge_m3 import BGEM3EmbeddingService, get_bge_m3_service
from hivememory.patchouli.config import EmbeddingConfig, load_app_config

def get_embedding_service(config: Optional[EmbeddingConfig] = None) -> BaseEmbeddingService:
    """
    通用 Embedding 服务工厂函数
    
    根据配置自动选择合适的实现类 (Local 或 BGE-M3)，并返回实例。
    实例管理由 SingletonModelService (Multiton) 负责。
    """
    if config is None:
        # 默认加载 default 配置
        config = load_app_config().embedding.default

    # 根据模型名称选择实现类
    if "bge-m3" in config.model_name.lower():
        return BGEM3EmbeddingService(config=config)
    else:
        return LocalEmbeddingService(config=config)

def get_perception_embedding_service(config: Optional[EmbeddingConfig] = None) -> BaseEmbeddingService:
    """
    获取感知层 Embedding 服务
    """
    if config is None:
        config = load_app_config().embedding.perception
    
    return get_embedding_service(config)

def get_default_embedding_service(config: Optional[EmbeddingConfig] = None) -> BaseEmbeddingService:
    """
    获取默认/存储层 Embedding 服务
    """
    if config is None:
        config = load_app_config().embedding.default
        
    return get_embedding_service(config)

__all__ = [
    "BaseEmbeddingService",
    "LocalEmbeddingService",
    "BGEM3EmbeddingService",
    "get_bge_m3_service",
    "get_embedding_service",
    "get_perception_embedding_service",
    "get_default_embedding_service",
]
