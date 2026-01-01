"""
HiveMemory 感知层配置

定义感知层的配置类，支持从环境变量和配置文件加载。

参考: PROJECT.md 8.3 节

作者: HiveMemory Team
版本: 1.0.0
"""

import os
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PerceptionConfig(BaseModel):
    """
    感知层配置

    Attributes:
        enable: 是否启用感知层
        embedding_model: Embedding 模型名称
        embedding_device: 运行设备（cpu/cuda）
        embedding_cache_dir: 模型缓存目录
        semantic_threshold: 语义相似度阈值 (0-1)
        short_text_threshold: 短文本强吸附阈值（tokens）
        max_processing_tokens: 单次处理的最大 Token 数
        max_buffer_tokens: Buffer 的最大 Token 数
        idle_timeout_seconds: 空闲超时时间（秒）
        ema_alpha: 指数移动平均系数
    """

    # 启用开关
    enable: bool = Field(
        default=False,
        description="是否启用感知层（False 时使用原有触发机制）"
    )

    # Embedding 配置
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding 模型名称（HuggingFace 模型 ID）"
    )
    embedding_device: str = Field(
        default="cpu",
        description="运行设备：cpu 或 cuda"
    )
    embedding_cache_dir: Optional[str] = Field(
        default=None,
        description="模型缓存目录（默认使用系统缓存）"
    )

    # 语义吸附配置
    semantic_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="语义相似度阈值，超过此值认为是同一话题"
    )
    short_text_threshold: int = Field(
        default=50,
        gt=0,
        description="短文本强吸附阈值（tokens），少于此值强制吸附"
    )

    # Token 限制
    max_processing_tokens: int = Field(
        default=8192,
        gt=0,
        description="单次处理的最大 Token 数"
    )
    max_buffer_tokens: int = Field(
        default=16384,
        gt=0,
        description="Buffer 的最大 Token 数"
    )

    # 超时配置
    idle_timeout_seconds: int = Field(
        default=900,
        gt=0,
        description="空闲超时时间（秒），默认 15 分钟"
    )

    # EMA 配置
    ema_alpha: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="指数移动平均系数，用于更新话题核心向量"
    )

    @field_validator("embedding_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """验证设备名称"""
        v = v.lower()
        if v not in ("cpu", "cuda", "mps"):
            raise ValueError("device 必须是 cpu、cuda 或 mps")
        return v

    @classmethod
    def from_env(cls) -> "PerceptionConfig":
        """
        从环境变量加载配置

        环境变量：
            HIVEMEMORY_PERCEPTION_ENABLE: 是否启用感知层
            HIVEMEMORY_PERCEPTION_EMBEDDING_MODEL: Embedding 模型
            HIVEMEMORY_PERCEPTION_EMBEDDING_DEVICE: 运行设备
            HIVEMEMORY_PERCEPTION_SEMANTIC_THRESHOLD: 语义阈值
            HIVEMEMORY_PERCEPTION_SHORT_TEXT_THRESHOLD: 短文本阈值
            HIVEMEMORY_PERCEPTION_MAX_TOKENS: 最大 Token 数
            HIVEMEMORY_PERCEPTION_IDLE_TIMEOUT: 空闲超时

        Returns:
            PerceptionConfig: 配置对象
        """
        config = cls()

        if "HIVEMEMORY_PERCEPTION_ENABLE" in os.environ:
            config.enable = os.environ["HIVEMEMORY_PERCEPTION_ENABLE"].lower() in ("true", "1", "yes")

        if "HIVEMEMORY_PERCEPTION_EMBEDDING_MODEL" in os.environ:
            config.embedding_model = os.environ["HIVEMEMORY_PERCEPTION_EMBEDDING_MODEL"]

        if "HIVEMEMORY_PERCEPTION_EMBEDDING_DEVICE" in os.environ:
            config.embedding_device = os.environ["HIVEMEMORY_PERCEPTION_EMBEDDING_DEVICE"]

        if "HIVEMEMORY_PERCEPTION_EMBEDDING_CACHE_DIR" in os.environ:
            config.embedding_cache_dir = os.environ["HIVEMEMORY_PERCEPTION_EMBEDDING_CACHE_DIR"]

        if "HIVEMEMORY_PERCEPTION_SEMANTIC_THRESHOLD" in os.environ:
            try:
                config.semantic_threshold = float(os.environ["HIVEMEMORY_PERCEPTION_SEMANTIC_THRESHOLD"])
            except ValueError:
                pass

        if "HIVEMORYORY_PERCEPTION_SHORT_TEXT_THRESHOLD" in os.environ:
            try:
                config.short_text_threshold = int(os.environ["HIVEMEMORY_PERCEPTION_SHORT_TEXT_THRESHOLD"])
            except ValueError:
                pass

        if "HIVEMEMORY_PERCEPTION_MAX_TOKENS" in os.environ:
            try:
                config.max_processing_tokens = int(os.environ["HIVEMEMORY_PERCEPTION_MAX_TOKENS"])
            except ValueError:
                pass

        if "HIVEMEMORY_PERCEPTION_IDLE_TIMEOUT" in os.environ:
            try:
                config.idle_timeout_seconds = int(os.environ["HIVEMEMORY_PERCEPTION_IDLE_TIMEOUT"])
            except ValueError:
                pass

        return config

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "enable": True,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_device": "cpu",
                "semantic_threshold": 0.6,
                "short_text_threshold": 50,
                "max_processing_tokens": 8192,
                "idle_timeout_seconds": 900,
                "ema_alpha": 0.3,
            }
        }


def create_default_perception_config(**kwargs) -> PerceptionConfig:
    """
    创建默认配置的感知层配置对象

    Args:
        **kwargs: 覆盖默认配置的参数

    Returns:
        PerceptionConfig: 配置对象

    Examples:
        >>> # 使用默认配置
        >>> config = create_default_perception_config()
        >>>
        >>> # 覆盖部分配置
        >>> config = create_default_perception_config(
        ...     enable=True,
        ...     semantic_threshold=0.7
        ... )
        >>>
        >>> # 从环境变量加载
        >>> config = PerceptionConfig.from_env()
    """
    # 从环境变量加载基础配置
    config = PerceptionConfig.from_env()

    # 应用覆盖
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


__all__ = [
    "PerceptionConfig",
    "create_default_perception_config",
]
