"""配置管理 API 路由"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from hivememory.patchouli.config import HiveMemoryConfig, load_app_config
from hivememory.server.deps import get_system

logger = logging.getLogger(__name__)

router = APIRouter(tags=["config"])


class ConfigResponse(BaseModel):
    """配置响应模型"""
    system: Dict[str, Any]
    logging: Dict[str, Any]
    llm: Dict[str, Any]
    embedding: Dict[str, Any]
    qdrant: Dict[str, Any]
    redis: Dict[str, Any]
    gateway: Dict[str, Any]
    perception: Dict[str, Any]
    generation: Dict[str, Any]
    retrieval: Dict[str, Any]
    lifecycle: Dict[str, Any]
    koakuma: Dict[str, Any]


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """
    获取当前系统配置

    返回当前运行时的完整配置，包括所有模块的参数设置。
    """
    try:
        system = get_system()
        config = system.config

        # 将配置转换为字典格式
        config_dict = config.model_dump(mode='json')

        return ConfigResponse(**config_dict)
    except Exception as e:
        logger.error(f"获取配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.post("/config")
async def update_config(new_config: Dict[str, Any]):
    """
    更新系统配置

    接收新的配置参数并验证，然后更新运行时配置。
    注意：某些配置项（如基础设施连接）需要重启服务才能生效。

    Args:
        new_config: 新的配置字典

    Returns:
        更新后的配置
    """
    try:
        system = get_system()

        # 验证新配置
        try:
            validated_config = HiveMemoryConfig(**new_config)
        except ValidationError as e:
            logger.warning(f"配置验证失败: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Configuration validation failed: {e.errors()}"
            )

        # 更新系统配置
        # 注意：这里只更新内存中的配置，不会持久化到文件
        system.config = validated_config

        logger.info("配置已更新")

        # 返回更新后的配置
        config_dict = validated_config.model_dump(mode='json')
        return ConfigResponse(**config_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@router.get("/config/defaults", response_model=ConfigResponse)
async def get_default_config():
    """
    获取默认配置

    返回系统的默认配置值，不包含任何环境变量或YAML文件的覆盖。
    可用于重置配置或查看默认值。
    """
    try:
        # 创建一个新的配置实例，使用默认值
        default_config = HiveMemoryConfig()

        config_dict = default_config.model_dump(mode='json')

        return ConfigResponse(**config_dict)
    except Exception as e:
        logger.error(f"获取默认配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default configuration: {str(e)}")
