"""配置管理 API 路由"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import yaml
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ValidationError

from hivememory.patchouli.config import HiveMemoryConfig, get_config_file_path
from hivememory.system import HiveMemorySystem
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
    gateway: Dict[str, Any]
    perception: Dict[str, Any]
    generation: Dict[str, Any]
    retrieval: Dict[str, Any]
    lifecycle: Dict[str, Any]
    koakuma: Dict[str, Any]


def _build_config_without_path_override(config_data: Dict[str, Any] | None = None) -> HiveMemoryConfig:
    original_path = os.environ.pop("HIVEMEMORY_CONFIG_PATH", None)
    try:
        if config_data is None:
            return HiveMemoryConfig()
        return HiveMemoryConfig(**config_data)
    finally:
        if original_path is not None:
            os.environ["HIVEMEMORY_CONFIG_PATH"] = original_path


def _persist_config_atomically(config: HiveMemoryConfig) -> Path:
    config_path = get_config_file_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    temp_file_path: str | None = None
    try:
        fd, temp_file_path = tempfile.mkstemp(
            prefix=f"{config_path.name}.",
            suffix=".tmp",
            dir=str(config_path.parent),
            text=True,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            yaml.safe_dump(
                config.model_dump(mode="json"),
                temp_file,
                allow_unicode=True,
                sort_keys=False,
            )

        os.replace(temp_file_path, config_path)

        with open(config_path, "r", encoding="utf-8") as persisted_file:
            persisted_data = yaml.safe_load(persisted_file) or {}
        _build_config_without_path_override(persisted_data)
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise RuntimeError(f"配置文件持久化失败: {e}") from e

    return config_path


@router.get("/config", response_model=ConfigResponse)
async def get_config(
    system: HiveMemorySystem = Depends(get_system),
):
    """
    获取当前系统配置

    返回当前运行时的完整配置，包括所有模块的参数设置。
    """
    try:
        config = system.config
        config_dict = config.model_dump(mode='json')
        return ConfigResponse(**config_dict)
    except Exception as e:
        logger.error(f"获取配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.post("/config")
async def update_config(
    new_config: Dict[str, Any],
    system: HiveMemorySystem = Depends(get_system),
):
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
        try:
            validated_config = _build_config_without_path_override(new_config)
        except ValidationError as e:
            logger.warning(f"配置验证失败: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Configuration validation failed: {e.errors()}"
            )

        _persist_config_atomically(validated_config)
        system.config = validated_config

        logger.info("配置已更新")

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
        default_config = _build_config_without_path_override()
        config_dict = default_config.model_dump(mode='json')
        return ConfigResponse(**config_dict)
    except Exception as e:
        logger.error(f"获取默认配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get default configuration: {str(e)}")
