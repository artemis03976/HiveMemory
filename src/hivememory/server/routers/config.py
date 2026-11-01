"""配置管理 API 路由

主配置接口不读写密钥。配置中的 LLM 与 Qdrant 带有 ``api_key``，运行时的值只来自
环境变量与 ``.env``（提供商凭证由 Provider 接口管理，不属于主配置）：

- 响应中的密钥与提供商、模型接口一样只返回打码值，HTTP 接口对本机任何进程开放；
- 保存时提交中的 ``api_key`` 一律沿用当前的运行时值，前端原样提交读到的打码值
  不会覆盖真实密钥；
- 写入 ``config.yaml`` 的内容不含 ``api_key``：该文件受版本控制，出现密钥时加载失败。
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends
from pydantic import ValidationError

from hivememory.config.app import (
    SECRET_FIELD,
    HiveMemoryConfig,
    get_config_file_path,
)
from hivememory.server.deps import get_system
from hivememory.server.error_handlers import APIError, error_responses, validation_issues
from hivememory.server.models.config import ConfigResponse
from hivememory.server.models.secrets import mask_api_key
from hivememory.system import HiveMemorySystem

logger = logging.getLogger(__name__)

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["config"])


def _masked_response(config: HiveMemoryConfig) -> ConfigResponse:
    return ConfigResponse(**_mask_secrets(config.model_dump(mode="json")))


def _mask_secrets(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: mask_api_key(value) if key == SECRET_FIELD else _mask_secrets(value)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_mask_secrets(item) for item in node]
    return node


def _keep_current_secrets(submitted: Any, current: Any) -> Any:
    """提交中的密钥一律换成当前的运行时值：密钥不经主配置接口修改。"""
    if not isinstance(submitted, dict):
        return submitted
    current = current if isinstance(current, dict) else {}
    return {
        key: (
            current.get(key)
            if key == SECRET_FIELD
            else _keep_current_secrets(value, current.get(key))
        )
        for key, value in submitted.items()
    }


def _without_secrets(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _without_secrets(value) for key, value in node.items() if key != SECRET_FIELD}
    if isinstance(node, list):
        return [_without_secrets(item) for item in node]
    return node


def _build_config_without_path_override(
    config_data: dict[str, Any] | None = None,
) -> HiveMemoryConfig:
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
                _without_secrets(config.model_dump(mode="json")),
                temp_file,
                allow_unicode=True,
                sort_keys=False,
            )

        os.replace(temp_file_path, config_path)

        with open(config_path, encoding="utf-8") as persisted_file:
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

    返回当前运行时的完整配置，包括所有模块的参数设置；密钥只返回打码值。
    """
    return _masked_response(system.config)


@router.post("/config")
async def update_config(
    new_config: dict[str, Any],
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
    submitted = _keep_current_secrets(new_config, system.config.model_dump(mode="json"))
    try:
        validated_config = _build_config_without_path_override(submitted)
    except ValidationError as e:
        # 校验异常含完整 input；响应与常规日志均不复制该输入。
        raise APIError("request.validation_failed", issues=validation_issues(e.errors())) from e
    _persist_config_atomically(validated_config)
    system.config = validated_config
    logger.info("配置已更新")
    return _masked_response(validated_config)


@router.get("/config/defaults", response_model=ConfigResponse)
async def get_default_config():
    """
    获取默认配置

    返回系统的默认配置值，不包含任何环境变量或YAML文件的覆盖。
    可用于重置配置或查看默认值；密钥只返回打码值。
    """
    return _masked_response(_build_config_without_path_override())
