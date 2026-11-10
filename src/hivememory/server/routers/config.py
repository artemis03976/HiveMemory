"""配置管理 API 路由

主配置有两份：运行配置是启动时装配、组件正在使用的配置；期望配置是下次启动将装载
的配置。读取返回期望配置；保存只把与默认值不同的字段写进个人覆盖文件
``config.local.yaml``，其中可立即生效的字段同时原地写入运行配置，其余字段等待
重启。仓库中的 ``config.yaml`` 不被写。

主配置接口不读写密钥。配置中的 Qdrant 带有 ``api_key``，运行时的值只来自环境变量与
``.env``（模型的提供商凭证由 Provider 接口管理，不属于主配置）：

- 响应中的密钥与提供商接口一样只返回打码值，HTTP 接口对本机任何进程开放；
- 保存时提交中的 ``api_key`` 一律沿用当前的运行时值，前端原样提交读到的打码值
  不会覆盖真实密钥；
- 覆盖文件不写 ``api_key``，出现密钥时加载失败。
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import ValidationError

from hivememory.config.app import (
    SECRET_FIELD,
    HiveMemoryConfig,
    load_default_config,
    write_local_overrides,
)
from hivememory.server.deps import get_system
from hivememory.server.error_handlers import APIError, error_responses, validation_issues
from hivememory.server.models.config import ConfigResponse
from hivememory.server.models.secrets import mask_api_key
from hivememory.system import HiveMemorySystem
from hivememory.system.running_config import (
    IMMEDIATE_FIELDS,
    apply_immediate_fields,
    pending_restart,
)

logger = logging.getLogger(__name__)

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["config"])


def _config_response(running: HiveMemoryConfig, desired: HiveMemoryConfig) -> ConfigResponse:
    return ConfigResponse(
        config=_mask_secrets(desired.model_dump(mode="json")),
        immediate_fields=list(IMMEDIATE_FIELDS),
        pending_restart=pending_restart(running, desired),
    )


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


@router.get("/config", response_model=ConfigResponse)
async def get_config(
    system: HiveMemorySystem = Depends(get_system),
):
    """读取期望配置，并列出可立即生效的字段与等待重启的字段；密钥只返回打码值。"""
    return _config_response(system.config, HiveMemoryConfig())


@router.post("/config", response_model=ConfigResponse)
async def update_config(
    new_config: dict[str, Any],
    system: HiveMemorySystem = Depends(get_system),
):
    """保存主配置。

    提交的完整配置通过校验后，只把与默认值不同的字段写进个人覆盖文件；再从文件重新
    装载期望配置，其中可立即生效的字段原地写入运行配置。环境变量覆盖的字段以环境
    变量为准，提交的值不生效。各步之间没有 await，两次保存不会交错。
    """
    submitted = _keep_current_secrets(new_config, system.config.model_dump(mode="json"))
    try:
        validated = HiveMemoryConfig(**submitted)
    except ValidationError as e:
        # 校验异常含完整 input；响应与常规日志均不复制该输入。
        raise APIError("request.validation_failed", issues=validation_issues(e.errors())) from e
    write_local_overrides(validated)
    desired = HiveMemoryConfig()
    applied = apply_immediate_fields(system.config, desired)
    logger.info("主配置已保存: immediate_applied=%d", len(applied))
    return _config_response(system.config, desired)


@router.get("/config/defaults", response_model=dict[str, Any])
async def get_default_config():
    """默认配置：不含个人覆盖文件，含仓库中的 ``config.yaml``、``.env`` 与环境变量。

    把它原样保存即清空个人覆盖文件；密钥只返回打码值。
    """
    return _mask_secrets(load_default_config().model_dump(mode="json"))
