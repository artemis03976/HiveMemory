"""提供商凭证路由 — 管理 LLM 提供商的 API 密钥与地址"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from hivememory.system.config.shared import ProviderCredentials
from hivememory.system.provider_registry import ProviderNotFoundError, ProviderRegistry
from hivememory.server.deps import get_provider_registry

router = APIRouter(tags=["providers"])


# ------------------------------------------------------------------
# 请求 / 响应模型
# ------------------------------------------------------------------


def _mask_api_key(api_key: Optional[str]) -> Optional[str]:
    """对 API 密钥做脱敏处理，只返回前后片段。"""
    if api_key is None:
        return None
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:3]}...{api_key[-4:]}"


class ProviderResponse(BaseModel):
    """提供商凭证的 API 响应体（api_key 已脱敏）"""

    name: str
    api_key_masked: Optional[str] = Field(
        default=None,
        description="脱敏后的 API 密钥；未设置则为 null",
    )
    api_base: Optional[str] = None
    is_from_env: bool = Field(
        default=False,
        description="True 表示来自环境变量（只读，不可通过 API 删除/覆盖）",
    )


class ProviderUpsertRequest(BaseModel):
    """创建或更新提供商凭证的请求体"""

    api_key: Optional[str] = Field(
        default=None,
        description="API 密钥。传 null 或不传表示不更改已有值（仅更新 api_base 时使用）",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="自定义 API 地址，留空使用提供商默认值",
    )


# ------------------------------------------------------------------
# 路由处理器
# ------------------------------------------------------------------


@router.get("/providers", response_model=List[ProviderResponse])
def list_providers(
    registry: ProviderRegistry = Depends(get_provider_registry),
):
    """列出所有已配置的提供商（API 密钥已脱敏）"""
    return [
        ProviderResponse(
            name=name,
            api_key_masked=_mask_api_key(cred.api_key),
            api_base=cred.api_base,
            is_from_env=from_env,
        )
        for name, cred, from_env in registry.list_all()
    ]


@router.put("/providers/{provider_name}", response_model=ProviderResponse)
def upsert_provider(
    provider_name: str,
    body: ProviderUpsertRequest,
    registry: ProviderRegistry = Depends(get_provider_registry),
):
    """
    创建或更新指定提供商的凭证。

    - 若提供商不存在则新建。
    - api_key 传 null 且提供商已存在时，保留已有 api_key（只更新 api_base）。
    - 来自环境变量的提供商同样可以在 yaml 层写入覆盖值，但 get() 时 env 优先。
    """
    lower = provider_name.lower()

    # 若不传 api_key，且 yaml 层已有记录，则保留原 api_key
    existing_yaml = registry._yaml.get(lower)
    if body.api_key is None and existing_yaml is not None:
        resolved_key = existing_yaml.api_key
    else:
        resolved_key = body.api_key

    cred = ProviderCredentials(api_key=resolved_key, api_base=body.api_base)
    try:
        registry.upsert(lower, cred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 返回合并后的凭证（env 层可能优先）
    final_cred = registry.get(lower) or cred
    from_env = lower in registry._env
    return ProviderResponse(
        name=lower,
        api_key_masked=_mask_api_key(final_cred.api_key),
        api_base=final_cred.api_base,
        is_from_env=from_env,
    )


@router.delete("/providers/{provider_name}", status_code=204)
def delete_provider(
    provider_name: str,
    registry: ProviderRegistry = Depends(get_provider_registry),
):
    """
    删除指定提供商的凭证（仅可删除 yaml 层；环境变量层无法通过 API 删除）。
    """
    try:
        registry.delete(provider_name.lower())
    except ProviderNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
