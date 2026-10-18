"""模型注册表路由 — 管理可用的 LLM 模型"""

from fastapi import APIRouter, Depends

from hivememory.core.errors import ModelNotFoundError
from hivememory.core.models.model_definition import ModelDefinition
from hivememory.server.deps import get_model_registry
from hivememory.server.error_handlers import APIError, error_responses
from hivememory.server.models.model_registry import (
    ModelCreateRequest,
    ModelResponse,
    ModelUpdateRequest,
)
from hivememory.system.model_registry import (
    ModelRegistry,
)

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["models"])


# ------------------------------------------------------------------
# 路由处理器
# ------------------------------------------------------------------


@router.get("/models", response_model=list[ModelResponse])
def list_models(
    registry: ModelRegistry = Depends(get_model_registry),
):
    """列出注册表中的所有模型（API 密钥已脱敏）"""
    return [ModelResponse.from_definition(m) for m in registry.list_models()]


@router.post("/models", response_model=ModelResponse, status_code=201)
def create_model(
    body: ModelCreateRequest,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """向注册表中添加一个新模型"""
    model = ModelDefinition(
        id=body.id,
        display_name=body.display_name,
        litellm_model=body.litellm_model,
        provider=body.provider,
        api_key=body.api_key,
        api_base=body.api_base,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        context_window=body.context_window,
        top_p=body.top_p,
        is_default=body.is_default,
    )
    registry.add_model(model)

    return ModelResponse.from_definition(model)


@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """获取指定模型的详情（API 密钥已脱敏）"""
    model = registry.get_model(model_id)
    if model is None:
        raise ModelNotFoundError()
    return ModelResponse.from_definition(model)


@router.put("/models/{model_id}", response_model=ModelResponse)
def update_model(
    model_id: str,
    body: ModelUpdateRequest,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    更新指定模型的字段。

    只需传入要修改的字段，未传入的字段保持不变。
    传入 api_key="" 可清除已设置的密钥。
    """
    updates = body.to_updates_dict()
    if not updates:
        raise APIError("request.empty_update")

    updated = registry.update_model(model_id, updates)

    return ModelResponse.from_definition(updated)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """从注册表中删除指定模型"""
    registry.delete_model(model_id)
