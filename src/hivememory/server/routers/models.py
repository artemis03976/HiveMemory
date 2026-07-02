"""模型注册表路由 — 管理可用的 LLM 模型"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from hivememory.core.models.model_definition import ModelDefinition
from hivememory.server.deps import get_model_registry
from hivememory.server.models.model_registry import (
    ModelCreateRequest,
    ModelResponse,
    ModelUpdateRequest,
)
from hivememory.system.model_registry import (
    DuplicateModelIdError,
    ModelNotFoundError,
    ModelRegistry,
)

router = APIRouter(tags=["models"])


# ------------------------------------------------------------------
# 路由处理器
# ------------------------------------------------------------------


@router.get("/models", response_model=List[ModelResponse])
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
        top_p=body.top_p,
        is_default=body.is_default,
    )
    try:
        registry.add_model(model)
    except DuplicateModelIdError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ModelResponse.from_definition(model)


@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """获取指定模型的详情（API 密钥已脱敏）"""
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在")
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
        raise HTTPException(status_code=400, detail="请求体中没有需要更新的字段")

    try:
        updated = registry.update_model(model_id, updates)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ModelResponse.from_definition(updated)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """从注册表中删除指定模型"""
    try:
        registry.delete_model(model_id)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
