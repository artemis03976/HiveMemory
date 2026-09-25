"""Memories 路由 — 记忆 CRUD"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from hivememory.core.errors import InvalidMemoryFieldError
from hivememory.core.models import IdentityScope
from hivememory.server.deps import get_identity_scope, get_memory_service
from hivememory.server.models.memory import (
    MemoryCreateRequest,
    MemoryFeedbackRequest,
    MemoryFeedbackResponse,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
)
from hivememory.system.application.memory_service import (
    MemoryApplicationService,
    MemoryLifecycleUnavailableError,
    MemoryNotFoundError,
)

router = APIRouter(tags=["memories"])


@router.post("/memories", response_model=MemoryResponse, status_code=201)
async def create_memory(
    body: MemoryCreateRequest,
    service: MemoryApplicationService = Depends(get_memory_service),
    identity_scope: IdentityScope = Depends(get_identity_scope),
):
    """创建新的记忆（管理用例，actor 为保留 system）"""
    try:
        atom = await service.create_memory(
            identity_scope=identity_scope,
            title=body.title,
            summary=body.summary,
            content=body.content,
            memory_type=body.memory_type,
            tags=body.tags,
            alias=body.alias,
        )
    except InvalidMemoryFieldError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return MemoryResponse.from_atom(atom)


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    query: str = Query(default=None, description="语义搜索查询"),
    memory_type: str = Query(default=None, description="按记忆类型过滤"),
    limit: int = Query(default=20, le=100, description="最大返回数量"),
    service: MemoryApplicationService = Depends(get_memory_service),
    identity_scope: IdentityScope = Depends(get_identity_scope),
):
    """检索记忆 — 支持语义搜索和过滤（owner-management 语义，不做 Agent 可见性过滤）"""
    atoms = await service.list_memories(
        identity_scope=identity_scope,
        query=query,
        memory_type=memory_type,
        limit=limit,
    )
    memories = [MemoryResponse.from_atom(a) for a in atoms]
    return MemoryListResponse(memories=memories, total=len(memories))


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    service: MemoryApplicationService = Depends(get_memory_service),
    identity_scope: IdentityScope = Depends(get_identity_scope),
):
    """获取单条记忆详情"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    try:
        atom = await service.get_memory(uid, identity_scope=identity_scope)
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return MemoryResponse.from_atom(atom)


@router.patch("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    body: MemoryUpdateRequest,
    service: MemoryApplicationService = Depends(get_memory_service),
    identity_scope: IdentityScope = Depends(get_identity_scope),
):
    """更新记忆的可编辑字段"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    try:
        atom = await service.update_memory(
            uid,
            identity_scope=identity_scope,
            title=body.title,
            summary=body.summary,
            content=body.content,
            alias=body.alias,
            tags=body.tags,
            agent_config=body.agent_config,
        )
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="记忆不存在")
    except InvalidMemoryFieldError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return MemoryResponse.from_atom(atom)


@router.post("/memories/{memory_id}/feedback", response_model=MemoryFeedbackResponse)
async def record_memory_feedback(
    memory_id: str,
    body: MemoryFeedbackRequest,
    service: MemoryApplicationService = Depends(get_memory_service),
    identity_scope: IdentityScope = Depends(get_identity_scope),
):
    """记录用户对某条记忆的显式反馈。"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    try:
        result = await service.record_feedback(
            uid,
            identity_scope=identity_scope,
            positive=body.positive,
            source=body.source,
        )
    except MemoryLifecycleUnavailableError:
        raise HTTPException(status_code=503, detail="Memory lifecycle engine is unavailable")
    except MemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return MemoryFeedbackResponse(
        success=True,
        id=str(result.memory_id),
        positive=body.positive,
        previous_vitality=result.previous_vitality,
        new_vitality=result.new_vitality,
        previous_confidence=result.previous_confidence,
        new_confidence=result.new_confidence,
        event_type=(
            result.event_type.value
            if hasattr(result.event_type, "value")
            else str(result.event_type)
        ),
    )


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    service: MemoryApplicationService = Depends(get_memory_service),
    identity_scope: IdentityScope = Depends(get_identity_scope),
):
    """删除记忆"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    success = await service.delete_memory(uid, identity_scope=identity_scope)
    if not success:
        raise HTTPException(status_code=404, detail="记忆不存在或删除失败")

    return {"success": True, "id": memory_id}
