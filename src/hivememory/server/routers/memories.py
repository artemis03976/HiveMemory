"""Memories 路由 — 记忆 CRUD"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from hivememory.core.errors import InvalidMemoryFieldError, MemoryAliasConflictError
from hivememory.server.deps import RequestAccess, get_memory_service, get_request_access
from hivememory.server.models.memory import (
    MemoryCreateRequest,
    MemoryFeedbackRequest,
    MemoryFeedbackResponse,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
)
from hivememory.workspace.capability.memory import (
    MemoryApplicationService,
    MemoryLifecycleUnavailableError,
    MemoryNotFoundError,
)

router = APIRouter(tags=["memories"])


@router.post("/memories", response_model=MemoryResponse, status_code=201)
async def create_memory(
    body: MemoryCreateRequest,
    service: MemoryApplicationService = Depends(get_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """创建新的记忆（管理用例，actor 为保留 system）"""
    try:
        atom = await service.create_memory(
            target_workspace=request_access.target_workspace,
            title=body.title,
            summary=body.summary,
            content=body.content,
            memory_type=body.memory_type,
            tags=body.tags,
            alias=body.alias,
            access=request_access.access,
        )
    except InvalidMemoryFieldError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except MemoryAliasConflictError as exc:
        # alias 在同一 Workspace 内已被占用：资源状态冲突，不是输入格式错误。
        raise HTTPException(status_code=409, detail=str(exc))
    return MemoryResponse.from_atom(atom)


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    query: str = Query(default=None, description="语义搜索查询"),
    memory_type: str = Query(default=None, description="按记忆类型过滤"),
    limit: int = Query(default=20, le=100, description="最大返回数量"),
    service: MemoryApplicationService = Depends(get_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """检索记忆 — 支持语义搜索和过滤（owner-management 语义，不做 Agent 可见性过滤）"""
    atoms = await service.list_memories(
        target_workspace=request_access.target_workspace,
        query=query,
        memory_type=memory_type,
        limit=limit,
        access=request_access.access,
    )
    memories = [MemoryResponse.from_atom(a) for a in atoms]
    return MemoryListResponse(memories=memories, total=len(memories))


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    service: MemoryApplicationService = Depends(get_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """获取单条记忆详情"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    try:
        atom = await service.get_memory(
            uid,
            target_workspace=request_access.target_workspace,
            access=request_access.access,
        )
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return MemoryResponse.from_atom(atom)


@router.patch("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    body: MemoryUpdateRequest,
    service: MemoryApplicationService = Depends(get_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """更新记忆的可编辑字段"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    try:
        atom = await service.update_memory(
            uid,
            target_workspace=request_access.target_workspace,
            title=body.title,
            summary=body.summary,
            content=body.content,
            alias=body.alias,
            tags=body.tags,
            agent_config=body.agent_config,
            access=request_access.access,
        )
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="记忆不存在")
    except InvalidMemoryFieldError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except MemoryAliasConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return MemoryResponse.from_atom(atom)


@router.post("/memories/{memory_id}/feedback", response_model=MemoryFeedbackResponse)
async def record_memory_feedback(
    memory_id: str,
    body: MemoryFeedbackRequest,
    service: MemoryApplicationService = Depends(get_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """记录用户对某条记忆的显式反馈。"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    try:
        result = await service.record_feedback(
            uid,
            target_workspace=request_access.target_workspace,
            positive=body.positive,
            source=body.source,
            access=request_access.access,
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
    request_access: RequestAccess = Depends(get_request_access),
):
    """删除记忆"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    success = await service.delete_memory(
        uid,
        target_workspace=request_access.target_workspace,
        access=request_access.access,
    )
    if not success:
        raise HTTPException(status_code=404, detail="记忆不存在或删除失败")

    return {"success": True, "id": memory_id}
