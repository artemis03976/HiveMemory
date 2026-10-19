"""Memories 路由 — 记忆 CRUD"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from hivememory.core.errors import (
    ResourceNotFoundError,
)
from hivememory.core.models import MemoryType
from hivememory.server.deps import RequestAccess, get_request_access, get_root_memory_service
from hivememory.server.error_handlers import APIError, error_responses
from hivememory.server.models.memory import (
    MemoryCreateRequest,
    MemoryFeedbackRequest,
    MemoryFeedbackResponse,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
)
from hivememory.workspace.capability.root.memory import RootMemoryService

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["memories"])


@router.post("/memories", response_model=MemoryResponse, status_code=201)
async def create_memory(
    body: MemoryCreateRequest,
    service: RootMemoryService = Depends(get_root_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """创建新的记忆（管理用例，actor 为保留 system）"""
    atom = await service.create_memory(
        target_workspace=request_access.target_workspace,
        title=body.title,
        summary=body.summary,
        content=body.content,
        memory_type=MemoryType(body.memory_type),
        tags=body.tags,
        alias=body.alias,
        access=request_access.access,
    )
    return MemoryResponse.from_atom(atom)


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    query: str = Query(default=None, description="语义搜索查询"),
    memory_type: str = Query(default=None, description="按记忆类型过滤"),
    limit: int = Query(default=20, le=100, description="最大返回数量"),
    service: RootMemoryService = Depends(get_root_memory_service),
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
    service: RootMemoryService = Depends(get_root_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """获取单条记忆详情"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise APIError("request.invalid_memory_id")

    atom = await service.get_memory(
        uid,
        target_workspace=request_access.target_workspace,
        access=request_access.access,
    )
    return MemoryResponse.from_atom(atom)


@router.patch("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    body: MemoryUpdateRequest,
    service: RootMemoryService = Depends(get_root_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """更新记忆的可编辑字段"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise APIError("request.invalid_memory_id")

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
    return MemoryResponse.from_atom(atom)


@router.post("/memories/{memory_id}/feedback", response_model=MemoryFeedbackResponse)
async def record_memory_feedback(
    memory_id: str,
    body: MemoryFeedbackRequest,
    service: RootMemoryService = Depends(get_root_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """记录用户对某条记忆的显式反馈。"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise APIError("request.invalid_memory_id")

    result = await service.record_feedback(
        uid,
        target_workspace=request_access.target_workspace,
        positive=body.positive,
        source=body.source,
        access=request_access.access,
    )

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
    service: RootMemoryService = Depends(get_root_memory_service),
    request_access: RequestAccess = Depends(get_request_access),
):
    """删除记忆"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise APIError("request.invalid_memory_id")

    success = await service.delete_memory(
        uid,
        target_workspace=request_access.target_workspace,
        access=request_access.access,
    )
    if not success:
        raise ResourceNotFoundError()

    return {"success": True, "id": memory_id}
