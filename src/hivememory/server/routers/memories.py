"""Memories 路由 — 记忆 CRUD"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from hivememory.server.deps import get_memory_service
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
):
    """创建新的记忆"""
    try:
        atom = service.create_memory(
            title=body.title,
            summary=body.summary,
            content=body.content,
            memory_type=body.memory_type,
            tags=body.tags,
            alias=body.alias,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return MemoryResponse.from_atom(atom)


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    query: str = Query(default=None, description="语义搜索查询"),
    user_id: str = Query(default=None, description="按用户 ID 过滤"),
    memory_type: str = Query(default=None, description="按记忆类型过滤"),
    limit: int = Query(default=20, le=100, description="最大返回数量"),
    service: MemoryApplicationService = Depends(get_memory_service),
):
    """检索记忆 — 支持语义搜索和过滤"""
    atoms = service.list_memories(
        query=query,
        user_id=user_id,
        memory_type=memory_type,
        limit=limit,
    )
    memories = [MemoryResponse.from_atom(a) for a in atoms]
    return MemoryListResponse(memories=memories, total=len(memories))


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    service: MemoryApplicationService = Depends(get_memory_service),
):
    """获取单条记忆详情"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    try:
        atom = service.get_memory(uid)
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return MemoryResponse.from_atom(atom)


@router.patch("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    body: MemoryUpdateRequest,
    service: MemoryApplicationService = Depends(get_memory_service),
):
    """更新记忆的可编辑字段"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    try:
        atom = service.update_memory(
            uid,
            title=body.title,
            summary=body.summary,
            content=body.content,
            alias=body.alias,
            tags=body.tags,
            agent_config=body.agent_config,
        )
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return MemoryResponse.from_atom(atom)


@router.post("/memories/{memory_id}/feedback", response_model=MemoryFeedbackResponse)
async def record_memory_feedback(
    memory_id: str,
    body: MemoryFeedbackRequest,
    service: MemoryApplicationService = Depends(get_memory_service),
):
    """Record explicit user feedback for a memory."""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    try:
        result = service.record_feedback(
            uid,
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
):
    """删除记忆"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    success = service.delete_memory(uid)
    if not success:
        raise HTTPException(status_code=404, detail="记忆不存在或删除失败")

    return {"success": True, "id": memory_id}
