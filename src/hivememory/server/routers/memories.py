"""Memories 路由 — 记忆 CRUD"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from hivememory.core.models import MemoryAtom
from hivememory.system import HiveMemorySystem
from hivememory.server.deps import get_system
from hivememory.server.models.memory import (
    MemoryFeedbackRequest,
    MemoryFeedbackResponse,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
    MemoryCreateRequest,
    _ALLOWED_MEMORY_TYPES,
)

router = APIRouter(tags=["memories"])


@router.post("/memories", response_model=MemoryResponse, status_code=201)
async def create_memory(body: MemoryCreateRequest, system: HiveMemorySystem = Depends(get_system)):
    """创建新的记忆"""
    from hivememory.core.models import MetaData, IndexLayer, PayloadLayer, Artifacts, MemoryType

    atom = MemoryAtom(
        meta=MetaData(source_agent_id="ui", user_id="default"),
        index=IndexLayer(
            title=body.title,
            summary=body.summary,
            tags=body.tags,
            memory_type=MemoryType(body.memory_type),
            alias=body.alias,
        ),
        payload=PayloadLayer(
            content=body.content,
            artifacts=Artifacts(),
        ),
    )
    try:
        system.patchouli.storage.upsert_memory(atom)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return MemoryResponse.from_atom(atom)


def _get_lifecycle_engine(system: HiveMemorySystem):
    runtime = getattr(system.patchouli, "runtime", None)
    if runtime is not None:
        engines = getattr(runtime, "_engines", {})
        if isinstance(engines, dict) and engines.get("lifecycle") is not None:
            return engines["lifecycle"]

    librarian = getattr(system.patchouli, "librarian_core", None)
    return getattr(librarian, "lifecycle_engine", None)


def _refresh_vitality_for_response(
    system: HiveMemorySystem,
    atoms: list[MemoryAtom],
) -> None:
    lifecycle = _get_lifecycle_engine(system)
    if lifecycle is None or not atoms:
        return
    try:
        lifecycle.refresh_vitality_batch(atoms, persist=False)
    except Exception:
        return


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    query: str = Query(default=None, description="语义搜索查询"),
    user_id: str = Query(default=None, description="按用户 ID 过滤"),
    memory_type: str = Query(default=None, description="按记忆类型过滤"),
    limit: int = Query(default=20, le=100, description="最大返回数量"),
    system: HiveMemorySystem = Depends(get_system),
):
    """检索记忆 — 支持语义搜索和过滤"""
    storage = system.patchouli.storage

    if query:
        filters = {}
        if user_id:
            filters["meta.user_id"] = user_id
        if memory_type:
            filters["index.memory_type"] = memory_type

        results = storage.search_memories(
            query_text=query,
            top_k=limit,
            filters=filters if filters else None,
        )
        atoms = [
            r["memory"]
            for r in results
            if "memory" in r and r["memory"].index.memory_type != "AGENT_PROFILE"
        ]
        _refresh_vitality_for_response(system, atoms)
        memories = [MemoryResponse.from_atom(a) for a in atoms]
    else:
        filters = {}
        if user_id:
            filters["meta.user_id"] = user_id
        if memory_type:
            filters["index.memory_type"] = memory_type

        atoms = storage.get_all_memories(
            filters=filters if filters else None,
            limit=limit,
        )
        atoms = [a for a in atoms if a.index.memory_type != "AGENT_PROFILE"]
        _refresh_vitality_for_response(system, atoms)
        memories = [MemoryResponse.from_atom(a) for a in atoms]

    return MemoryListResponse(memories=memories, total=len(memories))


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    system: HiveMemorySystem = Depends(get_system),
):
    """获取单条记忆详情"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    atom = system.patchouli.storage.get_memory(uid)
    if atom is None:
        raise HTTPException(status_code=404, detail="记忆不存在")

    _refresh_vitality_for_response(system, [atom])
    return MemoryResponse.from_atom(atom)


@router.patch("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    body: MemoryUpdateRequest,
    system: HiveMemorySystem = Depends(get_system),
):
    """更新记忆的可编辑字段"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    atom = system.patchouli.storage.get_memory(uid)
    if atom is None:
        raise HTTPException(status_code=404, detail="记忆不存在")

    from datetime import datetime, timezone
    if body.title is not None:
        atom.index.title = body.title
    if body.summary is not None:
        atom.index.summary = body.summary
    if body.content is not None:
        atom.payload.content = body.content
    if body.alias is not None:
        atom.index.alias = body.alias or None
    if body.tags is not None:
        atom.index.tags = body.tags
    if body.agent_config is not None:
        atom.payload.artifacts.agent_config = body.agent_config
    atom.meta.updated_at = datetime.now(timezone.utc)

    system.patchouli.storage.upsert_memory(atom)
    return MemoryResponse.from_atom(atom)


@router.post("/memories/{memory_id}/feedback", response_model=MemoryFeedbackResponse)
async def record_memory_feedback(
    memory_id: str,
    body: MemoryFeedbackRequest,
    system: HiveMemorySystem = Depends(get_system),
):
    """Record explicit user feedback for a memory."""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="鏃犳晥鐨勮蹇?ID 鏍煎紡")

    lifecycle = _get_lifecycle_engine(system)
    if lifecycle is None:
        raise HTTPException(status_code=503, detail="Memory lifecycle engine is unavailable")

    try:
        result = lifecycle.record_feedback(
            uid,
            positive=body.positive,
            source=body.source,
        )
    except ValueError as exc:
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
    system: HiveMemorySystem = Depends(get_system),
):
    """删除记忆"""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的记忆 ID 格式")

    success = system.patchouli.storage.delete_memory(uid)
    if not success:
        raise HTTPException(status_code=404, detail="记忆不存在或删除失败")

    return {"success": True, "id": memory_id}
