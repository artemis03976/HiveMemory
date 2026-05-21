"""Memories 路由 — 记忆 CRUD"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from hivememory.system import HiveMemorySystem
from hivememory.server.deps import get_system
from hivememory.server.models.memory import MemoryResponse, MemoryListResponse, MemoryUpdateRequest

router = APIRouter(tags=["memories"])


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
        memories = [
            MemoryResponse.from_atom(r["memory"])
            for r in results
            if "memory" in r and r["memory"].index.memory_type != "AGENT_PROFILE"
        ]
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
        memories = [MemoryResponse.from_atom(a) for a in atoms if a.index.memory_type != "AGENT_PROFILE"]

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
