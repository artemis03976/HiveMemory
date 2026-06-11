"""Memory Task 路由 — GET/DELETE /api/v1/memory-tasks"""

from fastapi import APIRouter, Depends, HTTPException

from hivememory.patchouli.runtime.memory_tasks import memory_task_to_dto
from hivememory.server.deps import get_memory_task_service
from hivememory.system.application.memory_task_service import MemoryTaskApplicationService

router = APIRouter(prefix="/memory-tasks", tags=["memory-tasks"])


@router.get("")
async def list_memory_tasks(
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
):
    tasks = await service.list_memory_tasks()
    return {"tasks": [memory_task_to_dto(memory_task).model_dump() for memory_task in tasks]}


@router.get("/{task_id}")
async def get_memory_task(
    task_id: str,
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
):
    memory_task = await service.get_memory_task(task_id)
    if memory_task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return memory_task_to_dto(memory_task).model_dump()


@router.post("/{task_id}/cancel")
async def cancel_memory_task(
    task_id: str,
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
):
    ok = await service.cancel_memory_task(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="task not found")
    memory_task = await service.get_memory_task(task_id)
    if memory_task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return memory_task_to_dto(memory_task, reason="user_requested").model_dump()
