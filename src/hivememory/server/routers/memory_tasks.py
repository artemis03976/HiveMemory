"""Memory Task 路由 — GET/DELETE /api/v1/memory-tasks"""

from fastapi import APIRouter, Depends, HTTPException

from hivememory.server.deps import get_memory_task_service
from hivememory.server.models.memory_task import MemoryTaskListResponse, MemoryTaskResponse
from hivememory.system.application.memory_task_service import MemoryTaskApplicationService

router = APIRouter(prefix="/memory-tasks", tags=["memory-tasks"])


@router.get("", response_model=MemoryTaskListResponse)
async def list_memory_tasks(
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
) -> MemoryTaskListResponse:
    tasks = await service.list_memory_tasks()
    return MemoryTaskListResponse(
        tasks=[MemoryTaskResponse.from_domain(memory_task) for memory_task in tasks]
    )


@router.get("/{task_id}", response_model=MemoryTaskResponse)
async def get_memory_task(
    task_id: str,
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
) -> MemoryTaskResponse:
    memory_task = await service.get_memory_task(task_id)
    if memory_task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return MemoryTaskResponse.from_domain(memory_task)


@router.post("/{task_id}/cancel", response_model=MemoryTaskResponse)
async def cancel_memory_task(
    task_id: str,
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
) -> MemoryTaskResponse:
    ok = await service.cancel_memory_task(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="task not found")
    memory_task = await service.get_memory_task(task_id)
    if memory_task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return MemoryTaskResponse.from_domain(memory_task, reason="user_requested")
