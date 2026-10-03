"""Memory Task 路由 — GET/DELETE /api/v1/memory-tasks"""

from fastapi import APIRouter, Depends, HTTPException

from hivememory.core.access import WorkspaceAccessContext
from hivememory.server.deps import (
    get_memory_task_service,
    get_request_access_context,
)
from hivememory.server.models.memory_task import MemoryTaskListResponse, MemoryTaskResponse
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService

router = APIRouter(prefix="/memory-tasks", tags=["memory-tasks"])


@router.get("", response_model=MemoryTaskListResponse)
async def list_memory_tasks(
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
    access: WorkspaceAccessContext = Depends(get_request_access_context),
) -> MemoryTaskListResponse:
    """列出本 Workspace 的记忆生成任务（观察绑定 ``task.observe``）。"""
    tasks = await service.list_memory_tasks(access=access)
    return MemoryTaskListResponse(
        tasks=[MemoryTaskResponse.from_domain(memory_task) for memory_task in tasks]
    )


@router.get("/{task_id}", response_model=MemoryTaskResponse)
async def get_memory_task(
    task_id: str,
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
    access: WorkspaceAccessContext = Depends(get_request_access_context),
) -> MemoryTaskResponse:
    """读取单个记忆生成任务（观察绑定 ``task.observe``）。"""
    memory_task = await service.get_memory_task(task_id, access=access)
    if memory_task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return MemoryTaskResponse.from_domain(memory_task)


@router.post("/{task_id}/cancel", response_model=MemoryTaskResponse)
async def cancel_memory_task(
    task_id: str,
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
    access: WorkspaceAccessContext = Depends(get_request_access_context),
) -> MemoryTaskResponse:
    """取消记忆生成任务（取消绑定 ``management.task``，观察不授予取消）。"""
    ok = await service.cancel_memory_task(task_id, access=access)
    if not ok:
        raise HTTPException(status_code=404, detail="task not found")
    memory_task = await service.get_memory_task(task_id, access=access)
    if memory_task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return MemoryTaskResponse.from_domain(memory_task, reason="user_requested")
