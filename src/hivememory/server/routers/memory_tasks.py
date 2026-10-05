"""Memory Task 路由 — 查询与取消 /api/v1/memory-tasks。"""

from fastapi import APIRouter, Depends, HTTPException

from hivememory.core.errors import ResourceNotFoundError
from hivememory.server.deps import RequestAccess, get_memory_task_service, get_request_access
from hivememory.server.models.memory_task import MemoryTaskListResponse, MemoryTaskResponse
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService

router = APIRouter(prefix="/memory-tasks", tags=["memory-tasks"])


@router.get("", response_model=MemoryTaskListResponse)
async def list_memory_tasks(
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
    request_access: RequestAccess = Depends(get_request_access),
) -> MemoryTaskListResponse:
    """列出本 Workspace 的记忆生成任务（观察绑定 ``task.observe``）。"""
    tasks = await service.list_memory_tasks(
        target_workspace=request_access.target_workspace,
        access=request_access.access,
    )
    return MemoryTaskListResponse(
        tasks=[MemoryTaskResponse.from_domain(memory_task) for memory_task in tasks]
    )


@router.get("/{task_id}", response_model=MemoryTaskResponse)
async def get_memory_task(
    task_id: str,
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
    request_access: RequestAccess = Depends(get_request_access),
) -> MemoryTaskResponse:
    """读取单个记忆生成任务（观察绑定 ``task.observe``）。"""
    try:
        memory_task = await service.get_memory_task(
            task_id,
            target_workspace=request_access.target_workspace,
            access=request_access.access,
        )
    except ResourceNotFoundError as exc:
        # 资源 owner 将不存在与越域统一隐藏；传输层保持同一 404 响应。
        raise HTTPException(status_code=404, detail="task not found") from exc
    return MemoryTaskResponse.from_domain(memory_task)


@router.post("/{task_id}/cancel", response_model=MemoryTaskResponse)
async def cancel_memory_task(
    task_id: str,
    service: MemoryTaskApplicationService = Depends(get_memory_task_service),
    request_access: RequestAccess = Depends(get_request_access),
) -> MemoryTaskResponse:
    """取消记忆生成任务（取消绑定 ``management.task``，观察不授予取消）。"""
    try:
        ok = await service.cancel_memory_task(
            task_id,
            target_workspace=request_access.target_workspace,
            access=request_access.access,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="task not found")
        memory_task = await service.get_memory_task(
            task_id,
            target_workspace=request_access.target_workspace,
            access=request_access.access,
        )
    except ResourceNotFoundError as exc:
        # 取消前校验和取消后投影都可能报告 not found，不暴露其他 Workspace。
        raise HTTPException(status_code=404, detail="task not found") from exc
    return MemoryTaskResponse.from_domain(memory_task, reason="user_requested")
