"""Memory Task 路由 — 查询与取消 /api/v1/memory-tasks。"""

from fastapi import APIRouter, Depends

from hivememory.core.errors import ResourceNotFoundError
from hivememory.server.deps import RequestAccess, get_request_access, get_root_memory_task_service
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.memory_task import MemoryTaskListResponse, MemoryTaskResponse
from hivememory.workspace.capability.root.memory_tasks import RootMemoryTaskService

router = APIRouter(
    responses=error_responses(400, 403, 404, 409, 422, 500, 503),
    prefix="/memory-tasks",
    tags=["memory-tasks"],
)


@router.get("", response_model=MemoryTaskListResponse)
async def list_memory_tasks(
    service: RootMemoryTaskService = Depends(get_root_memory_task_service),
    request_access: RequestAccess = Depends(get_request_access),
) -> MemoryTaskListResponse:
    """经 root API 列出本 Workspace 的记忆生成任务。"""
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
    service: RootMemoryTaskService = Depends(get_root_memory_task_service),
    request_access: RequestAccess = Depends(get_request_access),
) -> MemoryTaskResponse:
    """经 root API 读取本 Workspace 的单个记忆生成任务。"""
    memory_task = await service.get_memory_task(
        task_id,
        target_workspace=request_access.target_workspace,
        access=request_access.access,
    )
    return MemoryTaskResponse.from_domain(memory_task)


@router.post("/{task_id}/cancel", response_model=MemoryTaskResponse)
async def cancel_memory_task(
    task_id: str,
    service: RootMemoryTaskService = Depends(get_root_memory_task_service),
    request_access: RequestAccess = Depends(get_request_access),
) -> MemoryTaskResponse:
    """经 root API 取消本 Workspace 的记忆生成任务。"""
    ok = await service.cancel_memory_task(
        task_id,
        target_workspace=request_access.target_workspace,
        access=request_access.access,
    )
    if not ok:
        raise ResourceNotFoundError()
    memory_task = await service.get_memory_task(
        task_id,
        target_workspace=request_access.target_workspace,
        access=request_access.access,
    )
    return MemoryTaskResponse.from_domain(memory_task, reason="user_requested")
