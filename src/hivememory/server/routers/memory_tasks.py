"""Memory Task 路由 — GET/DELETE /api/v1/memory-tasks"""

from fastapi import APIRouter, Depends, HTTPException

from hivememory.server.deps import get_system

router = APIRouter(prefix="/memory-tasks", tags=["memory-tasks"])


def _task_to_dict(memory_task) -> dict:
    return {
        "task_id": memory_task.task_id,
        "topic_id": memory_task.topic_id,
        "label": memory_task.label,
        "source": memory_task.source.value,
        "pending_alias": memory_task.pending_alias,
        "status": memory_task.status.value,
        "canonical_alias": memory_task.canonical_alias,
        "error": memory_task.error,
        "created_at": memory_task.created_at.isoformat(),
        "started_at": memory_task.started_at.isoformat() if memory_task.started_at else None,
        "finished_at": memory_task.finished_at.isoformat() if memory_task.finished_at else None,
    }


def _cancel_response(memory_task) -> dict:
    payload = _task_to_dict(memory_task)
    payload.update(
        {
            "cancelled": memory_task.status.value == "cancelled",
            "cancel_requested": memory_task.cancelled,
            "reason": "user_requested" if memory_task.cancelled else None,
        }
    )
    return payload


@router.get("")
async def list_memory_tasks(system=Depends(get_system)):
    tasks = system._patchouli.service.list_memory_tasks()
    return {"tasks": [_task_to_dict(memory_task) for memory_task in tasks]}


@router.get("/{task_id}")
async def get_memory_task(task_id: str, system=Depends(get_system)):
    memory_task = system._patchouli.service.get_memory_task(task_id)
    if memory_task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return _task_to_dict(memory_task)


@router.post("/{task_id}/cancel")
async def cancel_memory_task(task_id: str, system=Depends(get_system)):
    service = system._patchouli.service
    ok = service.cancel_memory_task(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="task not found")
    memory_task = service.get_memory_task(task_id)
    if memory_task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return _cancel_response(memory_task)
