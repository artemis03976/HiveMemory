"""Memory Job 路由 — GET/DELETE /api/v1/memory-jobs"""

from fastapi import APIRouter, Depends, HTTPException

from hivememory.server.deps import get_system

router = APIRouter(prefix="/memory-jobs", tags=["memory-jobs"])


def _job_to_dict(job) -> dict:
    return {
        "job_id": job.job_id,
        "topic_id": job.topic_id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "tasks": [
            {
                "pending_alias": t.pending_alias,
                "source_verb": t.source_verb,
                "status": t.status.value,
                "canonical_alias": t.canonical_alias,
                "error": t.error,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "finished_at": t.finished_at.isoformat() if t.finished_at else None,
            }
            for t in job.tasks
        ],
    }


@router.get("")
async def list_memory_jobs(system=Depends(get_system)):
    jobs = system._patchouli.service.list_memory_jobs()
    return {"jobs": [_job_to_dict(j) for j in jobs]}


@router.get("/{job_id}")
async def get_memory_job(job_id: str, system=Depends(get_system)):
    job = system._patchouli.service.get_memory_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_to_dict(job)


@router.delete("/{job_id}/cancel")
async def cancel_memory_job(job_id: str, system=Depends(get_system)):
    ok = system._patchouli.service.cancel_memory_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, "cancelled": True}
