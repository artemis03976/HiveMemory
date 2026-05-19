"""Topics 路由 — 话题管理"""

from fastapi import APIRouter, Depends

from hivememory.core.models import Identity
from hivememory.server.deps import get_system, get_user_id
from hivememory.server.models.topic import (
    DeleteResponse,
    TopicListResponse,
    TopicSnapshotResponse,
    TriggerResponse,
)
from hivememory.system import HiveMemorySystem

router = APIRouter(tags=["topics"])


@router.get("/topics", response_model=TopicListResponse)
async def list_topics(
    user_id: str = Depends(get_user_id),
    system: HiveMemorySystem = Depends(get_system),
):
    """获取活跃话题列表"""
    identity = Identity(user_id=user_id)
    snapshots = system.runtime.librarian_core.get_active_topics_snapshots(identity)

    topics = [
        TopicSnapshotResponse(
            topic_id=s.topic_id,
            title=s.title,
            state_summary=getattr(s, "state_summary", ""),
            last_turn=getattr(s, "last_turn", None),
            total_tokens=getattr(s, "total_tokens", 0),
        )
        for s in snapshots
    ]

    return TopicListResponse(topics=topics)


@router.post("/topics/{topic_id}/archive", response_model=TriggerResponse)
async def archive_topic(
    topic_id: str,
    system: HiveMemorySystem = Depends(get_system),
):
    """手动归档话题"""
    result = await system.manual_archive_topic(topic_id=topic_id)
    return TriggerResponse(**result)


@router.delete("/topics/{topic_id}", response_model=DeleteResponse)
async def delete_topic(
    topic_id: str,
    system: HiveMemorySystem = Depends(get_system),
):
    """从活跃池驱逐话题（不归档，不写长期记忆）"""
    buf = system.runtime.librarian_core.perception_layer.buffer_manager.pop_buffer(topic_id)
    if buf is None:
        return DeleteResponse(success=False, message="话题不存在或已被驱逐")
    return DeleteResponse(success=True, message=f"话题 {topic_id} 已删除")
