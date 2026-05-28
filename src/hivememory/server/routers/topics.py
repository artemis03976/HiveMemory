"""Topics 路由 — 话题管理"""

from fastapi import APIRouter, Depends

from hivememory.server.deps import get_topic_service, get_user_id
from hivememory.server.models.topic import (
    DeleteResponse,
    TopicListResponse,
    TopicSnapshotResponse,
    TriggerResponse,
)
from hivememory.system.application.topic_service import TopicApplicationService

router = APIRouter(tags=["topics"])


@router.get("/topics", response_model=TopicListResponse)
async def list_topics(
    user_id: str = Depends(get_user_id),
    service: TopicApplicationService = Depends(get_topic_service),
):
    """获取活跃话题列表"""
    snapshots = await service.list_active_topics(user_id=user_id)

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
    service: TopicApplicationService = Depends(get_topic_service),
):
    """手动归档话题"""
    result = await service.archive_topic(topic_id=topic_id)
    return TriggerResponse(**result)


@router.delete("/topics/{topic_id}", response_model=DeleteResponse)
async def delete_topic(
    topic_id: str,
    service: TopicApplicationService = Depends(get_topic_service),
):
    """从活跃池驱逐话题（不归档，不写长期记忆）"""
    result = await service.evict_topic(topic_id=topic_id)
    return DeleteResponse(**result)
