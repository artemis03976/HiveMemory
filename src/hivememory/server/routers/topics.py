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

    topics = []
    for snapshot in snapshots:
        last_turn = getattr(snapshot, "last_turn", None)
        topics.append(
            TopicSnapshotResponse(
                topic_id=snapshot.topic_id,
                topic_title=snapshot.topic_title,
                state_summary=getattr(snapshot, "state_summary", ""),
                last_turn=last_turn.model_dump() if last_turn is not None else None,
                total_tokens=getattr(snapshot, "total_tokens", 0),
                model_used=getattr(snapshot, "model_used", ""),
            )
        )

    return TopicListResponse(topics=topics)


@router.post("/topics/{topic_id}/settle", response_model=TriggerResponse)
async def settle_topic(
    topic_id: str,
    user_id: str = Depends(get_user_id),
    service: TopicApplicationService = Depends(get_topic_service),
):
    """手动结算话题"""
    result = await service.settle_topic(user_id=user_id, topic_id=topic_id)
    return TriggerResponse(**result)


@router.delete("/topics/{topic_id}", response_model=DeleteResponse)
async def delete_topic(
    topic_id: str,
    user_id: str = Depends(get_user_id),
    service: TopicApplicationService = Depends(get_topic_service),
):
    """从活跃池驱逐话题（不结算，不写记忆）"""
    result = await service.evict_topic(user_id=user_id, topic_id=topic_id)
    return DeleteResponse(**result)
