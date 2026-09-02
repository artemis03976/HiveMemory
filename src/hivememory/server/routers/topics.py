"""Topics 路由 — 话题管理"""

from fastapi import APIRouter, Depends, HTTPException

from hivememory.patchouli.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.server.deps import get_topic_service, get_user_id
from hivememory.server.models.topic import (
    ActiveTopicListResponse,
    ActiveTopicResponse,
    TopicDeleteResponse,
    TopicSettleResponse,
)
from hivememory.system.application.topic_service import TopicApplicationService

router = APIRouter(tags=["topics"])


@router.get("/topics", response_model=ActiveTopicListResponse)
async def list_topics(
    user_id: str = Depends(get_user_id),
    service: TopicApplicationService = Depends(get_topic_service),
) -> ActiveTopicListResponse:
    """获取活跃话题列表"""
    snapshots = await service.list_active_topics(user_id=user_id)

    return ActiveTopicListResponse(
        topics=[ActiveTopicResponse.from_domain(snapshot) for snapshot in snapshots]
    )


@router.post("/topics/{topic_id}/settle", response_model=TopicSettleResponse)
async def settle_topic(
    topic_id: str,
    user_id: str = Depends(get_user_id),
    service: TopicApplicationService = Depends(get_topic_service),
) -> TopicSettleResponse:
    """手动结算话题"""
    try:
        result = await service.settle_topic(user_id=user_id, topic_id=topic_id)
    except TopicSettleAdmissionError as exc:
        raise HTTPException(
            status_code=503,
            detail="结算材料暂未被生成队列接纳，话题内容已保留，可重试",
        ) from exc
    except TopicBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="话题正在处理，请稍后重试",
        ) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="话题不存在") from exc
    return TopicSettleResponse.from_domain(result)


@router.delete("/topics/{topic_id}", response_model=TopicDeleteResponse)
async def delete_topic(
    topic_id: str,
    user_id: str = Depends(get_user_id),
    service: TopicApplicationService = Depends(get_topic_service),
) -> TopicDeleteResponse:
    """从活跃池驱逐话题（不结算，不写记忆）"""
    try:
        result = await service.evict_topic(user_id=user_id, topic_id=topic_id)
    except TopicBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="话题正在处理，请稍后重试",
        ) from exc
    return TopicDeleteResponse.from_domain(result)
