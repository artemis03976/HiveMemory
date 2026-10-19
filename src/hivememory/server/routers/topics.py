"""Topics 路由 — 话题管理"""

from fastapi import APIRouter, Depends

from hivememory.server.deps import RequestAccess, get_request_access, get_root_topic_service
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.topic import (
    ActiveTopicListResponse,
    ActiveTopicResponse,
    TopicDeleteResponse,
    TopicSettleResponse,
)
from hivememory.workspace.capability.root.topic import RootTopicService

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["topics"])


@router.get("/topics", response_model=ActiveTopicListResponse)
async def list_topics(
    access: RequestAccess = Depends(get_request_access),
    service: RootTopicService = Depends(get_root_topic_service),
) -> ActiveTopicListResponse:
    """经 root API 获取目标 Workspace 的活跃话题列表。"""
    snapshots = await service.list_active_topics(
        target_workspace=access.target_workspace,
        access=access.access,
    )
    return ActiveTopicListResponse(
        topics=[ActiveTopicResponse.from_domain(snapshot) for snapshot in snapshots]
    )


@router.post("/topics/{topic_id}/settle", response_model=TopicSettleResponse)
async def settle_topic(
    topic_id: str,
    access: RequestAccess = Depends(get_request_access),
    service: RootTopicService = Depends(get_root_topic_service),
) -> TopicSettleResponse:
    """经 root API 手动结算话题。"""
    result = await service.settle_topic(
        target_workspace=access.target_workspace,
        topic_id=topic_id,
        access=access.access,
    )
    return TopicSettleResponse.from_domain(result)


@router.delete("/topics/{topic_id}", response_model=TopicDeleteResponse)
async def delete_topic(
    topic_id: str,
    access: RequestAccess = Depends(get_request_access),
    service: RootTopicService = Depends(get_root_topic_service),
) -> TopicDeleteResponse:
    """经 root API 从活跃池驱逐话题，不结算或写入记忆。"""
    result = await service.evict_topic(
        target_workspace=access.target_workspace,
        topic_id=topic_id,
        access=access.access,
    )
    return TopicDeleteResponse.from_domain(result)
