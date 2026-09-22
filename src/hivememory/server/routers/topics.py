"""Topics 路由 — 话题管理"""

from fastapi import APIRouter, Depends, HTTPException, Query

from hivememory.patchouli.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.server.deps import (
    RequestIdentitySelection,
    get_identity_selection,
    get_topic_service,
    resolve_request_identity_scope,
)
from hivememory.server.models.topic import (
    ActiveTopicListResponse,
    ActiveTopicResponse,
    TopicDeleteResponse,
    TopicSettleResponse,
)
from hivememory.system.application.topic_service import TopicApplicationService

router = APIRouter(tags=["topics"])

# 兼容说明：Topic 旧入口经 query 传递 user_id；现统一收敛到
# resolve_request_identity_scope 的同一解析规则（query 与 header 冲突显式拒绝）。
TopicUserIdQuery = Query(default=None, description="用户 ID（兼容入口，与 header 冲突时拒绝）")
TopicWorkspaceIdQuery = Query(
    default=None, description="Workspace ID（缺省回退公共默认 Workspace）"
)


@router.get("/topics", response_model=ActiveTopicListResponse)
async def list_topics(
    user_id: str | None = TopicUserIdQuery,
    workspace_id: str | None = TopicWorkspaceIdQuery,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: TopicApplicationService = Depends(get_topic_service),
) -> ActiveTopicListResponse:
    """获取活跃话题列表"""
    identity_scope = resolve_request_identity_scope(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    snapshots = await service.list_active_topics(identity_scope=identity_scope)

    return ActiveTopicListResponse(
        topics=[ActiveTopicResponse.from_domain(snapshot) for snapshot in snapshots]
    )


@router.post("/topics/{topic_id}/settle", response_model=TopicSettleResponse)
async def settle_topic(
    topic_id: str,
    user_id: str | None = TopicUserIdQuery,
    workspace_id: str | None = TopicWorkspaceIdQuery,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: TopicApplicationService = Depends(get_topic_service),
) -> TopicSettleResponse:
    """手动结算话题"""
    identity_scope = resolve_request_identity_scope(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    try:
        result = await service.settle_topic(identity_scope=identity_scope, topic_id=topic_id)
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
    user_id: str | None = TopicUserIdQuery,
    workspace_id: str | None = TopicWorkspaceIdQuery,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: TopicApplicationService = Depends(get_topic_service),
) -> TopicDeleteResponse:
    """从活跃池驱逐话题（不结算，不写记忆）"""
    identity_scope = resolve_request_identity_scope(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    try:
        result = await service.evict_topic(identity_scope=identity_scope, topic_id=topic_id)
    except TopicBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="话题正在处理，请稍后重试",
        ) from exc
    return TopicDeleteResponse.from_domain(result)
