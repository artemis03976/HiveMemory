"""Topics 路由 — 话题管理"""

from fastapi import APIRouter, Depends, Query

from hivememory.server.deps import (
    RequestIdentitySelection,
    get_access_gateway,
    get_identity_selection,
    get_root_topic_service,
    get_server_principal_id,
    request_access_for_claims,
    resolve_request_identity_claims,
)
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.topic import (
    ActiveTopicListResponse,
    ActiveTopicResponse,
    TopicDeleteResponse,
    TopicSettleResponse,
)
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.capability.root.topic import RootTopicService

router = APIRouter(responses=error_responses(400, 403, 404, 409, 422, 500, 503), tags=["topics"])

# 兼容说明：Topic 旧入口经 query 传递 user_id；现统一收敛到
# resolve_request_identity_claims 的同一解析规则（query 与 header 冲突显式拒绝）。
TopicUserIdQuery = Query(default=None, description="用户 ID（兼容入口，与 header 冲突时拒绝）")
TopicWorkspaceIdQuery = Query(
    default=None, description="Workspace ID（缺省回退公共默认 Workspace）"
)


@router.get("/topics", response_model=ActiveTopicListResponse)
async def list_topics(
    user_id: str | None = TopicUserIdQuery,
    workspace_id: str | None = TopicWorkspaceIdQuery,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: RootTopicService = Depends(get_root_topic_service),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> ActiveTopicListResponse:
    """经 root API 获取目标 Workspace 的活跃话题列表。"""
    claims = resolve_request_identity_claims(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    async with request_access_for_claims(
        claims, gateway=gateway, principal_id=principal_id
    ) as request_access:
        snapshots = await service.list_active_topics(
            target_workspace=request_access.target_workspace,
            access=request_access.access,
        )
    return ActiveTopicListResponse(
        topics=[ActiveTopicResponse.from_domain(snapshot) for snapshot in snapshots]
    )


@router.post("/topics/{topic_id}/settle", response_model=TopicSettleResponse)
async def settle_topic(
    topic_id: str,
    user_id: str | None = TopicUserIdQuery,
    workspace_id: str | None = TopicWorkspaceIdQuery,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: RootTopicService = Depends(get_root_topic_service),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> TopicSettleResponse:
    """经 root API 手动结算话题。"""
    claims = resolve_request_identity_claims(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    async with request_access_for_claims(
        claims, gateway=gateway, principal_id=principal_id
    ) as request_access:
        result = await service.settle_topic(
            target_workspace=request_access.target_workspace,
            topic_id=topic_id,
            access=request_access.access,
        )
    return TopicSettleResponse.from_domain(result)


@router.delete("/topics/{topic_id}", response_model=TopicDeleteResponse)
async def delete_topic(
    topic_id: str,
    user_id: str | None = TopicUserIdQuery,
    workspace_id: str | None = TopicWorkspaceIdQuery,
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: RootTopicService = Depends(get_root_topic_service),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> TopicDeleteResponse:
    """经 root API 从活跃池驱逐话题，不结算或写入记忆。"""
    claims = resolve_request_identity_claims(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    async with request_access_for_claims(
        claims, gateway=gateway, principal_id=principal_id
    ) as request_access:
        result = await service.evict_topic(
            target_workspace=request_access.target_workspace,
            topic_id=topic_id,
            access=request_access.access,
        )
    return TopicDeleteResponse.from_domain(result)
