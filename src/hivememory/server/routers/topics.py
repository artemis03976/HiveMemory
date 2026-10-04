"""Topics 路由 — 话题管理"""

from fastapi import APIRouter, Depends, HTTPException, Query

from hivememory.patchouli.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.server.deps import (
    RequestIdentitySelection,
    get_access_gateway,
    get_identity_selection,
    get_server_principal_id,
    get_topic_service,
    request_access_for_claims,
    resolve_request_identity_claims,
)
from hivememory.server.models.topic import (
    ActiveTopicListResponse,
    ActiveTopicResponse,
    TopicDeleteResponse,
    TopicSettleResponse,
)
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.capability.topic import TopicApplicationService

router = APIRouter(tags=["topics"])

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
    service: TopicApplicationService = Depends(get_topic_service),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> ActiveTopicListResponse:
    """获取活跃话题列表（管理员话题列表，暂绑 ``management.topic``，P-9g）"""
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
    service: TopicApplicationService = Depends(get_topic_service),
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> TopicSettleResponse:
    """手动结算话题（生命周期变更绑定 ``management.topic``）"""
    claims = resolve_request_identity_claims(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    try:
        async with request_access_for_claims(
            claims, gateway=gateway, principal_id=principal_id
        ) as request_access:
            result = await service.settle_topic(
                target_workspace=request_access.target_workspace,
                topic_id=topic_id,
                access=request_access.access,
            )
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
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> TopicDeleteResponse:
    """从活跃池驱逐话题（不结算，不写记忆；绑定 ``management.topic``）"""
    claims = resolve_request_identity_claims(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    try:
        async with request_access_for_claims(
            claims, gateway=gateway, principal_id=principal_id
        ) as request_access:
            result = await service.evict_topic(
                target_workspace=request_access.target_workspace,
                topic_id=topic_id,
                access=request_access.access,
            )
    except TopicBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="话题正在处理，请稍后重试",
        ) from exc
    return TopicDeleteResponse.from_domain(result)
