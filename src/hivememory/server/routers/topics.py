"""Topics 路由 — 话题管理"""

from fastapi import APIRouter, Depends, HTTPException, Query

from hivememory.patchouli.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.server.deps import (
    RequestIdentitySelection,
    authenticate_request_access,
    get_access_gateway,
    get_identity_selection,
    get_server_principal_id,
    get_topic_service,
    release_request_access,
    resolve_request_identity_scope,
)
from hivememory.server.models.topic import (
    ActiveTopicListResponse,
    ActiveTopicResponse,
    TopicDeleteResponse,
    TopicSettleResponse,
)
from hivememory.workspace.capability.topic import TopicApplicationService

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
    gateway: ActorAuthenticationGateway = Depends(get_access_gateway),
    principal_id: str = Depends(get_server_principal_id),
) -> ActiveTopicListResponse:
    """获取活跃话题列表（读取绑定 ``resource.read``）"""
    identity_scope = resolve_request_identity_scope(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    access = await authenticate_request_access(
        identity_scope, gateway=gateway, principal_id=principal_id
    )
    try:
        snapshots = await service.list_active_topics(
            identity_scope=identity_scope, access=access
        )
    finally:
        release_request_access(access, gateway)
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
    identity_scope = resolve_request_identity_scope(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    access = await authenticate_request_access(
        identity_scope, gateway=gateway, principal_id=principal_id
    )
    try:
        result = await service.settle_topic(
            identity_scope=identity_scope, topic_id=topic_id, access=access
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
    finally:
        release_request_access(access, gateway)
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
    identity_scope = resolve_request_identity_scope(
        selection,
        explicit_user_id=user_id,
        explicit_workspace_id=workspace_id,
    )
    access = await authenticate_request_access(
        identity_scope, gateway=gateway, principal_id=principal_id
    )
    try:
        result = await service.evict_topic(
            identity_scope=identity_scope, topic_id=topic_id, access=access
        )
    except TopicBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="话题正在处理，请稍后重试",
        ) from exc
    finally:
        release_request_access(access, gateway)
    return TopicDeleteResponse.from_domain(result)
