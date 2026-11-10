"""接入登记与 Workspace 访问记录的管理路由。

两组接口分开维护，各守自己的所有权：

- 接入登记属于 System：与提供商、模型接口一样直接读写注册表，只由管理凭据
  保护，不经认证网关；只有外部来源可以编辑；
- 访问记录属于 workspace：经 root API，以请求的工作 actor 做 root 授权，目标
  是它驻留的 Workspace。

处理函数都是 ``async def``：编辑在事件循环中执行，与认证、授权的查询不会
并发交错。
"""

from fastapi import APIRouter, Depends, Response

from hivememory.core.access import WorkingActor
from hivememory.server.deps import (
    get_principal_registry,
    get_request_actor,
    get_root_access_service,
)
from hivememory.server.error_handlers import error_responses
from hivememory.server.models.access import (
    ActorRecordRequest,
    ActorRecordResponse,
    PrincipalCreateRequest,
    PrincipalCredentialResponse,
    PrincipalResponse,
    PrincipalSettings,
)
from hivememory.system.access import SystemActorAccessRegistry
from hivememory.workspace.capability.root.access import RootAccessService

router = APIRouter(
    prefix="/access",
    responses=error_responses(400, 403, 404, 409, 422, 500, 503),
    tags=["access"],
)


def _agent_set(agent_ids: list[str] | None) -> frozenset[str] | None:
    return frozenset(agent_ids) if agent_ids is not None else None


# ---- 接入登记（System） ----


@router.get("/principals", response_model=list[PrincipalResponse])
async def list_principals(
    registry: SystemActorAccessRegistry = Depends(get_principal_registry),
) -> list[PrincipalResponse]:
    """列出全部接入登记；凭据只给来源与打码值。"""
    return [PrincipalResponse.from_entry(entry) for entry in registry.entries()]


@router.post("/principals", response_model=PrincipalCredentialResponse, status_code=201)
async def register_principal(
    body: PrincipalCreateRequest,
    registry: SystemActorAccessRegistry = Depends(get_principal_registry),
) -> PrincipalCredentialResponse:
    """登记一个外部来源并生成凭据；凭据明文只在这次响应中返回。"""
    entry, credential = registry.register(
        body.principal_id.strip(),
        adapters=frozenset(body.adapters),
        allowed_agent_ids=_agent_set(body.allowed_agent_ids),
        enabled=body.enabled,
    )
    return PrincipalCredentialResponse(
        principal=PrincipalResponse.from_entry(entry), credential=credential
    )


@router.put("/principals/{principal_id}", response_model=PrincipalResponse)
async def update_principal(
    principal_id: str,
    body: PrincipalSettings,
    registry: SystemActorAccessRegistry = Depends(get_principal_registry),
) -> PrincipalResponse:
    """修改外部来源的启用状态、adapter 与可声明的 Agent。"""
    entry = registry.update(
        principal_id,
        adapters=frozenset(body.adapters),
        allowed_agent_ids=_agent_set(body.allowed_agent_ids),
        enabled=body.enabled,
    )
    return PrincipalResponse.from_entry(entry)


@router.post("/principals/{principal_id}/credential", response_model=PrincipalCredentialResponse)
async def rotate_principal_credential(
    principal_id: str,
    registry: SystemActorAccessRegistry = Depends(get_principal_registry),
) -> PrincipalCredentialResponse:
    """为外部来源生成新凭据，旧凭据随即失效。"""
    entry, credential = registry.rotate_credential(principal_id)
    return PrincipalCredentialResponse(
        principal=PrincipalResponse.from_entry(entry), credential=credential
    )


@router.delete("/principals/{principal_id}", status_code=204)
async def delete_principal(
    principal_id: str,
    registry: SystemActorAccessRegistry = Depends(get_principal_registry),
) -> Response:
    """删除外部来源与它的生成凭据。"""
    registry.delete(principal_id)
    return Response(status_code=204)


# ---- Workspace 访问记录（root API） ----


@router.get("/actors", response_model=list[ActorRecordResponse])
async def list_actor_records(
    actor: WorkingActor = Depends(get_request_actor),
    service: RootAccessService = Depends(get_root_access_service),
) -> list[ActorRecordResponse]:
    """列出当前 Workspace 的访问记录，不含保留的 system 记录。"""
    records = service.list_records(target_workspace=actor.workspace, actor=actor)
    return [ActorRecordResponse.from_record(record) for record in records]


@router.put("/actors/user", response_model=ActorRecordResponse)
async def put_user_record(
    body: ActorRecordRequest,
    actor: WorkingActor = Depends(get_request_actor),
    service: RootAccessService = Depends(get_root_access_service),
) -> ActorRecordResponse:
    """新增或修改用户级记录：覆盖没有精确记录的所有具体 Agent。"""
    return _put_record(None, body, actor, service)


@router.delete("/actors/user", status_code=204)
async def delete_user_record(
    actor: WorkingActor = Depends(get_request_actor),
    service: RootAccessService = Depends(get_root_access_service),
) -> Response:
    """删除用户级记录；之后没有精确记录的 Agent 不能进入 Workspace。"""
    service.delete_record(None, target_workspace=actor.workspace, actor=actor)
    return Response(status_code=204)


@router.put("/actors/agents/{agent_id}", response_model=ActorRecordResponse)
async def put_agent_record(
    agent_id: str,
    body: ActorRecordRequest,
    actor: WorkingActor = Depends(get_request_actor),
    service: RootAccessService = Depends(get_root_access_service),
) -> ActorRecordResponse:
    """新增或修改某个 Agent 的精确记录；精确记录优先于用户级记录。"""
    return _put_record(agent_id, body, actor, service)


@router.delete("/actors/agents/{agent_id}", status_code=204)
async def delete_agent_record(
    agent_id: str,
    actor: WorkingActor = Depends(get_request_actor),
    service: RootAccessService = Depends(get_root_access_service),
) -> Response:
    """删除某个 Agent 的精确记录；之后它回落到用户级记录。"""
    service.delete_record(agent_id, target_workspace=actor.workspace, actor=actor)
    return Response(status_code=204)


def _put_record(
    agent_id: str | None,
    body: ActorRecordRequest,
    actor: WorkingActor,
    service: RootAccessService,
) -> ActorRecordResponse:
    record = service.put_record(
        agent_id,
        enabled=body.enabled,
        allowed_operations=frozenset(body.allowed_operations),
        target_workspace=actor.workspace,
        actor=actor,
    )
    return ActorRecordResponse.from_record(record)
