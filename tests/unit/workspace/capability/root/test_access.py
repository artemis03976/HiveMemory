"""root 访问记录 API 的测试：入口 root 授权、只读写目标 Workspace 坐标、
保留的 system 记录不可编辑，以及编辑在下一次 actor 操作授权时生效。"""

from pathlib import Path

import pytest
import yaml

from hivememory.config.access import WorkspaceActorsFile, file_digest
from hivememory.core.access import CallerPrincipal, WorkingActor, WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AccessRegistrationConflictError, OperationDeniedError
from hivememory.core.models import ActorIdentity
from hivememory.system.access import (
    SystemActorAccessEntry,
    SystemActorAccessRegistry,
    SystemPrincipalAuthenticator,
)
from hivememory.workspace.authentication import ActorAuthenticationGateway, WorkspaceAuthenticator
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.root.access import RootAccessService
from hivememory.workspace.registry import WorkspaceActorAccessRegistry
from tests.helpers.workspace import make_workspace_identity

MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")

_ACTORS_YAML = """\
workspace_actors:
  - owner_user_id: "u1"
    workspace_id: "main_workspace"
    user_id: "u1"
    allowed_operations: ["memory.read", "memory.search"]
  - owner_user_id: "u1"
    workspace_id: "main_workspace"
    user_id: "u1"
    agent_id: "system"
  - owner_user_id: "u1"
    workspace_id: "other_workspace"
    user_id: "u1"
    agent_id: "a9"
    allowed_operations: ["memory.read"]
"""


class _Composition:
    """以登记文件装载的访问注册表，连同共用它的认证网关、授权者与 root 服务。"""

    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "workspace_actors.yaml"
        self.path.write_text(_ACTORS_YAML, encoding="utf-8")
        registry = WorkspaceActorAccessRegistry.from_config(
            WorkspaceActorsFile.model_validate(yaml.safe_load(self.path.read_text("utf-8"))),
            file_path=self.path,
            file_digest=file_digest(self.path),
        )
        self.authorizer = WorkspaceOperationAuthorizer(registry)
        self.service = RootAccessService(registry, operation_authorizer=self.authorizer)
        self._gateway = ActorAuthenticationGateway(
            principals=SystemPrincipalAuthenticator(
                SystemActorAccessRegistry(
                    [
                        SystemActorAccessEntry(
                            principal_id="local-process:test", allow_system_actor=True
                        )
                    ]
                )
            ),
            authenticator=WorkspaceAuthenticator(registry),
        )

    async def actor(self, agent_id: str) -> WorkingActor:
        return await self._gateway.authenticate(
            adapter="local",
            principal=CallerPrincipal("local-process:test"),
            credential=None,
            actor=ActorIdentity(user_id="u1", agent_id=agent_id),
            workspace=MAIN,
        )


@pytest.mark.asyncio
async def test_record_edits_take_effect_on_the_next_authorization(tmp_path):
    """root 写入的精确记录在下一次 actor 操作授权时生效，替换后同样立即生效。"""
    composition = _Composition(tmp_path)
    system = await composition.actor(SYSTEM_AGENT_ID)
    composition.service.put_record(
        "a1",
        enabled=True,
        allowed_operations=frozenset({WorkspaceOperation.MEMORY_READ}),
        target_workspace=MAIN,
        actor=system,
    )
    agent = await composition.actor("a1")

    with pytest.raises(OperationDeniedError):
        composition.authorizer.authorize_agent(agent, WorkspaceOperation.MEMORY_WRITE, MAIN)

    composition.service.put_record(
        "a1",
        enabled=True,
        allowed_operations=frozenset({WorkspaceOperation.MEMORY_WRITE}),
        target_workspace=MAIN,
        actor=system,
    )
    composition.authorizer.authorize_agent(agent, WorkspaceOperation.MEMORY_WRITE, MAIN)


@pytest.mark.asyncio
async def test_listing_shows_target_workspace_records_without_system(tmp_path):
    """列表只含目标 Workspace 坐标下的记录，不含保留的 system 记录。"""
    composition = _Composition(tmp_path)
    system = await composition.actor(SYSTEM_AGENT_ID)

    records = composition.service.list_records(target_workspace=MAIN, actor=system)

    assert [(record.workspace_id, record.agent_id) for record in records] == [
        ("main_workspace", None)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("edit", ["put", "delete"])
async def test_system_record_cannot_be_edited(tmp_path, edit):
    """修改或删除 system 记录以 protected 拒绝：root 授权以它准入 system，文件保持原样。"""
    composition = _Composition(tmp_path)
    system = await composition.actor(SYSTEM_AGENT_ID)
    before = composition.path.read_bytes()

    with pytest.raises(AccessRegistrationConflictError) as exc_info:
        if edit == "put":
            composition.service.put_record(
                SYSTEM_AGENT_ID,
                enabled=False,
                allowed_operations=frozenset(),
                target_workspace=MAIN,
                actor=system,
            )
        else:
            composition.service.delete_record(SYSTEM_AGENT_ID, target_workspace=MAIN, actor=system)

    assert exc_info.value.details["reason"] == "protected"
    assert composition.path.read_bytes() == before
    # system 仍能执行 root 操作
    composition.service.list_records(target_workspace=MAIN, actor=system)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["list", "put", "delete"])
async def test_every_entry_requires_root(tmp_path, method):
    """非 system 的工作 actor 调用任一入口都以 root_required 拒绝，记录不变。"""
    composition = _Composition(tmp_path)
    agent = await composition.actor("a1")
    before = composition.path.read_bytes()

    with pytest.raises(OperationDeniedError) as exc_info:
        if method == "list":
            composition.service.list_records(target_workspace=MAIN, actor=agent)
        elif method == "put":
            composition.service.put_record(
                "a1",
                enabled=True,
                allowed_operations=frozenset(WorkspaceOperation),
                target_workspace=MAIN,
                actor=agent,
            )
        else:
            composition.service.delete_record(None, target_workspace=MAIN, actor=agent)

    assert exc_info.value.details["reason"] == "root_required"
    assert composition.path.read_bytes() == before
