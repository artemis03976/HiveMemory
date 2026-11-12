"""接入登记与访问记录路由：接入登记只列出与修改、不返回凭据明文，用户级与精确记录的
路径各自编辑对应的记录，禁用却带白名单的请求在入口被拒。"""

from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.config.access import (
    SystemPrincipalsFile,
    WorkspaceActorsFile,
    file_digest,
)
from hivememory.server import deps
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.routers.access import router
from hivememory.system.access import SystemActorAccessRegistry, SystemPrincipalAuthenticator
from hivememory.workspace.authentication import ActorAuthenticationGateway, WorkspaceAuthenticator
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.root.access import RootAccessService
from hivememory.workspace.registry import WorkspaceActorAccessRegistry

_PRINCIPALS_YAML = """\
principals:
  - principal_id: "hivememory:http-server"
    kind: "http-server"
    adapters: ["http"]
    allow_system_actor: true
"""

_ACTORS_YAML = """\
workspace_actors:
  - owner_user_id: "default"
    workspace_id: "main_workspace"
    user_id: "default"
    allowed_operations: ["memory.read"]
  - owner_user_id: "default"
    workspace_id: "main_workspace"
    user_id: "default"
    agent_id: "system"
"""


def _load(path: Path, text: str, model):
    path.write_text(text, encoding="utf-8")
    return model.model_validate(yaml.safe_load(text)), file_digest(path)


@pytest.fixture
def principals(tmp_path):
    principals_file, principals_digest = _load(
        tmp_path / "system_principals.yaml", _PRINCIPALS_YAML, SystemPrincipalsFile
    )
    return SystemActorAccessRegistry.from_config(
        principals_file,
        file_path=tmp_path / "system_principals.yaml",
        file_digest=principals_digest,
        credentials_path=tmp_path / "principals.secrets.yaml",
    )


@pytest.fixture
def client(tmp_path, principals):
    """真实的注册表、认证网关与 root 服务，登记文件放在 tmp_path。"""
    actors_file, actors_digest = _load(
        tmp_path / "workspace_actors.yaml", _ACTORS_YAML, WorkspaceActorsFile
    )
    actors = WorkspaceActorAccessRegistry.from_config(
        actors_file, file_path=tmp_path / "workspace_actors.yaml", file_digest=actors_digest
    )
    authorizer = WorkspaceOperationAuthorizer(actors)
    gateway = ActorAuthenticationGateway(
        principals=SystemPrincipalAuthenticator(principals),
        authenticator=WorkspaceAuthenticator(actors),
    )
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides.update(
        {
            deps.get_access_gateway: lambda: gateway,
            deps.get_server_principal_id: lambda: "hivememory:http-server",
            deps.get_principal_registry: lambda: principals,
            deps.get_root_access_service: lambda: RootAccessService(
                actors, operation_authorizer=authorizer
            ),
        }
    )
    return TestClient(app)


def test_external_source_can_declare_more_agents_and_never_exposes_its_credential(
    client, principals
):
    """插件安装登记的来源可以增加可声明的 Agent；修改与列表的响应都只给凭据的打码值。"""
    _, credential = principals.register(
        "harness:claude-code",
        adapters=frozenset({"mcp", "import"}),
        allowed_agent_ids=frozenset({"claude_code"}),
    )

    updated = client.put(
        "/api/v1/access/principals/harness:claude-code",
        json={"adapters": ["mcp", "import"], "allowed_agent_ids": ["claude_code", "reviewer"]},
    )
    listing = client.get("/api/v1/access/principals")

    assert updated.status_code == 200
    assert credential not in updated.text and credential not in listing.text
    entry = next(p for p in listing.json() if p["principal_id"] == "harness:claude-code")
    assert entry["credential_source"] == "generated"
    assert entry["allowed_agent_ids"] == ["claude_code", "reviewer"]


def test_user_and_agent_routes_edit_their_own_records(client):
    """精确记录与用户级记录的路径各自编辑对应的记录，列表不含 system 记录。"""
    client.put(
        "/api/v1/access/actors/agents/a1",
        json={"enabled": True, "allowed_operations": ["memory.write"]},
    )
    client.put("/api/v1/access/actors/user", json={"enabled": False, "allowed_operations": []})

    records = {r["agent_id"]: r for r in client.get("/api/v1/access/actors").json()}
    assert set(records) == {None, "a1"}
    assert records["a1"]["allowed_operations"] == ["memory.write"]
    assert records[None]["enabled"] is False

    assert client.delete("/api/v1/access/actors/agents/a1").status_code == 204
    assert [r["agent_id"] for r in client.get("/api/v1/access/actors").json()] == [None]


def test_disabled_record_with_operations_is_rejected_at_the_entry(client):
    """禁用却带白名单的请求在入口以校验错误拒绝，不到达注册表。"""
    response = client.put(
        "/api/v1/access/actors/agents/a1",
        json={"enabled": False, "allowed_operations": ["memory.read"]},
    )

    assert response.status_code == 422
    assert [r["agent_id"] for r in client.get("/api/v1/access/actors").json()] == [None]
