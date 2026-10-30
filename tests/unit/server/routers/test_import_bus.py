"""Import Bus 路由单元测试。

被测对象：``POST /api/v1/import/turns`` 的传输校验、声明解析、以请求声明的
principal 与凭据经统一认证网关的请求级认证与错误分类。应用服务是真实的 ``ImportBusService``（导入
授权使用与网关同一组合的操作授权者），编排器以记录调用的替身代替；编排
本身的行为由 System 归属层测试覆盖。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.server import deps
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.routers.import_bus import router
from hivememory.system.application.import_bus_service import ImportBusService
from hivememory.system.services.import_bus import (
    ImportBusApplyFailedError,
    ImportBusUnavailableError,
    ImportedTurn,
    ImportTurnResult,
)
from tests.helpers.workspace import make_access_composition, make_actor_access_record

_PRINCIPAL = "harness:claude-code"
_TOKEN = "token-1"
_HEADERS = {"X-Principal-ID": _PRINCIPAL, "Authorization": f"Bearer {_TOKEN}"}


class _RecordingIngestor:
    """记录编排器收到的导入与授权结果；可配置为抛出指定错误。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._error = error

    async def ingest(self, turn, *, identity_scope, principal_id):
        self.calls.append(
            {"turn": turn, "identity_scope": identity_scope, "principal_id": principal_id}
        )
        if self._error is not None:
            raise self._error
        return ImportTurnResult(status="accepted", interaction_id="import_0123")


def _connector_access(records=None):
    """连接器的接入登记：外部 principal 只经 import adapter、携带凭据接入。"""
    composition = make_access_composition(
        records or [make_actor_access_record(owner_user_id="default", agent_id=None)],
        principal_id=_PRINCIPAL,
        adapters=("import",),
        credential=_TOKEN,
    )
    return {deps.get_access_gateway: lambda: composition.gateway}, composition


def _client(ingestor=None, *, access=None, limit: int = 1024 * 1024, headers=None):
    overrides, composition = access or _connector_access()
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix="/api/v1")
    service = ImportBusService(
        ingestor=ingestor or _RecordingIngestor(),
        operation_authorizer=composition.authorizer,
    )
    app.dependency_overrides[deps.get_import_bus_service] = lambda: service
    app.dependency_overrides[deps.get_import_request_limit] = lambda: limit
    app.dependency_overrides.update(overrides)
    return TestClient(app, headers=_HEADERS if headers is None else headers), composition


def _body(**overrides):
    body = {
        "source": "claude-code",
        "external_conversation_id": "session-1",
        "external_turn_id": "prompt-1",
        "agent_id": "claude_code",
        "outcome": "completed",
        "user_message": "修一下测试",
        "events": [
            {
                "kind": "tool_call",
                "sequence": 1,
                "content": "",
                "action_id": "toolu-1",
                "tool_name": "Bash",
                "tool_args": {"command": "pytest -q"},
            },
            {"kind": "assistant_message", "sequence": 2, "content": "已修复"},
        ],
        "recent_turns": [{"user_message": "上一轮", "assistant_text": "上一轮回复"}],
    }
    body.update(overrides)
    return body


def test_import_turn_authenticates_agent_claims_and_returns_result():
    """以请求声明的 principal、凭据与 (user, 具体 Agent) 认证；编排器收到不含认证声明的记录。"""
    ingestor = _RecordingIngestor()
    client, composition = _client(ingestor)

    response = client.post("/api/v1/import/turns", json=_body())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "interaction_id": "import_0123"}
    [call] = ingestor.calls
    assert type(call["turn"]) is ImportedTurn
    assert call["turn"].external_turn_id == "prompt-1"
    assert call["identity_scope"].actor_identity.user_id == "default"
    assert call["identity_scope"].actor_identity.agent_id == "claude_code"
    assert call["identity_scope"].workspace_identity.workspace_id == "main_workspace"
    assert call["principal_id"] == _PRINCIPAL


@pytest.mark.parametrize(
    "overrides",
    [
        {"unexpected": True},
        {"events": [{"kind": "user_message", "sequence": 1, "content": "再问一次"}]},
        {
            "events": [
                {"kind": "assistant_message", "sequence": 2, "content": "a"},
                {"kind": "assistant_message", "sequence": 2, "content": "b"},
            ]
        },
        {"source": "Claude Code"},
    ],
    ids=["unknown-field", "user-message-event", "non-increasing", "source-format"],
)
def test_invalid_import_body_is_rejected_before_import(overrides):
    """未知字段、用户消息事件、非递增序号与非法来源在入口拒绝。"""
    ingestor = _RecordingIngestor()
    client, _ = _client(ingestor)

    response = client.post("/api/v1/import/turns", json=_body(**overrides))

    assert response.status_code == 422
    assert response.json()["error"] == "request.validation_failed"
    assert ingestor.calls == []


@pytest.mark.parametrize(
    ("agent_id", "code"),
    [(None, "request.agent_required"), (SYSTEM_AGENT_ID, "request.agent_reserved")],
    ids=["missing", "system"],
)
def test_import_requires_concrete_agent(agent_id, code):
    """导入是 Agent action：缺失 Agent 或声明保留的 system 返回 400。"""
    ingestor = _RecordingIngestor()
    client, _ = _client(ingestor)
    body = _body()
    if agent_id is None:
        del body["agent_id"]
    else:
        body["agent_id"] = agent_id

    response = client.post("/api/v1/import/turns", json=body)

    assert response.status_code == 400
    assert response.json()["error"] == code
    assert ingestor.calls == []


@pytest.mark.parametrize(
    ("headers", "status", "error", "reason"),
    [
        ({"Authorization": f"Bearer {_TOKEN}"}, 400, "request.principal_required", None),
        (
            {"X-Principal-ID": _PRINCIPAL, "Authorization": "Bearer wrong"},
            403,
            "workspace.admission_denied",
            "invalid_credential",
        ),
    ],
    ids=["missing-principal", "wrong-credential"],
)
def test_import_requires_declared_principal_and_matching_credential(headers, status, error, reason):
    """缺少 principal 声明返回 400，凭据不一致返回 403；两者都不可重试，不交给编排器。"""
    ingestor = _RecordingIngestor()
    client, _ = _client(ingestor, headers=headers)

    response = client.post("/api/v1/import/turns", json=_body())

    assert response.status_code == status
    assert (response.json()["error"], response.json().get("reason")) == (error, reason)
    assert ingestor.calls == []


def test_unregistered_agent_is_denied_admission():
    """未登记的 Agent 不能导入：准入失败返回 403，不可重试。"""
    access = _connector_access(
        [make_actor_access_record(owner_user_id="default", agent_id="omni_doll")]
    )
    ingestor = _RecordingIngestor()
    client, _ = _client(ingestor, access=access)

    response = client.post("/api/v1/import/turns", json=_body())

    assert response.status_code == 403
    assert ingestor.calls == []


def test_oversized_body_is_rejected_as_not_retryable():
    """请求体超过上限返回 413，不交给编排器。"""
    ingestor = _RecordingIngestor()
    client, _ = _client(ingestor, limit=256)

    response = client.post("/api/v1/import/turns", json=_body(user_message="x" * 512))

    assert response.status_code == 413
    assert response.json()["error"] == "request.body_too_large"
    assert ingestor.calls == []


def test_oversized_body_is_rejected_before_it_is_parsed():
    """声明的长度超过上限时不读取、不解析请求体：内容不是合法 JSON 也返回 413。"""
    ingestor = _RecordingIngestor()
    client, _ = _client(ingestor, limit=256)

    response = client.post(
        "/api/v1/import/turns",
        content=b"{" * 512,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "request.body_too_large"
    assert ingestor.calls == []


def test_closed_authentication_gateway_returns_retryable_unavailable():
    """服务关闭中认证网关拒绝新认证，返回可重试的 503。"""
    ingestor = _RecordingIngestor()
    client, composition = _client(ingestor)
    composition.gateway.close()

    response = client.post("/api/v1/import/turns", json=_body())

    assert response.status_code == 503
    assert response.json()["reason"] == "authentication_gateway_closed"
    assert ingestor.calls == []


@pytest.mark.parametrize("reason", ["queue_unavailable", "apply_pending"])
def test_retryable_unavailability_returns_503_with_its_reason(reason):
    """提交队列拥塞与等待应用超时经稳定原因码返回 503，连接器据此保留该轮重发。"""
    client, _ = _client(_RecordingIngestor(ImportBusUnavailableError(reason=reason)))

    response = client.post("/api/v1/import/turns", json=_body())

    assert response.status_code == 503
    assert response.json()["error"] == "import_bus.unavailable"
    assert response.json()["reason"] == reason


def test_apply_failure_returns_500_with_stable_error_code():
    """应用失败返回 500 与稳定错误码：连接器有限次重发，重发时重新路由并再次提交。"""
    client, _ = _client(_RecordingIngestor(ImportBusApplyFailedError()))

    response = client.post("/api/v1/import/turns", json=_body())

    assert response.status_code == 500
    assert response.json()["error"] == "import_bus.apply_failed"
