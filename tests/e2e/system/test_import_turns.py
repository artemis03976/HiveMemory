"""确定性 HTTP 导入链路：真实 Gateway/Patchouli，外部模型与存储使用替身。

经公开入口导入一轮完整记录，最终在 Patchouli 中形成该轮的交互；同一轮
重发不形成第二份交互。编排、授权与错误分类的用例矩阵在各自的归属层。
"""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hivememory.config.gateway import SystemGatewayConfig
from hivememory.server import deps
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.routers.import_bus import router as import_bus_router
from hivememory.system.application.import_bus_service import ImportBusService
from hivememory.system.services.import_bus import ImportBusIngestor
from tests.helpers.workspace import make_access_composition, make_actor_access_record

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_http_import_turn_forms_interaction_in_patchouli(patchouli_chat_stack):
    """连接器导入一轮：经认证网关与真实 Gateway 落位，Patchouli 话题中出现该轮。"""
    stack = patchouli_chat_stack
    access = make_access_composition(
        [make_actor_access_record(agent_id="claude_code", allowed_operations=frozenset())],
        adapters=("http",),
    )
    workspace = access.default_workspace
    gateway_config = SystemGatewayConfig()
    service = ImportBusService(
        ingestor=ImportBusIngestor(
            stack.bus,
            gateway_request_timeout_ms=gateway_config.workflow.default_request_timeout_ms,
            gateway_session_turn_limit=gateway_config.context_preparation.session_turn_limit,
        ),
        operation_authorizer=access.authorizer,
    )
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(import_bus_router, prefix="/api/v1")
    app.dependency_overrides[deps.get_import_bus_service] = lambda: service
    app.dependency_overrides[deps.get_import_request_limit] = lambda: 1024 * 1024
    app.dependency_overrides[deps.get_access_gateway] = lambda: access.gateway
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    body = {
        "source": "claude-code",
        "external_conversation_id": "session-e2e",
        "external_turn_id": "prompt-e2e",
        "agent_id": "claude_code",
        "outcome": "completed",
        "user_message": "为什么部署脚本失败了",
        "events": [
            {
                "kind": "tool_call",
                "sequence": 1,
                "content": "",
                "action_id": "toolu-1",
                "tool_name": "Bash",
                "tool_args": {"command": "./deploy.sh"},
            },
            {
                "kind": "tool_result",
                "sequence": 2,
                "content": "permission denied",
                "action_id": "toolu-1",
                "status": "error",
            },
            {"kind": "assistant_message", "sequence": 3, "content": "脚本缺少执行权限"},
        ],
        "assistant_final_text": "脚本缺少执行权限",
        "model_used": "claude-test",
    }

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-User-ID": workspace.owner_user_id},
    ) as client:
        accepted = await client.post("/api/v1/import/turns", json=body)
        await stack.interaction_queue.drain_all()
        resent = await client.post("/api/v1/import/turns", json=body)
        await stack.interaction_queue.drain_all()

    assert accepted.status_code == 200
    assert accepted.json()["status"] == "accepted"
    assert resent.json() == {
        "status": "duplicate",
        "interaction_id": accepted.json()["interaction_id"],
    }
    [topic] = stack.short_term.list_by_workspace(workspace)
    [block] = topic.blocks
    assert block.turn.identity.agent_id == "claude_code"
    assert block.turn.user_query == "为什么部署脚本失败了"
    assert block.turn.assistant_final_text == "脚本缺少执行权限"
    assert [result.status for action in block.turn.actions for result in action.results] == [
        "error"
    ]
