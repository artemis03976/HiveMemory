"""会话 HTTP 管理入口与真实 root API、认证网关、会话存储的集成。"""

import httpx
import pytest
from fastapi import FastAPI

from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import InteractionPayload
from hivememory.server.deps import get_root_session_service
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.routers.sessions import router
from hivememory.workspace.capability.root.session import RootSessionService
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.sessions import make_compaction_checkpoint
from tests.helpers.workspace import make_server_access_overrides


@pytest.mark.asyncio
async def test_http_session_crud_and_busy_error():
    """HTTP 只投影恢复历史所需的字段，不含 Workspace 归属；忙会话不能被删除。"""
    overrides, access = make_server_access_overrides()
    store = ConversationSessionService(WorkspaceSessionConfig())
    root = RootSessionService(store, operation_authorizer=access.authorizer)
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides.update(overrides)
    app.dependency_overrides[get_root_session_service] = lambda: root

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post("/api/v1/sessions", json={"title": "旧标题"})
        assert created.status_code == 201
        assert "belong_to" not in created.json()
        session_id = created.json()["session_id"]
        renamed = await client.patch(f"/api/v1/sessions/{session_id}", json={"title": "新标题"})
        assert renamed.json()["title"] == "新标题"
        history = await client.get(f"/api/v1/sessions/{session_id}")
        assert history.json()["turns"] == []
        listed = await client.get("/api/v1/sessions")
        assert [(s["session_id"], s["title"]) for s in listed.json()] == [(session_id, "新标题")]
        store.claim(access.default_workspace, session_id, "p")
        store.append(
            access.default_workspace,
            session_id,
            process_id="p",
            agent_id="a",
            payload=InteractionPayload(
                user_message="问题",
                assistant_final_text="原文",
                turn_events=[
                    TurnEvent(
                        kind="tool_call",
                        sequence=1,
                        role="assistant",
                        content="READ m1",
                        action_id="act-1",
                        tool_name="read",
                        tool_kind="READ",
                        tool_args={"ids": ["m1"]},
                        target="m1",
                    )
                ],
            ),
            topic_id="t",
        )
        checkpoint = make_compaction_checkpoint(reduced_through="p")
        store.accept_checkpoint(
            access.default_workspace, session_id, process_id="p", checkpoint=checkpoint
        )
        restored = await client.get(f"/api/v1/sessions/{session_id}")
        assert restored.json()["compaction_checkpoint"] == checkpoint.model_dump(mode="json")
        restored_payload = restored.json()["turns"][0]["payload"]
        assert restored_payload == {
            "outcome": "completed",
            "user_message": "问题",
            "assistant_final_text": "原文",
            "turn_events": [
                {
                    "kind": "tool_call",
                    "sequence": 1,
                    "content": "READ m1",
                    "action_id": "act-1",
                    "tool_kind": "READ",
                    "target": "m1",
                    "tool_args": {"ids": ["m1"]},
                    "status": None,
                }
            ],
        }
        assert "compaction_checkpoint" not in (await client.get("/api/v1/sessions")).json()[0]
        busy = await client.delete(f"/api/v1/sessions/{session_id}")
        assert busy.status_code == 409
        assert busy.json()["error"] == "workspace.session.busy"
        store.release(access.default_workspace, session_id, "p")
        assert (await client.delete(f"/api/v1/sessions/{session_id}")).status_code == 204
        missing = await client.get(f"/api/v1/sessions/{session_id}")
        assert missing.status_code == 404
        assert missing.json()["error"] == "workspace.session.not_found"
