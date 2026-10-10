"""HTTP chat 经真实身份边界、任务进程和 Patchouli 物化链路的确定性 E2E。"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hivememory.core.models import ActorIdentity, TurnEvent, WriteFocus
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTaskStatus
from hivememory.server import deps
from hivememory.server.routers.chat import router
from hivememory.workspace.contracts import SubmitWriteIntentRequest
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_http_chat_preserves_split_identity_through_finalize_and_materialization(
    patchouli_chat_stack,
):
    """HTTP 完成后交互与物化任务保留身份，done 话题池来自真实归属读取。"""
    workspace = make_workspace_identity(owner_user_id="u1")
    other_workspace = make_workspace_identity(owner_user_id="u1", workspace_id="other")
    actor = ActorIdentity(user_id="u1", agent_id="omni_doll")
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")],
        adapters=("http",),
        default_workspace=workspace,
    )
    stack = patchouli_chat_stack
    global_bus = stack.bus
    short_term = stack.short_term
    mid_term = stack.mid_term
    controller = stack.controller
    foreign_topic = short_term.create(other_workspace, topic_title="其他归属")
    submitted = []

    async def write(submit):
        """测试 CPU 经操作入口提交意图，物化任务由进程认领。"""
        submitted.append(
            await submit(SubmitWriteIntentRequest(focus=WriteFocus(content="保存本轮事实")))
        )

    cpu = ScriptedCPU(
        result=make_cpu_result(
            final_text="本轮完成",
            turn_events=[
                TurnEvent(
                    kind="assistant_message", sequence=0, role="assistant", content="本轮完成"
                )
            ],
        ),
        operation_script=write,
    )
    process_service = make_task_process_service(
        global_bus,
        cpu=cpu,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: process_service
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                headers={"X-User-ID": "u1", "X-Workspace-ID": workspace.workspace_id},
                json={
                    "message": "保存本轮事实",
                    "agent_id": actor.agent_id,
                    "session_id": "external-session",
                    "enable_memory_retrieval": False,
                },
            )
        assert response.status_code == 200
        events = []
        event_name = ""
        for line in response.text.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((event_name, json.loads(line.removeprefix("data: "))))
        terminal, done = events[-1]
        assert terminal == "done"
        assert done["status"] == "completed"
        assert done["final_text"] == "本轮完成"
        (topic_snapshot,) = done["pool_topics"]
        assert topic_snapshot["workspace_identity"] == workspace.model_dump(mode="json")
        assert topic_snapshot["block_count"] == 1
        assert topic_snapshot["last_turn"] == {"user": "保存本轮事实", "assistant": "本轮完成"}
        topic = short_term.get(workspace, topic_snapshot["topic_id"])
        assert topic.blocks[0].turn.identity == actor
        assert "session_id" not in topic.blocks[0].turn.identity.model_dump()
        assert short_term.get(other_workspace, foreign_topic.topic_id) == foreign_topic

        (task_id,) = done["memory_task_ids"]
        task = await controller.wait_task(task_id, timeout=2)
        assert task.status == MemoryGenerationTaskStatus.COMPLETED
        assert task.belong_to == workspace
        assert task.from_actor == actor
        atom = await mid_term.get_by_alias(workspace, task.canonical_alias, from_actor=actor)
        assert atom.payload.content == "HTTP chat materialized content"
        assert atom.workspace_identity == workspace
        assert atom.meta.provenance.source_agent_id == actor.agent_id
        assert len(cpu.calls) == 1
        assert cpu.calls[0].manifest.labels.agent_id == actor.agent_id
        assert cpu.calls[0].manifest.labels.workspace_id == workspace.workspace_id
        assert cpu.closed is True
    finally:
        access.gateway.close()
        access.gateway.revoke_all_contexts()
