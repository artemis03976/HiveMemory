"""确定性 HTTP chat：真实 Alice 经凭据绑定的操作入口跨轮回读 workspace 写入意图。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hivememory.alice.system import AliceSystem
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models import OMNI_DOLL_PROFILE, PendingAtomStatus, ResolvedAgentProfile
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.server import deps
from hivememory.server.routers.chat import router
from tests.helpers.chat_handoff import make_gateway_decision, make_prepared_run
from tests.helpers.operations import OperationsHarness
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

pytestmark = pytest.mark.e2e


def _events(response):
    """把 HTTP SSE 公开输出解析为按顺序排列的事件。"""
    events = []
    name = ""
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            events.append((name, json.loads(line.removeprefix("data: "))))
    return events


@pytest.mark.asyncio
async def test_next_http_chat_reads_pending_alias_from_previous_completed_process(monkeypatch):
    """第一轮 WRITE 保留 MATERIALIZING，第二轮真实 MTP READ 取得草稿内容。"""
    workspace = make_workspace_identity(owner_user_id="u1")
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")],
        adapters=("http",),
        default_workspace=workspace,
    )
    bus = GlobalSystemBus()
    harness = OperationsHarness(bus, operation_authorizer=access.authorizer)
    payloads = []
    completions = 0

    async def gateway(**_kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def prepare(*, identity_scope, interaction_id=None, **_kwargs):
        return make_prepared_run(
            identity_scope=identity_scope, interaction_id=interaction_id or "chat"
        )

    async def profile(_alias, *, identity_scope):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def finalize(*, payload, **_kwargs):
        # 本验收只验证提交和跨轮回读，不让外部生成流水线提前结算句柄。
        payloads.append(payload)
        return []

    async def topics(**_kwargs):
        return []

    async def completion(**_kwargs):
        nonlocal completions
        turn = completions
        completions += 1
        if turn == 0:
            text = '⟪ WRITE | * | title="跨轮事实" content="HTTP 第一轮登记的内容" ⟫'
        elif turn == 2:
            text = f"⟪ READ | {payloads[0].materialize_tasks[0].pending_alias} | ⟫"
        else:
            text = "完成"

        async def chunks():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason="stop")]
            )

        return chunks()

    monkeypatch.setattr("hivememory.agent_runtime.execution.worker.litellm.acompletion", completion)
    for route, handler in (
        (GlobalRoutes.GATEWAY_PROCESS, gateway),
        (GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare),
        (GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, profile),
        (GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize),
        (GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, topics),
    ):
        bus.register(route, handler)
    alice = AliceSystem(
        config=HiveMemoryConfig().alice, global_bus=bus, operation_entry=harness.entry
    )
    service = make_task_process_service(
        bus,
        cpu=alice.cpu_port,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        workspace_runtime=harness.runtime,
        credential_registry=harness.credentials,
        operation_entry=harness.entry,
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: service
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    await alice.start()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            responses = []
            for message in ("保存事实", "读取刚才的草稿"):
                responses.append(
                    await asyncio.wait_for(
                        client.post(
                            "/api/v1/chat",
                            headers={"X-User-ID": "u1", "X-Workspace-ID": workspace.workspace_id},
                            json={
                                "message": message,
                                "agent_id": "omni_doll",
                                "enable_memory_retrieval": False,
                                "generation_options": {"model": "fake-test-model"},
                            },
                        ),
                        timeout=10,
                    )
                )
        assert [response.status_code for response in responses] == [200, 200]
        assert [_events(response)[-1][1]["status"] for response in responses] == [
            "completed",
            "completed",
        ]
        assert len(payloads) == 2
        (task,) = payloads[0].materialize_tasks
        assert (
            harness.registry.get(task.pending_alias, workspace).status
            == PendingAtomStatus.MATERIALIZING
        )
        assert payloads[1].materialize_tasks == []
        read_result = next(
            event for event in payloads[1].turn_events if event.kind == "tool_result"
        )
        assert read_result.tool_kind == "READ"
        assert read_result.status == "success"
        assert "HTTP 第一轮登记的内容" in read_result.content
        assert completions == 4
    finally:
        await alice.stop()
        harness.runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()
