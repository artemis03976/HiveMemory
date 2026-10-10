"""单次 HTTP chat 经真实 Alice 操作请求完成检索、引用、写入与 CALL。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hivememory.alice.system import AliceSystem
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomStatus,
)
from hivememory.engines.lifecycle.models import EventType
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTaskStatus
from hivememory.server import deps
from hivememory.server.routers.chat import router
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.credentials import ExecutionCredentialRegistry
from tests.helpers.memory import make_memory_metadata
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
    make_workspace_runtime,
)

pytestmark = pytest.mark.e2e


def _events(response):
    """按公开 SSE 协议还原有序事件，不读取服务器私有执行状态。"""
    events = []
    name = ""
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            events.append((name, json.loads(line.removeprefix("data: "))))
    return events


@pytest.mark.asyncio
async def test_http_chat_search_read_write_and_call_claims_intent_and_records_shared_citations(
    monkeypatch, patchouli_chat_stack
):
    """完整 HTTP 链路保留正式引用、CALL 共享读取和本进程 WRITE 的结算归属。"""
    stack = patchouli_chat_stack
    workspace = make_workspace_identity(owner_user_id="u1")
    actor = ActorIdentity(user_id="u1", agent_id="omni_doll")
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")],
        adapters=("http",),
        default_workspace=workspace,
    )
    formal = MemoryAtom(
        meta=make_memory_metadata(user_id="u1", source_agent_id="research_agent"),
        index=IndexLayer(
            title="HTTP 正式上下文",
            summary="供本轮检索与共享读取",
            alias="fact_http_context",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="SEARCH 预热后由 READ 和 CALL 共同交付的正式内容"),
    )
    await stack.mid_term.upsert(formal)
    stack.vector_store.search_matches["HTTP formal context"] = (formal.index.alias,)
    sink = RecordingRuntimeEventSink()
    publisher = RuntimeEventPublisher(sink)
    runtime = make_workspace_runtime(stack.bus)
    runtime.subscribe(stack.bus)
    credentials = ExecutionCredentialRegistry()
    entry = WorkspaceOperationEntry(
        MemoryApplicationService(
            stack.bus,
            operation_authorizer=access.authorizer,
            memory_reader=runtime.aliases,
        ),
        agent=AgentApplicationService(
            stack.bus,
            operation_authorizer=access.authorizer,
            profile_reader=runtime.profiles,
        ),
        credential_registry=credentials,
        intent_registry=runtime.intents,
    )
    script = (
        '⟪ SEARCH | * | query="HTTP formal context" ⟫',
        f"⟪ READ | {formal.index.alias} | ⟫",
        '⟪ WRITE | * | title="本轮新事实" content="单轮请求登记的新内容" ⟫',
        '⟪ CALL | omni_doll | task="总结共享资料" context_refs=["fact_http_context"] ⟫',
        "子帧已读取共享资料",
        "本轮能力迁移验收完成",
    )
    llm_messages = []

    async def completion(*, messages, **_kwargs):
        """只替换外部 LLM 流协议，MTP 解析、帧编排和结果回填保持真实。"""
        llm_messages.append([dict(message) for message in messages])
        text = script[len(llm_messages) - 1]

        async def chunks():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason="stop")]
            )

        return chunks()

    monkeypatch.setattr("hivememory.agent_runtime.execution.worker.litellm.acompletion", completion)
    alice = AliceSystem(
        config=HiveMemoryConfig().alice,
        global_bus=stack.bus,
        event_publisher=publisher,
        operation_entry=entry,
    )
    service = make_task_process_service(
        stack.bus,
        cpu=alice.cpu_port,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        event_publisher=publisher,
        workspace_runtime=runtime,
        credential_registry=credentials,
        operation_entry=entry,
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: service
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    await alice.start()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await asyncio.wait_for(
                client.post(
                    "/api/v1/chat",
                    headers={"X-User-ID": "u1", "X-Workspace-ID": workspace.workspace_id},
                    json={
                        "message": "按顺序检索、读取、登记并共享正式资料",
                        "agent_id": actor.agent_id,
                        "enable_memory_retrieval": False,
                        "generation_options": {"model": "fake-test-model"},
                    },
                ),
                timeout=10,
            )

        assert response.status_code == 200
        events = _events(response)
        terminal, done = events[-1]
        assert terminal == "done"
        assert done["status"] == "completed"
        assert done["stopped"] is False
        assert done["final_text"] == "本轮能力迁移验收完成"
        assert [data["verb"] for name, data in events if name == "mtp_start"] == [
            "SEARCH",
            "READ",
            "WRITE",
            "CALL",
        ]
        assert [data["status"] for name, data in events if name == "mtp_result"] == [
            "success",
            "success",
            "ack",
            "suspend",
        ]
        (child_end,) = [data for name, data in events if name == "sub_agent_end"]
        assert child_end["status"] == "success"
        assert child_end["terminal_status"] == "completed"
        assert child_end["final_text"] == "子帧已读取共享资料"
        assert {data["agent_id"] for name, data in events if name == "token"} == {actor.agent_id}
        assert len(llm_messages) == 6
        assert formal.payload.content in llm_messages[2][-1]["content"]
        assert formal.payload.content in llm_messages[4][0]["content"]
        assert "子帧已读取共享资料" in llm_messages[5][-1]["content"]

        # 已封口的真实 Topic 记录证明四次动作都穿过主帧并进入交互提交。
        (topic_snapshot,) = done["pool_topics"]
        topic = stack.short_term.get(workspace, topic_snapshot["topic_id"])
        (block,) = topic.blocks
        assert block.turn.identity == actor
        assert [event.tool_kind for event in block.turn_events if event.kind == "tool_call"] == [
            "SEARCH",
            "READ",
            "WRITE",
            "CALL",
        ]
        assert [event.status for event in block.turn_events if event.kind == "tool_result"] == [
            "success",
            "success",
            "ack",
            "success",
        ]

        # 引用由真实生命周期链落到存储；READ 与 CALL 各算一次，SEARCH 不计引用。
        cited = await stack.mid_term.get_by_alias(workspace, formal.index.alias, from_actor=actor)
        assert cited.meta.lifecycle.access_count == 2
        history = stack.lifecycle.lifecycle_engine.get_event_history(formal.id)
        assert [event.event_type for event in history] == [EventType.CITATION, EventType.CITATION]
        assert runtime.stats()["atom_hits"] == 2

        # 认领后的任务最终经真实结算事件反馈到 workspace 登记，不会随关闭取消。
        (task_id,) = done["memory_task_ids"]
        task = await stack.controller.wait_task(task_id, timeout=2)
        assert task.status == MemoryGenerationTaskStatus.COMPLETED
        assert task.belong_to == workspace
        assert task.from_actor == actor
        pending = runtime.intents.get(task.pending_alias, workspace)
        assert pending.status == PendingAtomStatus.SETTLED
        assert pending.process_id == done["process_id"]
        assert pending.focus.content == "单轮请求登记的新内容"
        assert pending.settlement.resolution == PendingAtomResolution.CREATED
        assert pending.settlement.canonical_alias == task.canonical_alias
        assert runtime.intents.claim_process(done["process_id"]) == []

        # 正式资源读回同时核对存储正文、归属与来源，避免仅凭结算回执判定物化成功。
        materialized = await stack.mid_term.get_by_alias(
            workspace, task.canonical_alias, from_actor=actor
        )
        assert materialized.payload.content == "HTTP chat materialized content"
        assert materialized.workspace_identity == workspace
        assert materialized.index.memory_type == MemoryType.FACT
        assert materialized.meta.provenance.source_agent_id == actor.agent_id

        observations = [
            event
            for event in sink.events
            if event.event_type
            in (
                RuntimeEventType.AGENT_RUN_STARTED.value,
                RuntimeEventType.AGENT_RUN_COMPLETED.value,
            )
        ]
        assert [event.event_type for event in observations] == [
            RuntimeEventType.AGENT_RUN_STARTED.value,
            RuntimeEventType.AGENT_RUN_COMPLETED.value,
        ]
        assert {event.agent_id for event in observations} == {actor.agent_id}
        assert {event.workspace_id for event in observations} == {workspace.workspace_id}
        assert {event.process_id for event in observations} == {done["process_id"]}
    finally:
        await alice.stop()
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()
