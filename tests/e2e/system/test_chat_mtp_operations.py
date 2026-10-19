"""确定性 HTTP chat：真实 Alice 经线程 context 绑定的操作入口执行 MTP 资源操作。

覆盖单轮内的检索、引用、写入与 CALL，跨轮回读写入意图，以及 CPU 未成功结束时已接纳写入的落库。
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hivememory.alice.system import AliceSystem
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomStatus,
    ResolvedAgentProfile,
    WriteFocus,
)
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.engines.lifecycle.models import EventType
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTaskStatus
from hivememory.server import deps
from hivememory.server.routers.chat import router
from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService
from hivememory.workspace.capability.operations import WorkspaceOperationEntry
from hivememory.workspace.contracts import SubmitWriteIntentRequest
from tests.helpers.chat_handoff import make_gateway_decision, make_prepared_run
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.llm import CallbackLLMService
from tests.helpers.memory import make_memory_metadata
from tests.helpers.operations import OperationsHarness
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
async def test_http_chat_search_read_write_and_call_dispatches_intent_and_records_shared_citations(
    patchouli_chat_stack,
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
    entry = WorkspaceOperationEntry(
        ActorMemoryService(
            stack.bus,
            operation_authorizer=access.authorizer,
            memory_reader=runtime.aliases,
            dispatch_topics=access.thread_spawner,
        ),
        agent=ActorAgentService(
            operation_authorizer=access.authorizer,
            thread_spawner=access.thread_spawner,
            profile_reader=runtime.profiles,
        ),
        operation_authorizer=access.authorizer,
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

    async def completion(messages, _params):
        """只替换外部 LLM 流协议，MTP 解析、帧编排和结果回填保持真实。"""
        llm_messages.append([dict(message) for message in messages])
        text = script[len(llm_messages) - 1]
        yield text

    alice = AliceSystem(
        config=HiveMemoryConfig().alice,
        event_publisher=publisher,
        operation_entry=entry,
        llm_service=CallbackLLMService(stream=completion),
    )
    service = make_task_process_service(
        stack.bus,
        cpu=alice.cpu_port,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        event_publisher=publisher,
        process_table=access.process_table,
        workspace_runtime=runtime,
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
        topic = stack.short_term.get(workspace, done["topic_id"])
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

        # 执行期间派发的任务最终经真实结算事件反馈到 workspace 登记，不会随关闭取消。
        (created,) = await stack.controller.list_tasks()
        task_id = created.task_id
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


@pytest.mark.asyncio
async def test_http_call_uses_callee_permissions_private_memory_and_write_ownership(
    patchouli_chat_stack,
):
    """HTTP CALL 子线程独立授权，完成后 WRITE 以被调用方身份进入真实 finalize。"""
    from hivememory.core.access import WorkspaceOperation

    stack = patchouli_chat_stack
    workspace = make_workspace_identity(owner_user_id="u1")
    caller = ActorIdentity(user_id="u1", agent_id="omni_doll")
    callee = ActorIdentity(user_id="u1", agent_id="worker")
    access = make_access_composition(
        [
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="omni_doll",
                allowed_operations={
                    WorkspaceOperation.AGENT_CALL,
                    WorkspaceOperation.MEMORY_SEARCH,
                },
            ),
            make_actor_access_record(
                owner_user_id="u1",
                agent_id="worker",
                allowed_operations={
                    WorkspaceOperation.MEMORY_READ,
                    WorkspaceOperation.MEMORY_WRITE,
                    WorkspaceOperation.AGENT_CALL,
                },
            ),
        ],
        adapters=("http",),
        default_workspace=workspace,
    )
    profile = MemoryAtom(
        meta=make_memory_metadata(user_id="u1", source_agent_id="worker"),
        index=IndexLayer(alias="worker", title="独立子线程", memory_type=MemoryType.AGENT_PROFILE),
        payload=PayloadLayer(
            content="按自己的权限完成任务", agent_config={"model_name": "fake-test-model"}
        ),
    )
    await stack.mid_term.upsert(profile)
    for actor in (caller, callee):
        await stack.mid_term.upsert(
            MemoryAtom(
                meta=make_memory_metadata(
                    user_id="u1", source_agent_id=actor.agent_id, visibility="PRIVATE"
                ),
                index=IndexLayer(
                    alias=f"private_{actor.agent_id}", title="私有记忆", memory_type=MemoryType.FACT
                ),
                payload=PayloadLayer(content=f"{actor.agent_id} 的私有正文"),
            )
        )
    runtime = make_workspace_runtime(stack.bus)
    runtime.subscribe(stack.bus)
    entry = WorkspaceOperationEntry(
        ActorMemoryService(
            stack.bus,
            operation_authorizer=access.authorizer,
            memory_reader=runtime.aliases,
            dispatch_topics=access.thread_spawner,
        ),
        agent=ActorAgentService(
            operation_authorizer=access.authorizer,
            profile_reader=runtime.profiles,
            thread_spawner=access.thread_spawner,
        ),
        operation_authorizer=access.authorizer,
    )
    script = (
        '⟪ CALL | worker | task="读取自己的私有资料并写入" ⟫',
        '⟪ SEARCH | * | query="不应继承父帧 SEARCH" ⟫',
        "⟪ READ | private_omni_doll | ⟫",
        "⟪ READ | private_worker | ⟫",
        '⟪ WRITE | * | content="子线程独立写入" ⟫',
        '⟪ CALL | omni_doll | task="禁止递归" ⟫',
        "子线程完成",
        "主线程完成",
    )
    messages_seen = []

    async def completion(messages, _params):
        messages_seen.append([dict(message) for message in messages])
        text = script[len(messages_seen) - 1]
        yield text

    alice = AliceSystem(
        config=HiveMemoryConfig().alice,
        operation_entry=entry,
        llm_service=CallbackLLMService(stream=completion),
    )
    service = make_task_process_service(
        stack.bus,
        cpu=alice.cpu_port,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        workspace_runtime=runtime,
        operation_entry=entry,
        process_table=access.process_table,
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
                        "message": "委托 worker",
                        "agent_id": "omni_doll",
                        "enable_memory_retrieval": False,
                        "generation_options": {"model": "fake-test-model"},
                    },
                ),
                timeout=10,
            )
        assert response.status_code == 200
        terminal, done = _events(response)[-1]
        assert (terminal, done["status"], done["final_text"]) == ("done", "completed", "主线程完成")
        assert len(messages_seen) == 8
        assert 'code="mtp.permission.denied"' in messages_seen[2][-1]["content"]
        assert "omni_doll 的私有正文" not in str(messages_seen)
        assert "worker 的私有正文" in messages_seen[4][-1]["content"]
        assert 'code="mtp.permission.denied"' in messages_seen[6][-1]["content"]
        (created,) = await stack.controller.list_tasks()
        task_id = created.task_id
        task = await stack.controller.wait_task(task_id, timeout=2)
        assert task.status == MemoryGenerationTaskStatus.COMPLETED
        assert task.from_actor == callee
        assert task.belong_to == workspace
        pending = runtime.intents.get(task.pending_alias, workspace)
        assert pending.process_id == done["process_id"]
        assert pending.from_actor == callee
        assert pending.focus.content == "子线程独立写入"
        assert pending.status == PendingAtomStatus.SETTLED
        memory = await stack.mid_term.get_by_alias(
            workspace, task.canonical_alias, from_actor=callee
        )
        assert memory.meta.provenance.source_agent_id == "worker"
        assert memory.payload.content == "HTTP chat materialized content"
    finally:
        await alice.stop()
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()


@pytest.mark.asyncio
async def test_next_http_chat_reads_pending_alias_from_previous_completed_process():
    """第一轮 WRITE 保留 MATERIALIZING，第二轮真实 MTP READ 取得草稿内容。"""
    workspace = make_workspace_identity(owner_user_id="u1")
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")],
        adapters=("http",),
        default_workspace=workspace,
    )
    bus = GlobalSystemBus()
    harness = OperationsHarness(bus, operation_authorizer=access.authorizer)
    entry = harness.make_entry(access.authorizer, access.thread_spawner)
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

    async def completion(_messages, _params):
        nonlocal completions
        turn = completions
        completions += 1
        if turn == 0:
            text = '⟪ WRITE | * | title="跨轮事实" content="HTTP 第一轮登记的内容" ⟫'
        elif turn == 2:
            text = f"⟪ READ | {harness.admission.submissions[0][0].pending_alias} | ⟫"
        else:
            text = "完成"
        yield text

    for route, handler in (
        (GatewayPublicRoutes.PROCESS, gateway),
        (PatchouliRoutes.PREPARE_AGENT_RUN, prepare),
        (PatchouliRoutes.GET_AGENT_PROFILE, profile),
        (PatchouliRoutes.FINALIZE_AGENT_RUN, finalize),
        (PatchouliRoutes.TOPIC_LIST_ACTIVE, topics),
    ):
        bus.register(route, handler)
    alice = AliceSystem(
        config=HiveMemoryConfig().alice,
        operation_entry=entry,
        llm_service=CallbackLLMService(stream=completion),
    )
    service = make_task_process_service(
        bus,
        cpu=alice.cpu_port,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        workspace_runtime=harness.runtime,
        process_table=access.process_table,
        operation_entry=entry,
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
        ((task, _scope),) = harness.admission.submissions
        assert (
            harness.registry.get(task.pending_alias, workspace).status
            == PendingAtomStatus.MATERIALIZING
        )
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


@pytest.mark.asyncio
async def test_http_write_survives_cancelled_cpu_and_process_close(patchouli_chat_stack):
    """WRITE 接纳后 CPU 无交互输出并取消，交互不提交，记忆仍经真实生成链落库。

    各种 CPU 终态下意图都不撤回由任务进程的意图集成测试逐项守护，此处只保留端到端代表。
    """
    status = "cancelled"
    from hivememory.components.events.publisher import RuntimeEventPublisher
    from hivememory.core.contracts.runtime_events import RuntimeEventType
    from hivememory.core.models import PendingAtomStatus
    from tests.helpers.workspace import make_workspace_runtime

    stack = patchouli_chat_stack
    stack.vector_store.write_allowed.clear()
    workspace = make_workspace_identity(owner_user_id="u1")
    actor = ActorIdentity(user_id="u1", agent_id="omni_doll")
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")],
        adapters=("http",),
        default_workspace=workspace,
    )
    runtime = make_workspace_runtime(stack.bus)
    runtime.subscribe(stack.bus)
    submitted = []

    async def write(submit):
        """进程仍在 Actor 阶段时接纳写入，后续 CPU 终态独立失败。"""
        submitted.append(
            await submit(SubmitWriteIntentRequest(WriteFocus(content="保留失败轮的写入")))
        )
        (created,) = [
            event
            for event in stack.events.events
            if event.event_type == RuntimeEventType.MEMORY_TASK_CREATED
        ]
        assert created.data["pending_alias"] == submitted[0].pending_alias

    cpu = ScriptedCPU(result=make_cpu_result(status=status, final_text=""), operation_script=write)
    service = make_task_process_service(
        stack.bus,
        cpu=cpu,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        workspace_runtime=runtime,
        event_publisher=RuntimeEventPublisher(stack.events),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: service
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                headers={"X-User-ID": "u1", "X-Workspace-ID": workspace.workspace_id},
                json={
                    "message": "当前一轮不应进入生成背景",
                    "agent_id": actor.agent_id,
                    "enable_memory_retrieval": False,
                },
            )
        assert response.status_code == 200
        done = json.loads(
            [
                line.removeprefix("data: ")
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ][-1]
        )
        assert done["status"] == status
        (created,) = [
            event
            for event in stack.events.events
            if event.event_type == RuntimeEventType.MEMORY_TASK_CREATED
        ]
        # HTTP 已交付失败/取消终态，此时尚未落库；关闭不能提前结算或撤回意图。
        assert (
            runtime.intents.get(submitted[0].pending_alias, workspace).status
            == PendingAtomStatus.MATERIALIZING
        )
        stack.vector_store.write_allowed.set()
        task = await stack.controller.wait_task(created.task_id, timeout=2)
        assert task.status == MemoryGenerationTaskStatus.COMPLETED
        pending = runtime.intents.get(submitted[0].pending_alias, workspace)
        assert pending.status == PendingAtomStatus.SETTLED
        atom = await stack.mid_term.get_by_alias(workspace, task.canonical_alias, from_actor=actor)
        assert atom.payload.content == "HTTP chat materialized content"
        assert atom.meta.provenance.source_agent_id == actor.agent_id
        assert stack.short_term.get(workspace, task.topic_id) is None
        (extraction,) = stack.extractor.inputs
        assert "当前一轮不应进入生成背景" not in extraction["transcript"]
        assert extraction["metadata"]["write_content"] == "保留失败轮的写入"
        assert cpu.finished is True
        assert [
            event.event_type
            for event in stack.events.events
            if event.event_type == RuntimeEventType.MEMORY_TASK_COMPLETED
        ] == [RuntimeEventType.MEMORY_TASK_COMPLETED]
    finally:
        stack.vector_store.write_allowed.set()
        runtime.close()
        access.gateway.close()
        access.gateway.revoke_all_contexts()
