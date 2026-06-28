from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    OMNI_DOLL_PROFILE,
    PayloadLayer,
    TurnEvent,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, WriteFocus
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    AnalyzeAndRetrieveResult,
    EyeGazeResult,
    RetrievalResponse,
)
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.application import (
    AgentProfileManagementService,
    MemoryManagementService,
    MemoryTaskManagementService,
    ModelReadinessService,
    TopicManagementService,
)
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.models import PreparedAgentRun, StreamPrelude
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from hivememory.system.contracts.events import GlobalEvents
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


def _memory_atom() -> MemoryAtom:
    return MemoryAtom(
        meta=MetaData(source_agent_id="agent-1", user_id="u1"),
        index=IndexLayer(
            title="memory title",
            summary="memory summary text",
            tags=["test"],
            memory_type=MemoryType.FACT,
            alias="memory_alias",
        ),
        payload=PayloadLayer(content="memory content"),
    )


def _gaze_result(identity: Identity, *, target_topic="topic_existing") -> EyeGazeResult:
    return EyeGazeResult(
        intent=GatewayIntent.RAG,
        rewritten_query="rewritten query",
        search_keywords=["memory"],
        worth_saving=True,
        raw_query="raw query",
        identity=identity,
        target_topic=target_topic,
        new_topic_title="new title" if target_topic == "NEW_TOPIC" else None,
        new_topic_summary="new summary" if target_topic == "NEW_TOPIC" else None,
    )


def _prepared_run(
    *,
    identity: Identity | None = None,
    retrieval_result: RetrievalResponse | None = None,
) -> PreparedAgentRun:
    identity = identity or Identity(user_id="u1", agent_id="omni_doll")
    return PreparedAgentRun(
        agent_run_context=AgentRunContext(
            identity=identity,
            topic_id="topic_1",
            user_message="remember this",
            topic_context=None,
            retrieval_result=retrieval_result or RetrievalResponse(),
            agent_profile=OMNI_DOLL_PROFILE,
            storage_available=True,
        ),
        gaze_result=_gaze_result(identity),
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_topics=[],
            memory_refs=[],
        ),
    )


def _wire_public_bridge(local_bus: PatchouliBus, global_bus: GlobalSystemBus):
    service = PatchouliService(bus=local_bus, eye=MagicMock())
    public_api = PatchouliPublicApi(
        chat=service,
        memory=MemoryManagementService(bus=local_bus),
        memory_tasks=MemoryTaskManagementService(bus=local_bus),
        agent_profiles=AgentProfileManagementService(bus=local_bus),
        topics=TopicManagementService(bus=local_bus),
        readiness=ModelReadinessService(local_bus),
    )
    bridge = PatchouliBridge(
        local_bus=local_bus,
        public_api=public_api,
        global_bus=global_bus,
    )
    return service, bridge


@pytest.mark.asyncio
async def test_public_prepare_agent_run_composes_local_primitives_through_bridge():
    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()
    identity = Identity(user_id="u1", agent_id="omni_doll", session_id="s1")
    retrieval_result = RetrievalResponse(
        memories=[_memory_atom()],
    )
    calls: list[str] = []

    def register(route, result):
        async def handler(*args, **kwargs):
            calls.append(route)
            return result

        local_bus.register(route, handler)

    register(PatchouliLocalRoutes.GET_AGENT_PROFILE, OMNI_DOLL_PROFILE)
    register(PatchouliLocalRoutes.TOPIC_LIST_ACTIVE, [])
    register(PatchouliLocalRoutes.GATEWAY_GAZE, _gaze_result(identity))
    register(PatchouliLocalRoutes.TOPIC_PREPARE, "topic_1")
    register(PatchouliLocalRoutes.TOPIC_GET, None)
    register(PatchouliLocalRoutes.MEMORY_RETRIEVE, retrieval_result)
    register(PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH, True)
    _, bridge = _wire_public_bridge(local_bus, global_bus)
    bridge.mount()

    prepared = await global_bus.request(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        user_message="remember this",
        user_id="u1",
        agent_id="omni_doll",
        session_id="s1",
    )

    assert prepared.topic_id == "topic_1"
    assert prepared.agent_run_context.retrieval_result is retrieval_result
    assert "memory summary text" in prepared.agent_run_context.memory_context
    assert "memory content" not in prepared.agent_run_context.memory_context
    assert prepared.agent_run_context.agent_profile is OMNI_DOLL_PROFILE
    assert prepared.agent_run_context.storage_available is True
    assert prepared.stream_prelude.memory_refs[0]["alias"] == "memory_alias"
    assert calls == [
        PatchouliLocalRoutes.GET_AGENT_PROFILE,
        PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
        PatchouliLocalRoutes.GATEWAY_GAZE,
        PatchouliLocalRoutes.TOPIC_PREPARE,
        PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
        PatchouliLocalRoutes.TOPIC_GET,
        PatchouliLocalRoutes.MEMORY_RETRIEVE,
        PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH,
    ]


@pytest.mark.asyncio
async def test_public_prepare_agent_run_cleans_new_topic_when_later_primitive_fails():
    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()
    identity = Identity(user_id="u1", agent_id="omni_doll")
    discard = AsyncMock(return_value=True)

    local_bus.register(PatchouliLocalRoutes.GET_AGENT_PROFILE, AsyncMock(return_value=OMNI_DOLL_PROFILE))
    local_bus.register(PatchouliLocalRoutes.TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))
    local_bus.register(
        PatchouliLocalRoutes.GATEWAY_GAZE,
        AsyncMock(return_value=_gaze_result(identity, target_topic="NEW_TOPIC")),
    )
    local_bus.register(PatchouliLocalRoutes.TOPIC_PREPARE, AsyncMock(return_value="topic_new"))
    local_bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(side_effect=RuntimeError("topic read failed")))
    local_bus.register(PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, discard)
    _, bridge = _wire_public_bridge(local_bus, global_bus)
    bridge.mount()

    with pytest.raises(RuntimeError, match="topic read failed"):
        await global_bus.request(
            GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
            user_message="start a new topic",
            user_id="u1",
        )

    discard.assert_awaited_once_with("topic_new")


@pytest.mark.asyncio
async def test_public_finalize_agent_run_submits_ingestion_active_generation_and_hits():
    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()
    memory = _memory_atom()
    materialize_task = PendingAtomMaterializeTask(
        pending_alias="draft_memory",
        intent_id="intent_1",
        source_verb="WRITE",
        identity=Identity(user_id="u1", agent_id="omni_doll"),
        focus=WriteFocus(content="remember this"),
    )
    submit_interaction = AsyncMock(return_value="topic_1")
    submit_active = AsyncMock(return_value=[MagicMock(task_id="memtask_1")])
    record_hit = AsyncMock(return_value=None)
    local_bus.register(PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION, submit_interaction)
    local_bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, submit_active)
    local_bus.register(PatchouliLocalRoutes.MEMORY_RECORD_HIT, record_hit)
    _, bridge = _wire_public_bridge(local_bus, global_bus)
    bridge.mount()

    result = await global_bus.request(
        GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN,
        prepared_run=_prepared_run(
            retrieval_result=RetrievalResponse(memories=[memory, memory]),
        ),
        loop_result=AgentRunResult(
            final_text="done",
            turn_events=[
                TurnEvent(
                    kind="tool_call",
                    sequence=0,
                    role="assistant",
                    content="search",
                    action_id="a1",
                    tool_kind="SEARCH",
                    tool_name="memory.search",
                    tool_args={"query": "memory"},
                ),
                TurnEvent(
                    kind="tool_result",
                    sequence=1,
                    role="user",
                    content="result",
                    action_id="a1",
                    tool_kind="SEARCH",
                    tool_name="memory.search",
                    status="success",
                ),
            ],
            materialize_tasks=[materialize_task],
        ),
    )

    assert [task.task_id for task in result] == ["memtask_1"]
    submit_interaction.assert_awaited_once()
    payload = submit_interaction.await_args.args[0]
    assert submit_interaction.await_args.kwargs == {"target_topic_id": "topic_1"}
    assert payload.materialize_tasks == [materialize_task]
    assert payload.mtp_traces[0].action == "SEARCH"
    submit_active.assert_awaited_once_with([materialize_task], topic_id="topic_1")
    record_hit.assert_awaited_once_with(memory.id, source="retrieval.finalize")


@pytest.mark.asyncio
async def test_public_passive_analyze_and_retrieve_composes_gateway_and_retrieval_primitives():
    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()
    identity = Identity(user_id="u1", agent_id="observer")
    gaze = AsyncMock(return_value=_gaze_result(identity))
    retrieve = AsyncMock(
        return_value=RetrievalResponse(
            memories=[_memory_atom()],
        )
    )
    local_bus.register(PatchouliLocalRoutes.GATEWAY_GAZE, gaze)
    local_bus.register(PatchouliLocalRoutes.MEMORY_RETRIEVE, retrieve)
    service, bridge = _wire_public_bridge(local_bus, global_bus)
    service._eye.gaze = AsyncMock(side_effect=AssertionError("direct eye call leaked"))
    bridge.mount()

    result = await global_bus.request(
        PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE,
        query="raw query",
        identity=identity,
        topic_snapshots=["topic"],
        enable_retrieval=True,
    )

    assert isinstance(result, AnalyzeAndRetrieveResult)
    assert result.gaze_result.rewritten_query == "rewritten query"
    assert result.retrieval_result.memories[0].get_alias() == "memory_alias"
    gaze.assert_awaited_once_with(
        query="raw query",
        topic_snapshots=["topic"],
        identity=identity,
    )
    retrieve.assert_awaited_once()
    service._eye.gaze.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_public_memory_task_route_reaches_application_service_then_local_primitive():
    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()
    list_tasks = AsyncMock(return_value=["task_1"])
    get_task = AsyncMock(return_value="task_1")
    cancel_task = AsyncMock(return_value=True)
    local_bus.register(PatchouliLocalRoutes.MEMORY_TASK_LIST, list_tasks)
    local_bus.register(PatchouliLocalRoutes.MEMORY_TASK_GET, get_task)
    local_bus.register(PatchouliLocalRoutes.MEMORY_TASK_CANCEL, cancel_task)
    _, bridge = _wire_public_bridge(local_bus, global_bus)
    bridge.mount()

    assert await global_bus.request(PatchouliRoutes.MEMORY_TASK_LIST) == ["task_1"]
    assert await global_bus.request(PatchouliRoutes.MEMORY_TASK_GET, "task_1") == "task_1"
    assert await global_bus.request(PatchouliRoutes.MEMORY_TASK_CANCEL, "task_1") is True

    list_tasks.assert_awaited_once_with()
    get_task.assert_awaited_once_with("task_1")
    cancel_task.assert_awaited_once_with("task_1")


@pytest.mark.asyncio
async def test_bridge_forwards_pending_atom_events_and_unmounts_public_routes():
    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()
    subscriber = AsyncMock()
    global_bus.subscribe(GlobalEvents.PENDING_ATOM_FAILED, subscriber)
    _, bridge = _wire_public_bridge(local_bus, global_bus)

    bridge.mount()
    assert PatchouliRoutes.PREPARE_AGENT_RUN in global_bus.list_routes()
    await local_bus.publish(
        PatchouliLocalEvents.PENDING_ATOM_FAILED,
        pending_alias="draft_memory",
    )
    subscriber.assert_awaited_once_with(pending_alias="draft_memory")

    bridge.unmount()
    assert PatchouliRoutes.PREPARE_AGENT_RUN not in global_bus.list_routes()
    subscriber.reset_mock()
    await local_bus.publish(
        PatchouliLocalEvents.PENDING_ATOM_FAILED,
        pending_alias="draft_memory",
    )
    subscriber.assert_not_awaited()
