"""Phase 3F Patchouli GatewayDecision 消费契约测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity
from hivememory.core.mtp.exceptions import AliasNotFoundError
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.core.protocol.models import AgentRunResult, RetrievalResponse
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionQueue,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from tests.helpers.workspace import make_identity_scope


def _decision(
    *,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    top_k: int = 7,
) -> GatewayDecision:
    return GatewayDecision(
        target_topic_id="topic-1",
        rewritten_query="保持原查询",
        search_keywords=("gateway",),
        memory_write_signal=MemoryWriteSignal.WRITE,
        retrieval_plan=RetrievalPlan(mode=mode, top_k=top_k),
        intent_type=IntentType.RAG,
    )


def _prepare_bus() -> tuple[PatchouliBus, AsyncMock, AsyncMock]:
    bus = PatchouliBus()
    retrieve = AsyncMock(return_value=RetrievalResponse())
    submit = AsyncMock(return_value="topic-1")
    bus.register(
        PatchouliLocalRoutes.GET_AGENT_PROFILE,
        AsyncMock(return_value=OMNI_DOLL_PROFILE),
    )
    bus.register(
        PatchouliLocalRoutes.TOPIC_PREPARE,
        AsyncMock(return_value="topic-1"),
    )
    bus.register(
        PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
        AsyncMock(return_value=[]),
    )
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=None))
    bus.register(PatchouliLocalRoutes.MEMORY_RETRIEVE, retrieve)
    bus.register(
        PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH,
        AsyncMock(return_value=True),
    )
    bus.register(PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION, submit)
    return bus, retrieve, submit


def _service(bus: PatchouliBus, submit: AsyncMock) -> PatchouliService:
    return PatchouliService(
        bus,
        interaction_queue=InteractionSubmissionQueue(submit),
    )


@pytest.mark.asyncio
async def test_prepare_explicit_missing_profile_fails_before_topic_creation() -> None:
    bus = PatchouliBus()
    failure = AliasNotFoundError(
        message_key="mtp.call.profile_not_found",
        params={"agent_alias": "missing_doll"},
    )
    get_profile = AsyncMock(side_effect=failure)
    prepare_topic = AsyncMock(return_value="should-not-run")
    bus.register(PatchouliLocalRoutes.GET_AGENT_PROFILE, get_profile)
    bus.register(PatchouliLocalRoutes.TOPIC_PREPARE, prepare_topic)

    identity_scope = make_identity_scope(
        user_id="u1",
        agent_id="missing_doll",
    )
    with pytest.raises(AliasNotFoundError) as exc_info:
        await _service(bus, AsyncMock(return_value="topic-1")).prepare_agent_run(
            "hello",
            identity_scope=identity_scope,
            interaction_id="interaction-test",
            gateway_decision=_decision(),
        )

    assert exc_info.value is failure
    get_profile.assert_awaited_once_with(
        "missing_doll",
        identity_scope=identity_scope,
    )
    prepare_topic.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_stores_decision_and_derives_retrieval_request() -> None:
    bus, retrieve, _submit = _prepare_bus()
    decision = _decision(top_k=9)

    prepared = await _service(bus, _submit).prepare_agent_run(
        "原问题",
        identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        interaction_id="interaction-test",
        gateway_decision=decision,
    )

    assert prepared.gateway_decision is decision
    request = retrieve.await_args.args[0]
    assert request.semantic_query == "保持原查询"
    assert request.keywords == ["gateway"]
    assert request.top_k == 9
    assert request.identity == Identity(user_id="u1")


@pytest.mark.asyncio
async def test_prepare_skips_retrieval_for_simple_chat_decision() -> None:
    bus, retrieve, _submit = _prepare_bus()
    decision = _decision(mode=RetrievalMode.SKIP, top_k=0).model_copy(
        update={
            "intent_type": IntentType.CHAT,
            "memory_write_signal": MemoryWriteSignal.SKIP,
        }
    )

    prepared = await _service(bus, _submit).prepare_agent_run(
        "你好",
        identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        interaction_id="interaction-test",
        gateway_decision=decision,
    )

    retrieve.assert_not_awaited()
    assert prepared.agent_run_context.retrieval_result.is_empty()


@pytest.mark.asyncio
async def test_finalize_uses_saved_decision_instead_of_legacy_gaze() -> None:
    bus, _retrieve, submit = _prepare_bus()
    decision = _decision()
    queue = InteractionSubmissionQueue(submit)
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = await service.prepare_agent_run(
        "原问题",
        identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        interaction_id="interaction-test",
        gateway_decision=decision,
    )

    try:
        await queue.start()
        await service.finalize_agent_run(
            prepared,
            AgentRunResult(final_text="回答"),
        )
    finally:
        await queue.stop()

    payload = submit.await_args.args[0]
    assert payload.rewritten_query == decision.rewritten_query
    assert payload.worth_saving is True
    assert payload.assistant_final_text == "回答"
    assert submit.await_args.kwargs["interaction_id"] == prepared.interaction_id


@pytest.mark.asyncio
async def test_retrieval_boundary_rejects_missing_scope_even_when_skipped() -> None:
    """防止 SKIP 分支绕过 scope 校验并形成内部默认 Workspace 先例。"""
    bus, _retrieve, submit = _prepare_bus()
    service = _service(bus, submit)

    with pytest.raises(ScopeRequiredError):
        await service.retrieve_for_decision(
            _decision(mode=RetrievalMode.SKIP, top_k=0),
            identity_scope=None,
        )
