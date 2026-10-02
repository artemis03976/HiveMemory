"""Phase 3F Patchouli GatewayDecision 消费契约测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import ActorIdentity
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.core.protocol.models import InteractionPayload
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
    # 检索 backing 的正式返回形状：原子列表（A2 §2.1）。prepare 不再解析
    # Profile——该项已迁到任务进程的 CPU 分配边界。
    retrieve = AsyncMock(return_value=[])
    submit = AsyncMock(return_value="topic-1")
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
    return bus, retrieve, submit


def _service(bus: PatchouliBus, submit: AsyncMock) -> PatchouliService:
    return PatchouliService(
        bus,
        interaction_queue=InteractionSubmissionQueue(submit),
    )


@pytest.mark.asyncio
async def test_prepare_derives_retrieval_request_from_decision() -> None:
    bus, retrieve, _submit = _prepare_bus()
    decision = _decision(top_k=9)

    await _service(bus, _submit).prepare_agent_run(
        identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        interaction_id="interaction-test",
        gateway_decision=decision,
    )

    request = retrieve.await_args.args[0]
    assert request.semantic_query == "保持原查询"
    assert request.keywords == ["gateway"]
    assert request.top_k == 9
    assert request.identity_scope.actor_identity == ActorIdentity(user_id="u1")


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
        identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        interaction_id="interaction-test",
        gateway_decision=decision,
    )

    retrieve.assert_not_awaited()
    assert prepared.retrieval_result.is_empty()


@pytest.mark.asyncio
async def test_finalize_submits_received_payload_with_prepared_identity() -> None:
    """finalize 原样提交封口方交给它的交互记录，关联 ID 取自 prepared run。

    payload 字段（rewritten_query/worth_saving 等）由提交方封口，进程侧
    封口测试负责逐字段守护；这里只守护 Patchouli 不再改写内容。
    """
    bus, _retrieve, submit = _prepare_bus()
    queue = InteractionSubmissionQueue(submit)
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = await service.prepare_agent_run(
        identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        interaction_id="interaction-test",
        gateway_decision=_decision(),
    )
    payload = InteractionPayload(user_message="原问题", assistant_final_text="回答")

    try:
        await queue.start()
        await service.finalize_agent_run(prepared, payload)
    finally:
        await queue.stop()

    # submit 是提交队列的 apply 回调：首参即进入 apply 的交互记录本体
    # （队列在 admission 与 apply 之间会对 payload 留档拷贝，这里断言值相等）。
    applied_payload = submit.await_args.args[0]
    assert applied_payload == payload
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
