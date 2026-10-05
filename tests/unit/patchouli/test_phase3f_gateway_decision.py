"""Phase 3F Patchouli GatewayDecision 消费契约测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.core.errors import ScopeRequiredError, WorkspaceMismatchError
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
    assert retrieve.await_args.kwargs["top_k"] == 9
    assert request.from_actor == ActorIdentity(user_id="u1", agent_id="omni_doll")
    assert request.belong_to.owner_user_id == "u1"


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
        await service.finalize_agent_run(
            prepared,
            payload,
            identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        )
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
        await service.prepare_agent_run(
            gateway_decision=_decision(mode=RetrievalMode.SKIP, top_k=0),
            identity_scope=None,
            interaction_id="invalid-skipped",
        )


@pytest.mark.asyncio
async def test_finalize_and_cleanup_reject_foreign_prepared_workspace() -> None:
    """越域 prepare 结果不得提交或删除话题，补偿入口也必须守住归属边界。"""
    bus, _retrieve, submit = _prepare_bus()
    queue = InteractionSubmissionQueue(submit)
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = await service.prepare_agent_run(
        identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        interaction_id="foreign-finalize",
        gateway_decision=_decision().model_copy(update={"target_topic_id": "NEW_TOPIC"}),
    )
    scope = make_identity_scope(user_id="u1", workspace_id="isolation_workspace")
    discarded = []

    async def discard(topic_id, *, belong_to):
        discarded.append((topic_id, belong_to))
        return True

    bus.register(PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, discard)
    with pytest.raises(WorkspaceMismatchError, match="workspace.mismatch"):
        await service.finalize_agent_run(
            prepared, InteractionPayload(user_message="问题"), identity_scope=scope
        )

    assert await queue.is_accepted(prepared.interaction_id) is False
    assert await service.cleanup_prepared_agent_run(prepared, identity_scope=scope) is False
    assert discarded == []


@pytest.mark.asyncio
async def test_finalize_uses_current_submission_actor_instead_of_prepare_actor() -> None:
    """同一归属的提交必须使用当前阶段发起者，prepare 结果不冻结执行者。"""
    bus, _retrieve, _submit = _prepare_bus()
    applied = []

    async def apply(payload, *, belong_to, from_actor, target_topic_id, **_kwargs):
        applied.append((payload.user_message, belong_to, from_actor))
        return target_topic_id

    queue = InteractionSubmissionQueue(apply)
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = await service.prepare_agent_run(
        identity_scope=make_identity_scope(user_id="u1", agent_id="prepare-actor"),
        interaction_id="current-finalize-actor",
        gateway_decision=_decision(),
    )
    scope = make_identity_scope(user_id="u1", agent_id="submit-actor")
    try:
        await queue.start()
        result = await service.finalize_agent_run(
            prepared, InteractionPayload(user_message="问题"), identity_scope=scope
        )
    finally:
        await queue.stop()

    assert result == []
    assert applied == [
        ("问题", prepared.belong_to, ActorIdentity(user_id="u1", agent_id="submit-actor"))
    ]


@pytest.mark.asyncio
async def test_finalize_requires_explicit_stage_scope() -> None:
    """公开 finalize 缺少本阶段 scope 时在提交前拒绝。"""
    bus, _retrieve, submit = _prepare_bus()
    queue = InteractionSubmissionQueue(submit)
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = await service.prepare_agent_run(
        identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        interaction_id="missing-finalize-scope",
        gateway_decision=_decision(),
    )
    with pytest.raises(TypeError, match="identity_scope"):
        await service.finalize_agent_run(prepared, InteractionPayload(user_message="问题"))
    assert await queue.is_accepted(prepared.interaction_id) is False
