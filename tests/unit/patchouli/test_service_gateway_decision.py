"""Patchouli 消费 GatewayDecision 的契约测试。

prepare 与 finalize 由系统以 root scope 调用：话题与归属取 root scope，检索
按单独的检索 scope 的 actor 可见性进行，交互的发起者取显式的参与者。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import ScopeRequiredError, WorkspaceMismatchError
from hivememory.core.models import ActorIdentity
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    IntentType,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionQueue,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from tests.helpers.sessions import make_interaction_payload
from tests.helpers.workspace import make_identity_scope

# 系统阶段的 root scope 与执行者经 memory.search 授权得到的检索 scope。
ROOT_SCOPE = make_identity_scope(user_id="u1", agent_id=SYSTEM_AGENT_ID)
RETRIEVAL_SCOPE = make_identity_scope(user_id="u1", agent_id="omni_doll")
# 交互中的 Agent：finalize 显式传入的参与者。
PARTICIPANT = ActorIdentity(user_id="u1", agent_id="omni_doll")


def _decision(
    *,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    top_k: int = 7,
) -> GatewayDecision:
    return GatewayDecision(
        target_topic_id="topic-1",
        rewritten_query="保持原查询",
        search_keywords=("gateway",),
        retrieval_plan=RetrievalPlan(mode=mode, top_k=top_k),
        intent_type=IntentType.RAG,
    )


def _prepare_bus() -> tuple[PatchouliBus, AsyncMock, AsyncMock]:
    bus = PatchouliBus()
    # 检索 backing 的正式返回形状：原子列表。prepare 不解析
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
    # prepare 的接口不再读取 Topic 内容；未注册该路由，旧读取会直接失败。
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
    """检索请求取自决定；可见性按检索 scope 的执行者，而不是 root scope 的 system。"""
    bus, retrieve, _submit = _prepare_bus()
    decision = _decision(top_k=9)

    await _service(bus, _submit).prepare_agent_run(
        identity_scope=ROOT_SCOPE,
        retrieval_scope=RETRIEVAL_SCOPE,
        interaction_id="interaction-test",
        gateway_decision=decision,
    )

    request = retrieve.await_args.args[0]
    assert request.semantic_query == "保持原查询"
    assert request.keywords == ["gateway"]
    assert retrieve.await_args.kwargs["top_k"] == 9
    assert request.from_actor == ActorIdentity(user_id="u1", agent_id="omni_doll")
    assert request.belong_to == ROOT_SCOPE.workspace_identity


@pytest.mark.asyncio
async def test_prepare_without_retrieval_scope_prepares_topic_without_retrieval() -> None:
    """执行者没有检索许可时不给检索 scope：话题照常准备，决定要求检索也不检索。"""
    bus, retrieve, _submit = _prepare_bus()

    prepared = await _service(bus, _submit).prepare_agent_run(
        identity_scope=ROOT_SCOPE,
        interaction_id="interaction-test",
        gateway_decision=_decision(top_k=9),
    )

    retrieve.assert_not_awaited()
    assert (prepared.topic_id, prepared.memories) == ("topic-1", ())


@pytest.mark.asyncio
async def test_prepare_rejects_retrieval_scope_of_another_workspace_before_topic_preparation():
    """检索 scope 与 root scope 不在同一 Workspace 时拒绝，不准备话题也不检索。"""
    retrieve = AsyncMock(return_value=[])
    prepared_topics = []

    async def prepare_topic(**kwargs):
        prepared_topics.append(kwargs)
        return "topic-1"

    bus = PatchouliBus()
    bus.register(PatchouliLocalRoutes.TOPIC_PREPARE, prepare_topic)
    bus.register(PatchouliLocalRoutes.MEMORY_RETRIEVE, retrieve)

    with pytest.raises(WorkspaceMismatchError):
        await _service(bus, AsyncMock()).prepare_agent_run(
            identity_scope=ROOT_SCOPE,
            retrieval_scope=make_identity_scope(
                user_id="u1", agent_id="omni_doll", workspace_id="isolation_workspace"
            ),
            interaction_id="interaction-test",
            gateway_decision=_decision(),
        )

    assert prepared_topics == []
    retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_skips_retrieval_for_simple_chat_decision() -> None:
    bus, retrieve, _submit = _prepare_bus()
    decision = _decision(mode=RetrievalMode.SKIP, top_k=0).model_copy(
        update={
            "intent_type": IntentType.CHAT,
        }
    )

    prepared = await _service(bus, _submit).prepare_agent_run(
        identity_scope=ROOT_SCOPE,
        retrieval_scope=RETRIEVAL_SCOPE,
        interaction_id="interaction-test",
        gateway_decision=decision,
    )

    retrieve.assert_not_awaited()
    assert prepared.memories == ()


@pytest.mark.asyncio
async def test_finalize_submits_received_payload_with_prepared_identity() -> None:
    """finalize 原样提交封口方交给它的交互记录，关联 ID 取自 prepared run。

    payload 字段由提交方封口，进程侧
    封口测试负责逐字段守护；这里只守护 Patchouli 不再改写内容。
    """
    bus, _retrieve, submit = _prepare_bus()
    queue = InteractionSubmissionQueue(submit)
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = await service.prepare_agent_run(
        identity_scope=ROOT_SCOPE,
        retrieval_scope=RETRIEVAL_SCOPE,
        interaction_id="interaction-test",
        gateway_decision=_decision(),
    )
    payload = make_interaction_payload("原问题", "回答")

    try:
        await queue.start()
        await service.finalize_agent_run(
            prepared, payload, identity_scope=ROOT_SCOPE, participant=PARTICIPANT
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
        identity_scope=ROOT_SCOPE,
        interaction_id="foreign-finalize",
        gateway_decision=_decision().model_copy(update={"target_topic_id": "NEW_TOPIC"}),
    )
    scope = make_identity_scope(
        user_id="u1", agent_id=SYSTEM_AGENT_ID, workspace_id="isolation_workspace"
    )
    discarded = []

    async def discard(topic_id, *, belong_to):
        discarded.append((topic_id, belong_to))
        return True

    bus.register(PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, discard)
    with pytest.raises(WorkspaceMismatchError, match="workspace.mismatch"):
        await service.finalize_agent_run(
            prepared,
            make_interaction_payload("问题"),
            identity_scope=scope,
            participant=PARTICIPANT,
        )

    assert await queue.is_accepted(prepared.interaction_id) is False
    assert await service.cleanup_prepared_agent_run(prepared, identity_scope=scope) is False
    assert discarded == []


@pytest.mark.asyncio
async def test_finalize_records_explicit_participant_instead_of_scope_actor() -> None:
    """交互由 system 以 root scope 提交：发起者是显式的参与者，归属取 prepare 结果。

    prepare 结果不冻结任何身份，发起者也不取自 scope 的 actor。
    """
    bus, _retrieve, _submit = _prepare_bus()
    applied = []

    async def apply(payload, *, belong_to, from_actor, target_topic_id, **_kwargs):
        applied.append((payload.user_message, belong_to, from_actor))
        return target_topic_id

    queue = InteractionSubmissionQueue(apply)
    service = PatchouliService(bus, interaction_queue=queue)
    prepared = await service.prepare_agent_run(
        identity_scope=ROOT_SCOPE,
        retrieval_scope=make_identity_scope(user_id="u1", agent_id="prepare-actor"),
        interaction_id="current-finalize-actor",
        gateway_decision=_decision(),
    )
    try:
        await queue.start()
        result = await service.finalize_agent_run(
            prepared,
            make_interaction_payload("问题"),
            identity_scope=ROOT_SCOPE,
            participant=ActorIdentity(user_id="u1", agent_id="submit-actor"),
        )
    finally:
        await queue.stop()

    assert result is None
    assert applied == [
        ("问题", prepared.belong_to, ActorIdentity(user_id="u1", agent_id="submit-actor"))
    ]


@pytest.mark.asyncio
async def test_finalize_unknown_admission_preserves_cause_and_uncertainty() -> None:
    """队列提交异常不证明没有接纳，finalize 只记录阶段与稳定的未知原因。"""
    from hivememory.patchouli.contracts.errors import ActiveInteractionFinalizationError

    bus, _, submit = _prepare_bus()
    service = _service(bus, submit)
    prepared = await service.prepare_agent_run(
        identity_scope=ROOT_SCOPE, interaction_id="uncertain", gateway_decision=_decision()
    )
    failure = ValueError("private admission detail")
    queue = AsyncMock(spec=InteractionSubmissionQueue)
    queue.submit.side_effect = failure
    finalizer = PatchouliService(bus, interaction_queue=queue)
    with pytest.raises(ActiveInteractionFinalizationError) as caught:
        await finalizer.finalize_agent_run(
            prepared,
            make_interaction_payload("q"),
            identity_scope=ROOT_SCOPE,
            participant=PARTICIPANT,
        )
    assert caught.value.stage == "interaction_admission"
    assert caught.value.reason == "admission_unknown"
    assert caught.value.error_class == "ValueError"
    assert caught.value.__cause__ is failure
    assert "private admission detail" not in str(caught.value)
