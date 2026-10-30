"""交互提交队列在真实感知链路中的集成测试：retry 与折叠。

测试驱动 InteractionSubmissionQueue、真实 InteractionSubmissionHandler 与
PerceptionFamiliar（Engine + WorkingSet + Store）。retry 在调用感知之前注入确定性
的瞬态错误；折叠由 Relay 边界控制何时返回。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.components.work_queue import (
    QueuePolicy,
    WorkState,
)
from hivememory.config.patchouli import SemanticFlowPerceptionConfig
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
    TransientInteractionSubmissionError,
    topic_ordering_key,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from tests.helpers.sessions import make_interaction_payload
from tests.helpers.workspace import make_identity_scope


def _payload(message: str = "retry question") -> InteractionPayload:
    """构造不可变的结构化交互载荷。"""
    return make_interaction_payload(message, f"answer:{message}")


def _build_familiar(
    store: ShortTermMemoryStore,
    *,
    engine: MemoryPerceptionEngine | None = None,
) -> PerceptionFamiliar:
    """真实感知编排；未指定引擎时使用不触发折叠的默认阈值。"""
    return PerceptionFamiliar(
        engine=engine
        or MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=999999),
            relay_controller=Mock(),
        ),
        store=store,
        working_set=TopicWorkingSet(max_resident=4),
        bus=Mock(request=AsyncMock(return_value=None)),
        config=SimpleNamespace(idle_timeout_seconds=900),
    )


def _queue_policy() -> QueuePolicy:
    """为 retry 场景提供短而确定的队列策略。"""
    return QueuePolicy(
        capacity=4,
        max_concurrency=1,
        ordered_by_key=True,
        cancellable=False,
        timeout_seconds=1,
        max_attempts=2,
        terminal_retention=8,
    )


@pytest.mark.asyncio
async def test_interaction_retry_preserves_workspace_and_applies_block_once():
    """捕获 retry 丢失 scope、写错 Workspace 或重复写入 block 的缺陷。"""
    store = ShortTermMemoryStore()
    familiar = _build_familiar(store)

    scope = make_identity_scope(
        user_id="u-retry",
        agent_id="agent-a",
        workspace_id="isolation_workspace",
    )
    other_scope = make_identity_scope(
        user_id="u-retry",
        agent_id="agent-a",
        workspace_id="main_workspace",
    )
    attempts: list[tuple[int, object]] = []

    async def submit_with_one_transient_failure(
        payload,
        *,
        belong_to,
        from_actor,
        target_topic_id,
        interaction_id,
        asset_refs=(),
        new_topic_title=None,
        new_topic_summary=None,
    ):
        attempts.append((len(attempts) + 1, belong_to))
        if len(attempts) == 1:
            raise TransientInteractionSubmissionError("retry once")
        return await familiar.apply_interaction(
            payload,
            belong_to=belong_to,
            from_actor=from_actor,
            target_topic_id=target_topic_id,
            interaction_id=interaction_id,
        )

    queue = InteractionSubmissionQueue(
        submit_with_one_transient_failure,
        policy=_queue_policy(),
    )
    submission = InteractionSubmission(
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
        interaction_id="interaction-workspace-retry",
        payload=_payload(),
        requested_topic_id="NEW_TOPIC",
        ordering_key="topic:workspace-retry",
        origin="active_chat",
        correlation={"turn_id": "turn-workspace-retry"},
    )

    try:
        await queue.start()
        receipt = await queue.submit(submission)
        outcome = await queue.wait(receipt, timeout=2)
    finally:
        await queue.stop()

    assert outcome.state is WorkState.SUCCEEDED
    assert isinstance(outcome.topic_id, str)
    assert [attempt for attempt, _ in attempts] == [1, 2]
    assert [attempt_scope for _, attempt_scope in attempts] == [scope.workspace_identity] * 2

    stored = store.get(scope.workspace_identity, outcome.topic_id)
    assert stored.topic_id == outcome.topic_id
    assert [block.user_query for block in stored.blocks] == ["retry question"]
    assert store.list_by_workspace(other_scope.workspace_identity) == []


@pytest.mark.asyncio
async def test_compaction_runs_after_apply_and_next_same_topic_submission_follows_it() -> None:
    """交互不等折叠即完成；同一话题的下一次提交在折叠写回之后才写入话题。"""
    store = ShortTermMemoryStore()
    relay_started = asyncio.Event()
    release_relay = asyncio.Event()

    summary_calls = 0

    async def slow_summary(blocks_to_fold, previous_summary=None):
        nonlocal summary_calls
        summary_calls += 1
        if summary_calls > 1:
            # 第三次提交触发的折叠停在摘要上，由队列停止时取消，不改动话题。
            await asyncio.Event().wait()
        relay_started.set()
        await release_relay.wait()
        return "folded-summary"

    relay = Mock()
    relay.generate_summary = AsyncMock(side_effect=slow_summary)
    engine = MemoryPerceptionEngine(
        config=SemanticFlowPerceptionConfig(fold_token_threshold=1, fold_retain_recent_blocks=1),
        relay_controller=relay,
    )
    familiar = _build_familiar(store, engine=engine)
    scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.apply_interaction(
        _payload("first"),
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
        interaction_id="i-first",
    )
    queue = InteractionSubmissionQueue(familiar.apply_interaction, compaction=familiar)

    def submission(interaction_id: str, message: str) -> InteractionSubmission:
        return InteractionSubmission(
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            interaction_id=interaction_id,
            payload=_payload(message),
            requested_topic_id=topic_id,
            ordering_key=topic_ordering_key(topic_id),
            origin="import_bus",
            correlation={"turn_id": interaction_id},
        )

    try:
        await queue.start()
        second = await queue.submit(submission("i-second", "second"))
        second_outcome = await queue.wait(second, timeout=2)
        await asyncio.wait_for(relay_started.wait(), timeout=2)
        third = await queue.submit(submission("i-third", "third"))
        release_relay.set()
        third_outcome = await queue.wait(third, timeout=2)
    finally:
        release_relay.set()
        await queue.stop()

    assert second_outcome.state is WorkState.SUCCEEDED
    assert third_outcome.state is WorkState.SUCCEEDED
    topic = store.get(scope.workspace_identity, topic_id)
    assert topic.state_summary == "folded-summary"
    assert [block.user_query for block in topic.blocks] == ["second", "third"]
