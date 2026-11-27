"""交互提交队列在真实感知链路中的集成测试：retry 与折叠。

测试驱动 InteractionSubmissionQueue、真实 InteractionSubmissionHandler 与
PerceptionFamiliar（Engine + WorkingSet + Store）。retry 在调用感知之前注入确定性
的瞬态错误；折叠由折叠摘要写作者的替身控制何时返回。
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
from hivememory.config.patchouli import PerceptionEngineConfig
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.memory_library.adapters.short_term import JsonlShortTermStorage
from hivememory.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.contracts.errors import TopicBusyError
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
    topic_ordering_key,
)
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
    fold_summary_writer=None,
) -> PerceptionFamiliar:
    """真实感知编排；未指定引擎时使用不触发折叠的默认阈值。"""
    return PerceptionFamiliar(
        engine=engine
        or MemoryPerceptionEngine(config=PerceptionEngineConfig(fold_token_threshold=999999)),
        fold_summary_writer=fold_summary_writer or Mock(),
        progress_writer=None,
        store=store,
        working_set=TopicWorkingSet(max_resident=4),
        bus=Mock(request=AsyncMock(return_value=None)),
        config=SimpleNamespace(settle_after_seconds=900),
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
async def test_interaction_retry_preserves_workspace_and_applies_block_once(tmp_path):
    """捕获 retry 丢失 scope、写错 Workspace 或重复写入 block 的缺陷。"""
    store = ShortTermMemoryStore(JsonlShortTermStorage(tmp_path / "topics"))
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
            raise TopicBusyError("retry once")
        return await familiar.receive_interaction(
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
    assert [block.user_message for block in stored.blocks] == ["retry question"]
    assert store.list_by_workspace(other_scope.workspace_identity) == []


@pytest.mark.asyncio
async def test_fold_runs_after_apply_and_next_same_topic_submission_follows_it(
    tmp_path,
) -> None:
    """交互不等折叠即完成；同一话题的下一次提交在折叠写回之后才写入话题。"""
    store = ShortTermMemoryStore(JsonlShortTermStorage(tmp_path / "topics"))
    summary_started = asyncio.Event()
    release_summary = asyncio.Event()

    summary_calls = 0

    async def slow_summary(blocks_to_fold, **_context):
        nonlocal summary_calls
        summary_calls += 1
        if summary_calls > 1:
            # 第三次提交触发的折叠停在摘要上，由队列停止时取消，不改动话题。
            await asyncio.Event().wait()
        summary_started.set()
        await release_summary.wait()
        return "folded-summary"

    writer = Mock()
    writer.generate_summary = AsyncMock(side_effect=slow_summary)
    engine = MemoryPerceptionEngine(
        config=PerceptionEngineConfig(fold_token_threshold=1, fold_retain_recent_blocks=1),
    )
    familiar = _build_familiar(store, engine=engine, fold_summary_writer=writer)
    scope = make_identity_scope(user_id="u1", agent_id="a1")
    topic_id = await familiar.receive_interaction(
        _payload("first"),
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
        interaction_id="i-first",
    )
    queue = InteractionSubmissionQueue(familiar.receive_interaction, maintenance=familiar)

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
        await asyncio.wait_for(summary_started.wait(), timeout=2)
        third = await queue.submit(submission("i-third", "third"))
        release_summary.set()
        third_outcome = await queue.wait(third, timeout=2)
    finally:
        release_summary.set()
        await queue.stop()

    assert second_outcome.state is WorkState.SUCCEEDED
    assert third_outcome.state is WorkState.SUCCEEDED
    topic = store.get(scope.workspace_identity, topic_id)
    assert topic.fold_summary == "folded-summary"
    assert [block.user_message for block in topic.blocks] == ["second", "third"]
