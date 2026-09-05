"""Workspace scope 在真实 Interaction submission retry 链路中的集成测试。

测试驱动 InteractionSubmissionQueue、真实 InteractionSubmissionHandler、
PerceptionFamiliar（Engine + WorkingSet + Store）与真实 Journal；只在首次
attempt 的外部提交边界注入一次确定性的瞬态错误。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
    TransientInteractionSubmissionError,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.system.runtime.work_queue import QueuePolicy, WorkState
from tests.helpers.workspace import make_identity_scope


def _payload() -> InteractionPayload:
    """构造不可变的结构化交互载荷。"""
    return InteractionPayload(
        user_message="retry question",
        assistant_final_text="retry answer",
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content="retry answer",
            )
        ],
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
    journal = InMemoryInteractionApplyJournal()
    familiar = PerceptionFamiliar(
        engine=MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=999999),
            relay_controller=Mock(),
        ),
        store=store,
        working_set=TopicWorkingSet(max_resident=4),
        bus=Mock(request=AsyncMock(return_value=None)),
        config=SimpleNamespace(idle_timeout_seconds=900),
        interaction_journal=journal,
    )

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
        identity_scope,
        target_topic_id,
        interaction_id,
    ):
        attempts.append((len(attempts) + 1, identity_scope))
        if len(attempts) == 1:
            raise TransientInteractionSubmissionError("retry once")
        return await familiar.apply_interaction(
            payload,
            identity_scope=identity_scope,
            target_topic_id=target_topic_id,
            interaction_id=interaction_id,
        )

    queue = InteractionSubmissionQueue(
        submit_with_one_transient_failure,
        policy=_queue_policy(),
    )
    submission = InteractionSubmission(
        identity_scope=scope,
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
    assert all(attempt_scope == scope for _, attempt_scope in attempts)

    stored = store.get(scope, outcome.topic_id)
    assert stored.topic_id == outcome.topic_id
    assert [block.user_query for block in stored.blocks] == ["retry question"]
    assert store.list_by_workspace(other_scope) == []
