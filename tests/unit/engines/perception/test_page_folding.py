from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from hivememory.core.models import (
    BufferState,
    Identity,
    LogicalBlock,
    TraceItem,
    TurnEvent,
    TurnRecord,
)
from hivememory.core.protocol import InteractionPayload
from hivememory.engines.perception.models import TriggerReason
from hivememory.patchouli.services.topic_buffer import TopicBufferService
from hivememory.system.config import SemanticFlowPerceptionConfig
from tests.helpers.perception import build_perception_stack
from tests.helpers.workspace import make_identity_scope


def _make_identity():
    return Identity(user_id="u1", agent_id="a1")


def _identity_scope(identity=None):
    return make_identity_scope(actor_identity=identity or _make_identity())


def _make_payload(user_msg="hello", assistant_msg="world", identity=None, traces=None):
    identity = identity or _make_identity()
    return InteractionPayload(
        user_message=user_msg,
        assistant_final_text=assistant_msg,
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=assistant_msg,
            )
        ],
        mtp_traces=traces or [],
    )


def _make_layer(
    *,
    fold_token_threshold=999999,
    fold_retain_recent_blocks=2,
    relay=None,
    store=None,
):
    config = SemanticFlowPerceptionConfig(
        fold_token_threshold=fold_token_threshold,
        fold_retain_recent_blocks=fold_retain_recent_blocks,
    )
    layer, store, service = build_perception_stack(
        store=store, relay=relay, config=config
    )
    return layer, store, service


def _make_service_with_content(relay, *, block_count=10, tokens_per_block=20):
    """构造带若干 block 的 Topic 与对应服务，供 compact 行为测试使用。"""
    layer, store, service = build_perception_stack(relay=relay)
    scope = _identity_scope()
    topic = store.create(scope)
    blocks = tuple(
        LogicalBlock(
            turn=TurnRecord(
                identity=scope.actor_identity,
                user_query=f"question {i}",
                assistant_final_text=f"answer {i}",
            ),
            total_tokens=tokens_per_block,
        )
        for i in range(block_count)
    )
    store.put(
        topic.model_copy(update={"blocks": blocks, "total_tokens": tokens_per_block * block_count})
    )
    return service, store, scope, topic.topic_id


class TestBlockTokenComputation:
    @pytest.mark.asyncio
    async def test_block_total_tokens_computed(self):
        layer, store, _ = _make_layer()

        topic_id, settle_payload = await layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("What is Python?", "Python is a language"),
            identity_scope=_identity_scope(),
        )

        assert settle_payload is None
        topic_data = store.get(_identity_scope(), topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 1
        assert topic_data.blocks[0].total_tokens > 0
        assert topic_data.total_tokens > 0

    @pytest.mark.asyncio
    async def test_block_tokens_include_traces(self):
        layer, store, _ = _make_layer()

        topic_id, settle_payload = await layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("q", "a", traces=[
                TraceItem(action="SEARCH", query="how to sort a list"),
                TraceItem(action="READ", target="my_notes_alias"),
            ]),
            identity_scope=_identity_scope(),
        )

        assert settle_payload is None
        topic_data = store.get(_identity_scope(), topic_id, touch=False)
        assert topic_data is not None
        with_traces = topic_data.blocks[0].total_tokens

        # 对照：同文本无 traces 的 block，token 数应更小（验证 traces 被计入）
        topic_id2, _ = await layer.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("q", "a"),
            identity_scope=_identity_scope(),
        )
        without_traces_data = store.get(_identity_scope(), topic_id2, touch=False)
        assert without_traces_data is not None
        assert with_traces > without_traces_data.blocks[0].total_tokens


class TestPageFoldingThreshold:
    @pytest.mark.asyncio
    async def test_fold_not_triggered_below_threshold(self):
        layer, store, _ = _make_layer(fold_token_threshold=999999)
        identity = _make_identity()

        topic_id = None
        settle_payload = None
        for i in range(5):
            target = topic_id or "NEW_TOPIC"
            topic_id, settle_payload = await layer.route_and_ingest(
                target,
                _make_payload(f"msg{i}", f"reply{i}", identity),
                identity_scope=_identity_scope(identity),
            )

        assert settle_payload is None
        topic_data = store.get(_identity_scope(identity), topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 5
        assert topic_data.state_summary == ""

    @pytest.mark.asyncio
    async def test_token_overflow_compacts_old_prefix_and_retains_recent_blocks(self):
        relay = Mock()
        relay.should_relay.return_value = None
        relay.generate_summary.return_value = "Test summary"
        layer, store, _ = _make_layer(
            fold_token_threshold=10,
            fold_retain_recent_blocks=2,
            relay=relay,
        )

        topic_id = await layer.create_new_topic(_identity_scope())
        settle_payload = None
        for i in range(3):
            _, settle_payload = await layer.route_and_ingest(
                topic_id,
                _make_payload(f"question-{i}-" * 80, f"answer-{i}"),
                identity_scope=_identity_scope(),
            )

        assert settle_payload is None
        relay.generate_summary.assert_called_once()
        folded_blocks = relay.generate_summary.call_args.kwargs["blocks_to_fold"]
        assert [block.user_query for block in folded_blocks] == [
            "question-0-" * 80
        ]
        topic_data = store.get(_identity_scope(), topic_id, touch=False)
        assert topic_data is not None
        assert topic_data.state_summary == "Test summary"
        assert [block.user_query for block in topic_data.blocks] == [
            "question-1-" * 80,
            "question-2-" * 80,
        ]
        assert topic_data.total_tokens == sum(
            block.total_tokens for block in topic_data.blocks
        )

    @pytest.mark.asyncio
    async def test_retain_count_larger_than_blocks_defers_folding(self):
        relay = Mock()
        relay.should_relay.return_value = None
        layer, store, _ = _make_layer(
            fold_token_threshold=10,
            fold_retain_recent_blocks=5,
            relay=relay,
        )

        topic_id = await layer.create_new_topic(_identity_scope())
        for i in range(3):
            await layer.route_and_ingest(
                topic_id,
                _make_payload(f"question-{i}-" * 80, f"answer-{i}"),
                identity_scope=_identity_scope(),
            )

        topic_data = store.get(_identity_scope(), topic_id, touch=False)
        assert topic_data is not None
        assert [block.user_query for block in topic_data.blocks] == [
            "question-0-" * 80,
            "question-1-" * 80,
            "question-2-" * 80,
        ]
        assert topic_data.state_summary == ""
        relay.generate_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_fold_failure_releases_processing_and_retry_resumes_without_duplicate(self):
        """摘要失败后释放单写者预约；等价 retry 重新预约并继续后置义务。"""
        relay = Mock()
        relay.should_relay.return_value = None
        relay.generate_summary.side_effect = [
            RuntimeError("fold failed"),
            "recovered summary",
        ]
        layer, store, _ = _make_layer(
            fold_token_threshold=10,
            fold_retain_recent_blocks=1,
            relay=relay,
        )
        identity_scope = _identity_scope()
        topic_id = await layer.create_new_topic(identity_scope)
        await layer.route_and_ingest(
            topic_id,
            _make_payload("first-" * 80, "answer-1"),
            identity_scope=identity_scope,
            interaction_id="interaction-1",
        )
        retry_payload = _make_payload("second-" * 80, "answer-2")

        with pytest.raises(RuntimeError, match="fold failed"):
            await layer.route_and_ingest(
                topic_id,
                retry_payload,
                identity_scope=identity_scope,
                interaction_id="interaction-2",
            )

        after_failure = store.get(identity_scope, topic_id, touch=False)
        assert after_failure is not None
        assert after_failure.state.value == "idle"
        assert [block.user_query for block in after_failure.blocks] == [
            "first-" * 80,
            "second-" * 80,
        ]

        retried_topic_id, settlement = await layer.route_and_ingest(
            topic_id,
            retry_payload,
            identity_scope=identity_scope,
            interaction_id="interaction-2",
        )

        assert retried_topic_id == topic_id
        assert settlement is None
        after_retry = store.get(identity_scope, topic_id, touch=False)
        assert after_retry is not None
        assert after_retry.state.value == "idle"
        assert after_retry.state_summary == "recovered summary"
        assert [block.user_query for block in after_retry.blocks] == ["second-" * 80]


class TestCompaction:
    @pytest.mark.asyncio
    async def test_compaction_writes_summary_and_retains_recent_blocks(self):
        relay = Mock()
        relay.generate_summary.return_value = "Test summary"
        service, store, scope, topic_id = _make_service_with_content(relay)

        execution = service.handle_trigger(
            scope, topic_id, TriggerReason.MANUAL_COMPACT,
            retain_recent_blocks=2,
        )

        assert execution.compacted is True
        topic_data = store.get(scope, topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 2
        assert topic_data.state_summary == "Test summary"
        assert topic_data.total_tokens == 40

    @pytest.mark.asyncio
    async def test_compaction_rejects_zero_retain_count(self):
        """compact 必须至少保留一个最新 block；0 在输入边界以具体异常拒绝。"""
        relay = Mock()
        service, store, scope, topic_id = _make_service_with_content(relay)

        with pytest.raises(ValueError, match="retain_recent_blocks must be >= 1"):
            service.handle_trigger(
                scope, topic_id, TriggerReason.MANUAL_COMPACT,
                retain_recent_blocks=0,
            )

    @pytest.mark.asyncio
    async def test_compaction_rejects_negative_retain_count(self):
        relay = Mock()
        service, store, scope, topic_id = _make_service_with_content(relay)

        with pytest.raises(ValueError, match="retain_recent_blocks must be >= 1"):
            service.handle_trigger(
                scope, topic_id, TriggerReason.MANUAL_COMPACT,
                retain_recent_blocks=-1,
            )

    @pytest.mark.asyncio
    async def test_manual_compact_failure_restores_idle_state(self):
        """manual compact 失败后必须恢复 IDLE，话题不能永久 busy。"""

        class FailingRelay:
            def generate_summary(self, blocks_to_fold, previous_summary=None):
                raise RuntimeError("summary failed")

        layer, store, service = build_perception_stack(relay=FailingRelay())
        scope = _identity_scope()
        topic = store.create(scope)
        block = LogicalBlock(
            turn=TurnRecord(identity=scope.actor_identity, user_query="q", assistant_final_text="a"),
            total_tokens=1,
        )
        store.put(topic.model_copy(update={"blocks": (block, block), "total_tokens": 2}))

        with pytest.raises(RuntimeError, match="summary failed"):
            await service.handle_trigger(
                scope, topic.topic_id, TriggerReason.MANUAL_COMPACT,
                retain_recent_blocks=1,
            )

        assert store.get(scope, topic.topic_id, touch=False).state is BufferState.IDLE


class TestPageFoldingConfig:
    @pytest.mark.parametrize("retain_count", [0, -1])
    def test_public_config_rejects_non_positive_retain_count(self, retain_count):
        with pytest.raises(ValidationError):
            SemanticFlowPerceptionConfig(
                fold_retain_recent_blocks=retain_count,
            )


class TestPageFoldingCumulative:
    @pytest.mark.asyncio
    async def test_fold_cumulative_summary(self):
        relay = Mock()
        relay.should_relay.return_value = None
        relay.generate_summary.side_effect = (
            lambda blocks_to_fold, previous_summary: previous_summary + "---folded"
        )
        layer, store, _ = _make_layer(fold_token_threshold=50, relay=relay)
        identity = _make_identity()

        topic_id = await layer.create_new_topic(_identity_scope(identity))
        for i in range(4):
            await layer.route_and_ingest(
                topic_id,
                _make_payload(f"wave1 q{i} " * 20, f"wave1 a{i} " * 20, identity),
                identity_scope=_identity_scope(identity),
            )

        topic_data = store.get(_identity_scope(identity), topic_id, touch=False)
        assert topic_data is not None
        first_summary = topic_data.state_summary
        assert first_summary != ""

        for i in range(4):
            await layer.route_and_ingest(
                topic_id,
                _make_payload(f"wave2 q{i} " * 20, f"wave2 a{i} " * 20, identity),
                identity_scope=_identity_scope(identity),
            )

        topic_data = store.get(_identity_scope(identity), topic_id, touch=False)
        assert topic_data is not None
        assert "---" in topic_data.state_summary
        assert first_summary in topic_data.state_summary
