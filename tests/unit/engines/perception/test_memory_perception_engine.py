"""MemoryPerceptionEngine 纯算法单元测试。

测试覆盖:
- build_block: turn_events 结构化路径（assistant_final_text 保留、事件归并为
  actions、mtp_traces 直传）、token 估算（含 traces 贡献）
- should_compact: 阈值边界（严格大于）
- select_blocks_to_fold: 最旧前缀选择、retain 边界、非法 retain 拒绝
- generate_fold_summary: 委托给持有的 RelayController

纯单元测试：Engine 为唯一真实对象，RelayController 使用 mock。
"""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from hivememory.core.models import ActorIdentity, LogicalBlock, TraceItem, TurnEvent, TurnRecord
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.system.config import SemanticFlowPerceptionConfig
from tests.helpers.workspace import make_identity_scope


def _identity():
    return ActorIdentity(user_id="u1", agent_id="a1")


def _scope():
    return make_identity_scope(actor_identity=_identity())


def _turn_event(kind="tool_call", tool_kind="READ", target="alias_x") -> TurnEvent:
    return TurnEvent(
        kind=kind,
        sequence=0,
        role="assistant",
        content="",
        tool_kind=tool_kind,
        tool_name=target if tool_kind == "RUN" else None,
        target=target,
    )


def _payload(user_msg="hello", assistant_msg="world", traces=None, turn_events=None):
    return InteractionPayload(
        user_message=user_msg,
        assistant_final_text=assistant_msg,
        turn_events=turn_events
        if turn_events is not None
        else [
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=assistant_msg,
            )
        ],
        mtp_traces=traces or [],
    )


def _block(total_tokens=10) -> LogicalBlock:
    return LogicalBlock(
        turn=TurnRecord(identity=_identity(), user_query="q", assistant_final_text="a"),
        total_tokens=total_tokens,
    )


def _mock_relay():
    """创建 mock RelayController，用于不涉及摘要生成的测试。"""
    mock = Mock()
    mock.generate_summary.return_value = "mocked summary"
    return mock


# ========== build_block：结构化摄入路径 ==========


class TestBuildBlock:
    def test_persists_assistant_final_text_and_turn_events(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )
        turn_event = _turn_event()

        block = engine.build_block(
            _payload("hello", "clean reply", turn_events=[turn_event]), _scope()
        )

        assert block.assistant_final_text == "clean reply"
        assert len(block.turn_events) == 1
        assert block.turn_events[0].kind == turn_event.kind
        assert block.turn_events[0].tool_kind == turn_event.tool_kind

    def test_reduces_turn_events_to_actions(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )
        payload = InteractionPayload(
            user_message="hello",
            assistant_final_text="clean reply",
            turn_events=[
                TurnEvent(
                    kind="tool_call",
                    sequence=0,
                    role="assistant",
                    content="<< READ | alias_x >>",
                    action_id="a1",
                    tool_kind="READ",
                    tool_name="alias_x",
                    target="alias_x",
                ),
                TurnEvent(
                    kind="tool_result",
                    sequence=1,
                    role="user",
                    content="result",
                    action_id="a1",
                    tool_kind="READ",
                    tool_name="alias_x",
                    status="success",
                    render_as="system_tool_result",
                ),
            ],
        )

        block = engine.build_block(payload, _scope())

        assert len(block.actions) == 1
        assert block.actions[0].action_id == "a1"
        assert block.actions[0].tool_kind == "READ"
        assert block.actions[0].tool_name == "alias_x"
        assert block.actions[0].status == "success"
        assert len(block.actions[0].results) == 1

    def test_persists_payload_mtp_traces(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )
        payload = _payload("hello", "clean", traces=[TraceItem(action="SEARCH", query="my query")])

        block = engine.build_block(payload, _scope())

        assert [trace.action for trace in block.semantic_traces] == ["SEARCH"]

    def test_keeps_semantic_traces_empty_when_payload_empty(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )

        block = engine.build_block(_payload("hello", "clean"), _scope())

        assert block.semantic_traces == ()

    def test_empty_final_text_stays_empty(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )

        block = engine.build_block(_payload("hello", ""), _scope())

        assert block.assistant_final_text == ""


# ========== token 估算 ==========


class TestTokenEstimation:
    def test_total_tokens_computed_from_query_and_answer(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )

        block = engine.build_block(_payload("What is Python?", "Python is a language"), _scope())

        assert block.total_tokens > 0

    def test_traces_increase_total_tokens(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )

        with_traces = engine.build_block(
            _payload(
                "q",
                "a",
                traces=[
                    TraceItem(action="SEARCH", query="how to sort a list"),
                    TraceItem(action="READ", target="my_notes_alias"),
                ],
            ),
            _scope(),
        )
        without_traces = engine.build_block(_payload("q", "a"), _scope())

        assert with_traces.total_tokens > without_traces.total_tokens


# ========== should_compact：阈值判断 ==========


class TestShouldCompact:
    def test_not_triggered_below_threshold(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=100),
            relay_controller=_mock_relay()
        )

        assert engine.should_compact(99) is False

    def test_not_triggered_at_exact_threshold(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=100),
            relay_controller=_mock_relay()
        )

        # 与旧 is_idle/折叠语义一致：达到阈值不触发，严格大于才触发
        assert engine.should_compact(100) is False

    def test_triggered_above_threshold(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=100),
            relay_controller=_mock_relay()
        )

        assert engine.should_compact(101) is True


# ========== select_blocks_to_fold：折叠选择 ==========


class TestSelectBlocksToFold:
    def test_selects_oldest_prefix_and_retains_recent(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )
        blocks = tuple(_block(total_tokens=i) for i in range(3))

        folded = engine.select_blocks_to_fold(blocks, retain_recent=2)

        assert folded == [blocks[0]]  # 只折叠最旧的 1 块

    def test_defers_folding_when_retain_covers_all_blocks(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )
        blocks = tuple(_block() for _ in range(3))

        assert engine.select_blocks_to_fold(blocks, retain_recent=5) == []

    def test_rejects_retain_below_one(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay()
        )

        with pytest.raises(ValueError, match="retain_recent must be >= 1"):
            engine.select_blocks_to_fold((_block(),), retain_recent=0)


# ========== generate_fold_summary：委托 RelayController ==========


class TestGenerateFoldSummary:
    def test_delegates_to_relay_controller(self):
        mock_relay = _mock_relay()
        mock_relay.generate_summary.return_value = "summarized content"
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=mock_relay
        )
        blocks = [_block(), _block()]

        result = engine.generate_fold_summary(blocks, "previous summary")

        assert result == "summarized content"
        mock_relay.generate_summary.assert_called_once_with(blocks, "previous summary")


# ========== 配置契约 ==========


class TestEngineConfig:
    @pytest.mark.parametrize("retain_count", [0, -1])
    def test_public_config_rejects_non_positive_retain_count(self, retain_count):
        with pytest.raises(ValidationError):
            SemanticFlowPerceptionConfig(fold_retain_recent_blocks=retain_count)
