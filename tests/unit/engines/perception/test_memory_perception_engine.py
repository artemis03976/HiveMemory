"""MemoryPerceptionEngine 纯算法单元测试。

测试覆盖:
- build_block: turn_events 结构化路径（assistant_final_text 保留、事件归并为
  actions、mtp_traces 直传）、token 估算（含 traces 贡献；思考与外部工具动作
  按生成转写的截断规则计入，MTP 动作不增加渲染统计）
- should_compact: 阈值边界（严格大于）
- select_blocks_to_fold: 最旧前缀选择、retain 边界、非法 retain 拒绝
- generate_fold_summary: 委托给持有的 RelayController

纯单元测试：Engine 为唯一真实对象，RelayController 使用 mock。
"""

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from hivememory.config.patchouli import GenerationTranscriptConfig, SemanticFlowPerceptionConfig
from hivememory.core.models import ActorIdentity, LogicalBlock, TraceItem, TurnEvent, TurnRecord
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from tests.helpers.workspace import make_identity_scope


def _identity():
    return ActorIdentity(user_id="u1", agent_id="a1")


def _scope():
    return make_identity_scope(actor_identity=_identity())


def _read_call(sequence: int = 1) -> TurnEvent:
    return TurnEvent(
        kind="tool_call",
        sequence=sequence,
        content="",
        action_id="a1",
        verb="READ",
        targets=("alias_x",),
    )


def _payload(user_msg="hello", assistant_msg="world", traces=None, turn_events=None):
    """``turn_events`` 是用户消息之后的事件，缺省为一条回复正文。"""
    if turn_events is None:
        turn_events = (
            [TurnEvent(kind="assistant_message", sequence=1, content=assistant_msg)]
            if assistant_msg
            else []
        )
    return InteractionPayload(
        user_message=user_msg,
        assistant_final_text=assistant_msg,
        turn_events=[TurnEvent(kind="user_message", sequence=0, content=user_msg), *turn_events],
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
    mock.generate_summary = AsyncMock(return_value="mocked summary")
    return mock


# ========== build_block：结构化摄入路径 ==========


class TestBuildBlock:
    def test_persists_assistant_final_text_and_turn_events(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )
        payload = _payload("hello", "clean reply", turn_events=[_read_call()])

        block = engine.build_block(payload, _scope().actor_identity)

        assert block.assistant_final_text == "clean reply"
        assert block.turn_events == tuple(payload.turn_events)

    def test_reduces_turn_events_to_actions(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )
        payload = _payload(
            "hello",
            "clean reply",
            turn_events=[
                _read_call(),
                TurnEvent(
                    kind="tool_result",
                    sequence=2,
                    content="result",
                    action_id="a1",
                    status="success",
                ),
            ],
        )

        block = engine.build_block(payload, _scope().actor_identity)

        assert len(block.actions) == 1
        assert block.actions[0].action_id == "a1"
        assert block.actions[0].tool_kind == "READ"
        assert block.actions[0].tool_name == "alias_x"
        assert block.actions[0].status == "success"
        assert len(block.actions[0].results) == 1

    def test_persists_payload_mtp_traces(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )
        payload = _payload("hello", "clean", traces=[TraceItem(action="SEARCH", query="my query")])

        block = engine.build_block(payload, _scope().actor_identity)

        assert [trace.action for trace in block.semantic_traces] == ["SEARCH"]

    def test_keeps_semantic_traces_empty_when_payload_empty(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )

        block = engine.build_block(_payload("hello", "clean"), _scope().actor_identity)

        assert block.semantic_traces == ()

    def test_empty_final_text_stays_empty(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )

        block = engine.build_block(_payload("hello", ""), _scope().actor_identity)

        assert block.assistant_final_text == ""


# ========== token 估算 ==========


class TestTokenEstimation:
    def test_total_tokens_computed_from_query_and_answer(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )

        block = engine.build_block(
            _payload("What is Python?", "Python is a language"), _scope().actor_identity
        )

        assert block.total_tokens > 0

    def test_traces_increase_total_tokens(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
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
            _scope().actor_identity,
        )
        without_traces = engine.build_block(_payload("q", "a"), _scope().actor_identity)

        assert with_traces.total_tokens > without_traces.total_tokens


def _external_tool_payload(result: str, *, thought: str = "先跑测试") -> InteractionPayload:
    """一轮导入的外部工具交互：一条思考、一次 Bash 调用与它的结果。"""
    return _payload(
        "q",
        "a",
        turn_events=[
            TurnEvent(kind="thought", sequence=1, content=thought),
            TurnEvent(
                kind="tool_call",
                sequence=2,
                content="",
                action_id="toolu-1",
                tool_name="Bash",
                tool_args={"command": "pytest -q"},
            ),
            TurnEvent(
                kind="tool_result",
                sequence=3,
                content=result,
                action_id="toolu-1",
                status="success",
            ),
        ],
    )


class TestRenderedActionTokens:
    def test_external_tool_output_counts_only_up_to_transcript_limits(self):
        """工具结果按生成转写的单条上限计入：超出上限的部分不再增加统计。"""
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(),
            relay_controller=_mock_relay(),
            transcript_config=GenerationTranscriptConfig(max_result_chars=200),
        )

        long_output = engine.build_block(
            _external_tool_payload("x" * 5_000), _scope().actor_identity
        )
        longer_output = engine.build_block(
            _external_tool_payload("x" * 50_000), _scope().actor_identity
        )
        no_actions = engine.build_block(_payload("q", "a"), _scope().actor_identity)

        assert long_output.total_tokens == longer_output.total_tokens
        assert long_output.total_tokens > no_actions.total_tokens

    def test_mtp_actions_do_not_add_rendered_tokens(self):
        """MTP 动作只按轨迹的查询与目标计入，结果正文不进入统计。"""
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )
        trace = TraceItem(action="READ", action_id="a1", target="alias_x")
        with_events = _payload(
            "q",
            "a",
            traces=[trace],
            turn_events=[
                _read_call(),
                TurnEvent(
                    kind="tool_result",
                    sequence=2,
                    content="已有记忆正文" * 200,
                    action_id="a1",
                    status="success",
                ),
            ],
        )

        assert (
            engine.build_block(with_events, _scope().actor_identity).total_tokens
            == engine.build_block(
                _payload("q", "a", traces=[trace]), _scope().actor_identity
            ).total_tokens
        )

    def test_fold_threshold_counts_rendered_thoughts_and_tool_results(self):
        """折叠阈值按新统计触发：短问答加一段工具输出即可越过阈值。"""
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=100),
            relay_controller=_mock_relay(),
        )

        with_output = engine.build_block(
            _external_tool_payload("collected 120 items, 3 failed " * 20),
            _scope().actor_identity,
        )
        messages_only = engine.build_block(_payload("q", "a"), _scope().actor_identity)

        assert engine.should_compact(with_output.total_tokens) is True
        assert engine.should_compact(messages_only.total_tokens) is False


# ========== should_compact：阈值判断 ==========


class TestShouldCompact:
    def test_not_triggered_below_threshold(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=100),
            relay_controller=_mock_relay(),
        )

        assert engine.should_compact(99) is False

    def test_not_triggered_at_exact_threshold(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=100),
            relay_controller=_mock_relay(),
        )

        # 与旧 is_idle/折叠语义一致：达到阈值不触发，严格大于才触发
        assert engine.should_compact(100) is False

    def test_triggered_above_threshold(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=100),
            relay_controller=_mock_relay(),
        )

        assert engine.should_compact(101) is True


# ========== select_blocks_to_fold：折叠选择 ==========


class TestSelectBlocksToFold:
    def test_selects_oldest_prefix_and_retains_recent(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )
        blocks = tuple(_block(total_tokens=i) for i in range(3))

        folded = engine.select_blocks_to_fold(blocks, retain_recent=2)

        assert folded == [blocks[0]]  # 只折叠最旧的 1 块

    def test_defers_folding_when_retain_covers_all_blocks(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )
        blocks = tuple(_block() for _ in range(3))

        assert engine.select_blocks_to_fold(blocks, retain_recent=5) == []

    def test_rejects_retain_below_one(self):
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=_mock_relay()
        )

        with pytest.raises(ValueError, match="retain_recent must be >= 1"):
            engine.select_blocks_to_fold((_block(),), retain_recent=0)


# ========== generate_fold_summary：委托 RelayController ==========


class TestGenerateFoldSummary:
    @pytest.mark.asyncio
    async def test_delegates_to_relay_controller(self):
        mock_relay = _mock_relay()
        mock_relay.generate_summary.return_value = "summarized content"
        engine = MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(), relay_controller=mock_relay
        )
        blocks = [_block(), _block()]

        result = await engine.generate_fold_summary(blocks, "previous summary")

        assert result == "summarized content"
        mock_relay.generate_summary.assert_awaited_once_with(blocks, "previous summary")


# ========== 配置契约 ==========


class TestEngineConfig:
    @pytest.mark.parametrize("retain_count", [0, -1])
    def test_public_config_rejects_non_positive_retain_count(self, retain_count):
        with pytest.raises(ValidationError):
            SemanticFlowPerceptionConfig(fold_retain_recent_blocks=retain_count)
