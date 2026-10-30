"""MemoryPerceptionEngine 纯算法单元测试。

测试覆盖:
- build_block: 逻辑块只保存结局、来源身份与载荷中的事件集；token 统计等于
  整轮渲染材料的估算（工具输出按截断上限计入，能力调用的结果不计入）
- should_compact: 阈值边界（严格大于）
- select_blocks_to_fold: 最旧前缀选择、retain 边界、非法 retain 拒绝
- generate_fold_summary: 委托给持有的 RelayController

纯单元测试：Engine 为唯一真实对象，RelayController 使用 mock。
"""

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from hivememory.config.patchouli import GenerationTranscriptConfig, SemanticFlowPerceptionConfig
from hivememory.core.models import ActorIdentity, LogicalBlock, TurnEvent
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.material import render_turn_text
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.utils.token_estimator import estimate_tokens
from tests.helpers.sessions import make_interaction_payload, make_logical_block


def _identity():
    return ActorIdentity(user_id="u1", agent_id="a1")


def _payload(*events: TurnEvent, user: str = "q") -> InteractionPayload:
    """以用户消息开头，其后是给出的事件。"""
    return make_interaction_payload(
        turn_events=[TurnEvent(kind="user_message", sequence=0, content=user), *events]
    )


def _block(total_tokens=10) -> LogicalBlock:
    return make_logical_block("q", "a", total_tokens=total_tokens)


def _mock_relay():
    """创建 mock RelayController，用于不涉及摘要生成的测试。"""
    mock = Mock()
    mock.generate_summary = AsyncMock(return_value="mocked summary")
    return mock


def _engine(**transcript_limits) -> MemoryPerceptionEngine:
    return MemoryPerceptionEngine(
        config=SemanticFlowPerceptionConfig(fold_token_threshold=100),
        relay_controller=_mock_relay(),
        transcript_config=GenerationTranscriptConfig(**transcript_limits),
    )


def _bash(result: str, *, thought: str = "先跑测试") -> InteractionPayload:
    """一轮外部工具交互：一条思考与一次带结局的 Bash 调用。"""
    return _payload(
        TurnEvent(kind="thought", sequence=1, content=thought),
        TurnEvent(
            kind="tool_call",
            sequence=2,
            content="",
            action_id="toolu-1",
            tool_name="Bash",
            tool_args={"command": "pytest -q"},
            status="success",
            result=result,
        ),
    )


def _read(result: str) -> InteractionPayload:
    """一轮能力调用：读取一条记忆，结果是取回的正文。"""
    return _payload(
        TurnEvent(
            kind="tool_call",
            sequence=1,
            content="",
            action_id="a1",
            verb="READ",
            targets=("alias_x",),
            status="success",
            result=result,
        ),
    )


# ========== build_block：结构化摄入 ==========


class TestBuildBlock:
    def test_block_keeps_events_outcome_and_identity_only(self):
        """逻辑块只保存事件集与结局；用户消息与回复正文从事件读取。"""
        payload = make_interaction_payload("hello", "clean reply", outcome="cancelled")

        block = _engine().build_block(payload, _identity())

        assert block.turn_events == tuple(payload.turn_events)
        assert (block.turn.outcome, block.identity) == ("cancelled", _identity())
        assert (block.user_message, block.reply_text) == ("hello", "clean reply")

    def test_total_tokens_estimate_the_rendered_turn_material(self):
        """块的 token 统计就是整轮渲染材料的估算，与生成看到的材料一致。"""
        engine = _engine(max_result_chars=200)
        payload = _bash("x" * 5_000)

        block = engine.build_block(payload, _identity())

        assert block.total_tokens == estimate_tokens(
            render_turn_text(payload.turn_events, GenerationTranscriptConfig(max_result_chars=200))
        )


class TestRenderedMaterialTokens:
    def test_external_tool_output_counts_only_up_to_transcript_limits(self):
        """工具结果按材料的单条上限计入：超出上限的部分不再增加统计。"""
        engine = _engine(max_result_chars=200)

        long_output = engine.build_block(_bash("x" * 5_000), _identity())
        longer_output = engine.build_block(_bash("x" * 50_000), _identity())
        no_actions = engine.build_block(make_interaction_payload("q", "a"), _identity())

        assert long_output.total_tokens == longer_output.total_tokens
        assert long_output.total_tokens > no_actions.total_tokens

    def test_capability_results_do_not_add_tokens(self):
        """能力调用只有一行摘要：取回的记忆正文多长都不进入统计。"""
        engine = _engine()

        short = engine.build_block(_read("短"), _identity())
        long = engine.build_block(_read("已有记忆正文" * 200), _identity())

        assert short.total_tokens == long.total_tokens

    def test_fold_threshold_counts_rendered_thoughts_and_tool_results(self):
        """折叠阈值按材料触发：短问答加一段工具输出即可越过阈值。"""
        engine = _engine()

        with_output = engine.build_block(_bash("collected 120 items, 3 failed " * 20), _identity())
        messages_only = engine.build_block(make_interaction_payload("q", "a"), _identity())

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
