"""
GenerationTranscriptBuilder / GenerationContext 单测

覆盖以下场景:
1. build_context() 从逻辑块构建轮次：轮次标识、用户消息、结局、来源身份、按顺序的材料
   条目与各段；各类步骤的渲染规则由感知层材料模块的测试覆盖，切段规则由切段模块的测试覆盖
2. build_transcript() 格式：话题状态、轮首结局标注、每段之前的段标签、用户消息之后按顺序
   的材料；没有段的轮次（升级前保存的生成任务）不印段标签
3. GenerationRequest.has_context
4. MemoryGenerationEngine.process() 识别 request.context；Mode B 与上下文兼容
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import ActorIdentity, TurnEvent, WriteFocus
from hivememory.engines.generation.models import (
    GenerationContext,
    GenerationRequest,
    GenerationSegment,
    GenerationTurn,
)
from hivememory.prompts.transcript import GenerationTranscriptBuilder
from tests.helpers.memory import make_drift_guard, make_memory_identity_scope
from tests.helpers.sessions import make_logical_block

# ============ 辅助工厂 ============


def _identity(agent_id: str = "a1") -> ActorIdentity:
    return ActorIdentity(user_id="u1", agent_id=agent_id)


def _request(**values) -> GenerationRequest:
    return GenerationRequest(**values)


def _turn(
    user_message: str = "q",
    *entries: str,
    outcome: str = "completed",
    segments: list[GenerationSegment] | None = None,
) -> GenerationTurn:
    return GenerationTurn(
        turn_id="turn-1",
        user_message=user_message,
        material_entries=list(entries),
        outcome=outcome,
        segments=segments or [],
        identity=_identity(),
    )


def _segment(first: int, last: int, entry_count: int) -> GenerationSegment:
    return GenerationSegment(first_sequence=first, last_sequence=last, entry_count=entry_count)


builder = GenerationTranscriptBuilder()


# ============ 1. build_context ============


class TestBuildContext:
    def test_empty_blocks_produces_empty_turns(self):
        ctx = builder.build_context([], state_summary="")
        assert ctx.turns == []

    def test_state_summary_propagated(self):
        ctx = builder.build_context([], state_summary="话题摘要")
        assert ctx.state_summary == "话题摘要"

    def test_block_becomes_turn_with_ordered_material_entries(self):
        """用户消息单独成字段，其后的材料按出现顺序排列，助手正文穿插其中；
        没有响应序号的轮次整轮一段。"""
        block = make_logical_block(
            identity=_identity("claude_code"),
            outcome="cancelled",
            turn_events=[
                TurnEvent(kind="user_message", sequence=0, content="之前怎么部署的"),
                TurnEvent(kind="assistant_message", sequence=1, content="先查记忆。"),
                TurnEvent(
                    kind="tool_call",
                    sequence=2,
                    content="",
                    action_id="toolu-1",
                    verb="READ",
                    targets=("fact_deploy",),
                    status="success",
                    result="记忆正文",
                ),
                TurnEvent(kind="assistant_message", sequence=4, content="按记忆中的步骤部署。"),
            ],
        )

        [turn] = builder.build_context([block]).turns

        assert turn == GenerationTurn(
            turn_id=block.turn.turn_id,
            outcome="cancelled",
            user_message="之前怎么部署的",
            material_entries=[
                "[Assistant]: 先查记忆。",
                "- READ: fact_deploy",
                "[Assistant]: 按记忆中的步骤部署。",
            ],
            segments=[_segment(0, 4, 3)],
            identity=_identity("claude_code"),
        )

    def test_segments_count_the_entries_each_one_renders(self):
        """各段记下事件区间与渲染出的条目数，条目拼起来就是整轮材料；不渲染的写入不计条目。"""
        block = make_logical_block(
            turn_events=[
                TurnEvent(kind="user_message", sequence=0, content="记下部署方式"),
                TurnEvent(
                    kind="tool_call",
                    sequence=1,
                    content="",
                    response_index=1,
                    action_id="toolu-1",
                    verb="WRITE",
                    tool_args={"content": "手动部署"},
                ),
                TurnEvent(
                    kind="assistant_message", sequence=2, content="已记下。", response_index=2
                ),
                TurnEvent(
                    kind="tool_call",
                    sequence=3,
                    content="",
                    response_index=3,
                    action_id="toolu-2",
                    tool_name="Bash",
                    tool_args={"command": "ls"},
                ),
            ]
        )

        [turn] = builder.build_context([block]).turns

        assert turn.segments == [_segment(0, 1, 0), _segment(2, 3, 2)]
        assert turn.material_entries == ["[Assistant]: 已记下。", '- TOOL: Bash {"command": "ls"}']

    def test_multiple_blocks_produce_multiple_turns(self):
        blocks = [make_logical_block(f"问题{i}", f"答案{i}") for i in range(3)]
        ctx = builder.build_context(blocks)
        assert [turn.user_message for turn in ctx.turns] == ["问题0", "问题1", "问题2"]


# ============ 2. build_transcript 格式渲染 ============


class TestBuildTranscript:
    def test_empty_context_produces_empty_string(self):
        assert builder.build_transcript(GenerationContext()) == ""

    def test_turn_renders_segment_labels_before_each_segment(self):
        """每段之前印出“轮次位置.段号”的标签，第一段从用户消息开始，材料按顺序排列；
        没有单独的动作段，也没有末尾的回复行。"""
        ctx = GenerationContext(
            turns=[
                _turn("你好", "[Assistant]: 你好。", segments=[_segment(0, 1, 1)]),
                _turn(
                    "修一下测试",
                    '- SEARCH: "测试约定"',
                    "[Assistant]: 先跑一遍。",
                    '- TOOL: Bash {"command": "pytest"}\n  RESULT (error): 1 failed',
                    "[Assistant]: 已修复。",
                    segments=[_segment(0, 1, 1), _segment(2, 4, 2), _segment(5, 5, 1)],
                ),
            ]
        )

        assert builder.build_transcript(ctx) == "\n".join(
            [
                "[Turn 1]",
                "[Segment 1.1]",
                "[User]: 你好",
                "[Assistant]: 你好。",
                "",
                "[Turn 2]",
                "[Segment 2.1]",
                "[User]: 修一下测试",
                '- SEARCH: "测试约定"',
                "[Segment 2.2]",
                "[Assistant]: 先跑一遍。",
                '- TOOL: Bash {"command": "pytest"}',
                "  RESULT (error): 1 failed",
                "[Segment 2.3]",
                "[Assistant]: 已修复。",
            ]
        )

    def test_turns_without_segments_render_without_labels(self):
        """话题状态在最前；结局不是 completed 的轮次在轮首标注；升级前保存的生成任务
        没有段，材料不印段标签。"""
        ctx = GenerationContext(
            state_summary="重构进行中",
            turns=[
                _turn("问题1", "[Assistant]: 答案1"),
                _turn("问题2", "[Assistant]: 半截", outcome="cancelled"),
            ],
        )

        assert builder.build_transcript(ctx) == (
            "[Topic State]\n重构进行中\n\n"
            "[Turn 1]\n[User]: 问题1\n[Assistant]: 答案1\n\n"
            "[Turn 2]\n[Outcome: cancelled] This turn did not complete normally.\n"
            "[User]: 问题2\n[Assistant]: 半截"
        )


# ============ 5. GenerationRequest.has_context ============


class TestGenerationRequestHasContext:
    def test_has_context_false_when_no_context(self):
        req = _request()
        assert not req.has_context

    def test_has_context_false_when_context_empty_turns(self):
        req = _request(context=GenerationContext())
        assert not req.has_context

    def test_has_context_true_when_context_has_turns(self):
        ctx = GenerationContext(turns=[_turn("q")])
        req = _request(context=ctx)
        assert req.has_context


class TestGenerationRequestIdentity:
    def test_request_is_identity_agnostic(self):
        """捕获 GenerationRequest 重新持有权限/ownership 字段的缺陷。"""
        request = GenerationRequest(context=GenerationContext())
        assert not hasattr(request, "identity_scope")
        assert not hasattr(request, "identity")


# ============ 6. MemoryGenerationEngine 新路径集成测试 ============


class TestEngineWithGenerationContext:
    def _make_engine(self, extractor_returns=None, deduplicator_returns=None):
        from hivememory.engines.generation.engine import MemoryGenerationEngine

        storage = MagicMock()
        storage.search = AsyncMock(return_value=[])
        storage.upsert = AsyncMock()
        storage.list_alias_holders = AsyncMock(return_value=[])
        extractor = AsyncMock()
        deduplicator = MagicMock()

        extractor.extract.return_value = [] if extractor_returns is None else extractor_returns

        if deduplicator_returns is not None:
            deduplicator.check_duplicate.return_value = deduplicator_returns
        else:
            deduplicator.check_duplicate.return_value = (MagicMock(), MagicMock())

        engine = MemoryGenerationEngine(
            mid_term=storage,
            extractor=extractor,
            deduplicator=deduplicator,
            drift_guard=make_drift_guard(),
        )
        return engine, extractor, deduplicator

    @pytest.mark.asyncio
    async def test_process_with_context_calls_extractor(self):
        """context 存在时调用 extractor.extract"""
        engine, extractor, _ = self._make_engine()
        ctx = GenerationContext(
            state_summary="摘要",
            turns=[_turn("q", "[Assistant]: a")],
        )
        req = _request(context=ctx)
        await engine.process(
            req,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )
        extractor.extract.assert_called_once()
        transcript = extractor.extract.call_args.args[0]
        assert "[Turn 1]" in transcript
        assert "摘要" in transcript
        assert "[Topic State]" in transcript

    @pytest.mark.asyncio
    async def test_process_empty_context_skipped(self):
        """context 为空时跳过 extractor"""
        engine, extractor, _ = self._make_engine()
        req = _request(context=GenerationContext())  # 无 turns
        result = await engine.process(
            req,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )
        assert result == []
        extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_without_context_and_focus_skipped(self):
        """无上下文且无 focus 时直接跳过"""
        engine, extractor, _ = self._make_engine()
        req = _request()
        result = await engine.process(
            req,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )
        assert result == []
        extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_mode_b_with_context(self):
        """Mode B (write_focus) + context 兼容"""
        from hivememory.engines.generation.models import DuplicateDecision

        engine, extractor, deduplicator = self._make_engine()
        from hivememory.engines.generation.models import ExtractedMemoryDraft

        draft = ExtractedMemoryDraft(
            title="test_title",
            summary="a summary longer than ten chars",
            tags=["a"],
            memory_type="FACT",
            content="test content here",
            confidence_score=0.9,
            has_value=True,
        )
        extractor.extract_write.return_value = draft
        deduplicator.check_duplicate.return_value = (DuplicateDecision.DISTINCT, None)

        ctx = GenerationContext(turns=[_turn("q", "[Assistant]: a")])
        focus = WriteFocus(content="content to write")
        req = _request(context=ctx, write_focus=focus)
        await engine.process(
            req,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        extractor.extract_write.assert_called_once()
        assert "[Turn 1]" in extractor.extract_write.call_args.args[0]
