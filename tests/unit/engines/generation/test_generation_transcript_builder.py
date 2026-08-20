"""
GenerationTranscriptBuilder / GenerationContext 单测

覆盖 Phase 3 实施方案清单 §16.11 要求的测试场景:
1. GenerationTranscriptBuilder.build_context() 基础构建
2. trace_summaries 映射 (SEARCH / READ / RUN)
3. assistant_final_text 优先于 clean_response
4. state_summary 进入 GenerationContext
5. build_transcript() 格式渲染
6. MemoryGenerationEngine.process() 识别 request.context
7. Mode B / Mode C 与 GenerationContext 兼容
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import Identity, LogicalBlock, TraceItem, TurnRecord, WriteFocus
from hivememory.engines.generation.models import (
    GenerationContext,
    GenerationRequest,
    GenerationTurn,
)
from hivememory.prompts.transcript import GenerationTranscriptBuilder
from tests.helpers.memory import make_memory_creation_context

# ============ 辅助工厂 ============

def _identity(agent_id: str = "a1") -> Identity:
    return Identity(user_id="u1", agent_id=agent_id)


def _request(**values) -> GenerationRequest:
    return GenerationRequest(**values)


def _block(
    user_query: str = "问题",
    assistant_final_text: str = "",
    traces: list = None,
) -> LogicalBlock:
    return LogicalBlock(
        turn=TurnRecord(
            identity=_identity(),
            user_query=user_query,
            assistant_final_text=assistant_final_text,
            semantic_traces=traces or [],
        )
    )


def _trace(action: str, **kwargs) -> TraceItem:
    return TraceItem(action=action, **kwargs)


builder = GenerationTranscriptBuilder()


# ============ 1. build_context 基础行为 ============

class TestBuildContextBasic:
    def test_empty_blocks_produces_empty_turns(self):
        ctx = builder.build_context([], state_summary="")
        assert ctx.turns == []

    def test_state_summary_propagated(self):
        ctx = builder.build_context([], state_summary="话题摘要")
        assert ctx.state_summary == "话题摘要"

    def test_single_block_produces_one_turn(self):
        blocks = [_block("你好", assistant_final_text="你好呀")]
        ctx = builder.build_context(blocks)
        assert len(ctx.turns) == 1
        assert ctx.turns[0].user_query == "你好"

    def test_multiple_blocks_produce_multiple_turns(self):
        blocks = [_block(f"问题{i}", assistant_final_text=f"答案{i}") for i in range(3)]
        ctx = builder.build_context(blocks)
        assert len(ctx.turns) == 3

    def test_block_with_only_user_query_is_kept(self):
        """user_query 非空即保留（即使无 final_text）"""
        blocks = [_block("只有问题", assistant_final_text="")]
        ctx = builder.build_context(blocks)
        assert len(ctx.turns) == 1
        assert ctx.turns[0].user_query == "只有问题"

    def test_block_with_no_query_and_no_text_is_filtered(self):
        """user_query 和 assistant_final_text 都为空的 block 被过滤"""
        blocks = [LogicalBlock(turn=TurnRecord(identity=_identity()))]  # 全空
        ctx = builder.build_context(blocks)
        assert len(ctx.turns) == 0


# ============ 2. assistant_final_text 优先级 ============

class TestFinalTextPriority:
    def test_both_empty_produces_empty_final_text(self):
        block = _block(assistant_final_text="")
        ctx = builder.build_context([block])
        assert ctx.turns[0].assistant_final_text == ""


# ============ 3. trace_summaries 映射 ============

class TestTraceSummaries:
    def test_search_trace(self):
        block = _block(traces=[_trace("SEARCH", query="docker deploy")])
        ctx = builder.build_context([block])
        assert ctx.turns[0].trace_summaries == ['SEARCH: "docker deploy"']

    def test_read_trace(self):
        block = _block(traces=[_trace("READ", target="fact_auth_flow")])
        ctx = builder.build_context([block])
        assert ctx.turns[0].trace_summaries == ["READ: fact_auth_flow"]

    def test_run_trace(self):
        block = _block(traces=[_trace("RUN", tool="git_log", status="success")])
        ctx = builder.build_context([block])
        assert ctx.turns[0].trace_summaries == ["RUN: git_log (success)"]

    def test_run_trace_unknown_status(self):
        block = _block(traces=[_trace("RUN", tool="web_search")])
        ctx = builder.build_context([block])
        assert ctx.turns[0].trace_summaries == ["RUN: web_search (unknown)"]

    def test_multiple_traces_in_order(self):
        block = _block(traces=[
            _trace("SEARCH", query="query1"),
            _trace("READ", target="alias_x"),
            _trace("RUN", tool="tool_y", status="error"),
        ])
        ctx = builder.build_context([block])
        summaries = ctx.turns[0].trace_summaries
        assert len(summaries) == 3
        assert summaries[0].startswith("SEARCH")
        assert summaries[1].startswith("READ")
        assert summaries[2].startswith("RUN")

    def test_no_traces_produces_empty_summaries(self):
        block = _block(traces=[])
        ctx = builder.build_context([block])
        assert ctx.turns[0].trace_summaries == []

    def test_call_trace_is_skipped(self):
        """CALL 类型 trace 不生成摘要"""
        block = _block(traces=[_trace("CALL", target="sub_agent")])
        ctx = builder.build_context([block])
        assert ctx.turns[0].trace_summaries == []


# ============ 4. build_transcript 格式渲染 ============

class TestBuildTranscript:
    def test_empty_context_produces_empty_string(self):
        ctx = GenerationContext()
        assert builder.build_transcript(ctx) == ""

    def test_state_summary_in_header(self):
        ctx = GenerationContext(state_summary="当前话题是 X", turns=[])
        result = builder.build_transcript(ctx)
        assert "[Topic State]" in result
        assert "当前话题是 X" in result

    def test_no_state_summary_no_header(self):
        ctx = GenerationContext(state_summary="", turns=[
            GenerationTurn(user_query="hi", assistant_final_text="hello", identity=_identity())
        ])
        result = builder.build_transcript(ctx)
        assert "[Topic State]" not in result

    def test_turn_user_query_rendered(self):
        ctx = GenerationContext(turns=[
            GenerationTurn(user_query="我的问题", assistant_final_text="", identity=_identity())
        ])
        result = builder.build_transcript(ctx)
        assert "[User]: 我的问题" in result

    def test_turn_assistant_rendered(self):
        ctx = GenerationContext(turns=[
            GenerationTurn(user_query="q", assistant_final_text="我的回答", identity=_identity())
        ])
        result = builder.build_transcript(ctx)
        assert "[Assistant]: 我的回答" in result

    def test_actions_section_rendered(self):
        ctx = GenerationContext(turns=[
            GenerationTurn(
                user_query="q",
                assistant_final_text="a",
                trace_summaries=['SEARCH: "auth flow"', "READ: fact_x"],
                identity=_identity(),
            )
        ])
        result = builder.build_transcript(ctx)
        assert "[Actions]:" in result
        assert '- SEARCH: "auth flow"' in result
        assert "- READ: fact_x" in result

    def test_full_format_with_state_and_two_turns(self):
        ctx = GenerationContext(
            state_summary="重构进行中",
            turns=[
                GenerationTurn(
                    user_query="问题1",
                    trace_summaries=['SEARCH: "x"'],
                    assistant_final_text="答案1",
                    identity=_identity(),
                ),
                GenerationTurn(
                    user_query="问题2",
                    assistant_final_text="答案2",
                    identity=_identity(),
                ),
            ]
        )
        result = builder.build_transcript(ctx)
        assert "[Topic State]" in result
        assert "[Turn 1]" in result
        assert "[Turn 2]" in result
        assert "重构进行中" in result
        assert "问题1" in result
        assert "问题2" in result


# ============ 5. GenerationRequest.has_context ============

class TestGenerationRequestHasContext:
    def test_has_context_false_when_no_context(self):
        req = _request()
        assert not req.has_context

    def test_has_context_false_when_context_empty_turns(self):
        req = _request(context=GenerationContext())
        assert not req.has_context

    def test_has_context_true_when_context_has_turns(self):
        ctx = GenerationContext(turns=[
            GenerationTurn(user_query="q", identity=_identity())
        ])
        req = _request(context=ctx)
        assert req.has_context


class TestGenerationRequestIdentity:
    def test_request_is_identity_agnostic(self):
        """捕获 GenerationRequest 重新持有权限/ownership 字段的缺陷。"""
        request = GenerationRequest(context=GenerationContext())
        assert not hasattr(request, "creation_context")
        assert not hasattr(request, "identity")


# ============ 6. MemoryGenerationEngine 新路径集成测试 ============

class TestEngineWithGenerationContext:
    def _make_engine(self, extractor_returns=None, deduplicator_returns=None):
        from hivememory.engines.generation.engine import MemoryGenerationEngine
        storage = MagicMock()
        storage.search = AsyncMock(return_value=[])
        storage.upsert = AsyncMock()
        extractor = MagicMock()
        deduplicator = MagicMock()

        if extractor_returns is not None:
            extractor.extract.return_value = extractor_returns
        else:
            draft = MagicMock()
            draft.has_value = False
            extractor.extract.return_value = draft

        if deduplicator_returns is not None:
            deduplicator.check_duplicate.return_value = deduplicator_returns
        else:
            deduplicator.check_duplicate.return_value = (MagicMock(), MagicMock())

        engine = MemoryGenerationEngine(
            mid_term=storage,
            extractor=extractor,
            deduplicator=deduplicator,
        )
        return engine, extractor, deduplicator

    @pytest.mark.asyncio
    async def test_process_with_context_calls_extractor(self):
        """context 存在时调用 extractor.extract"""
        engine, extractor, _ = self._make_engine()
        ctx = GenerationContext(
            state_summary="摘要",
            turns=[GenerationTurn(user_query="q", assistant_final_text="a", identity=_identity())]
        )
        req = _request(context=ctx)
        await engine.process(req, identity_scope=make_memory_creation_context())
        extractor.extract.assert_called_once()
        transcript = extractor.extract.call_args[1]["transcript"]
        assert "[Turn 1]" in transcript
        assert "摘要" in transcript
        assert "[Topic State]" in transcript

    @pytest.mark.asyncio
    async def test_process_empty_context_skipped(self):
        """context 为空时跳过 extractor"""
        engine, extractor, _ = self._make_engine()
        req = _request(context=GenerationContext())  # no turns
        result = await engine.process(req, identity_scope=make_memory_creation_context())
        assert result == []
        extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_without_context_and_focus_skipped(self):
        """无上下文且无 focus 时直接跳过"""
        engine, extractor, _ = self._make_engine()
        req = _request()
        result = await engine.process(req, identity_scope=make_memory_creation_context())
        assert result == []
        extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_mode_b_with_context(self):
        """Mode B (write_focus) + context 兼容"""
        from hivememory.engines.generation.models import DuplicateDecision
        engine, extractor, deduplicator = self._make_engine()
        from hivememory.engines.generation.models import ExtractedMemoryDraft
        draft = ExtractedMemoryDraft(
            title="test_title", summary="a summary longer than ten chars", tags=["a"], memory_type="FACT",
            content="test content here", confidence_score=0.9, has_value=True,
        )
        extractor.extract.return_value = draft
        deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)

        ctx = GenerationContext(turns=[
            GenerationTurn(user_query="q", assistant_final_text="a", identity=_identity())
        ])
        focus = WriteFocus(content="content to write")
        req = _request(context=ctx, write_focus=focus)
        await engine.process(req, identity_scope=make_memory_creation_context())

        extractor.extract.assert_called_once()
        call_kwargs = extractor.extract.call_args[1]
        assert call_kwargs["metadata"]["mode"] == "write"
        assert "[Turn 1]" in call_kwargs["transcript"]
