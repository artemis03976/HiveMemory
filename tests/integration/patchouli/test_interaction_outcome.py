"""中断结局贯穿感知记录与两种提示词转写，防止不完整输出被当成正常事实。"""

import pytest

from hivememory.config.patchouli import SemanticFlowPerceptionConfig
from hivememory.core.models import ActorIdentity
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.prompts.transcript.generation import GenerationTranscriptBuilder
from hivememory.prompts.transcript.history import HistoryTranscriptBuilder
from tests.helpers.sessions import make_conversation_turn


@pytest.mark.parametrize("outcome, note", [("cancelled", "已中断"), ("failed", "执行失败")])
def test_outcome_survives_perception_and_transcript_projection(outcome, note):
    """实际 block 与生成材料保留结局，下一轮历史也明确提示不完整。"""
    turn = make_conversation_turn(
        user_message="问题", assistant_final_text="半截回答", outcome=outcome
    )
    engine = MemoryPerceptionEngine(config=SemanticFlowPerceptionConfig(), relay_controller=None)
    block = engine.build_block(turn.payload, ActorIdentity(agent_id="default"))
    assert block.turn.outcome == outcome
    generation = GenerationTranscriptBuilder()
    transcript = generation.build_transcript(generation.build_context([block]))
    assert f"[Outcome: {outcome}]" in transcript
    assert "半截回答" in transcript
    history = HistoryTranscriptBuilder().build_messages([turn], language="zh")
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[-1]["content"].startswith("半截回答\n\n")
    assert note in history[-1]["content"]
