"""会话视图的触发、稳定边界与近期原文预算；不接触模型或会话存储。"""

import pytest

from hivememory.alice.orchestration.context_view import SessionContextView
from hivememory.config.alice import CompactionConfig
from hivememory.core.models import OMNI_DOLL_PROFILE, TurnEvent
from hivememory.core.protocol.models import AgentRunContext
from tests.helpers.sessions import make_compaction_checkpoint, make_conversation_turn
from tests.helpers.workspace import make_execution_labels


def _read_turn(name, size):
    return make_conversation_turn(
        interaction_id=name,
        user_message="read",
        turn_events=[
            TurnEvent(
                kind="tool_result",
                sequence=0,
                role="user",
                content="x" * size,
                tool_kind="READ",
                target=name,
                status="success",
            )
        ],
    )


def _view(turns, *, checkpoint=None, config=None, limit=1000):
    return SessionContextView(
        context=AgentRunContext(
            labels=make_execution_labels(),
            agent_profile=OMNI_DOLL_PROFILE,
            permitted_mtp_verbs=frozenset({"RUN"}),
            session_history=turns,
        ),
        fixed_messages=[{"role": "user", "content": "next"}],
        checkpoint=checkpoint,
        input_token_limit=limit,
        config=config or CompactionConfig(),
    )


def test_threshold_is_strict_and_internal_force_skips_it():
    """恰好等于触发阈值时无需压缩，内部强制参数可复用相同减重入口。"""
    turn = make_conversation_turn(user_message="x" * 2964)
    view = _view((turn,))  # 历史 745 token，加本轮 5 token 恰为 750。
    assert view.estimated_tokens == 750
    assert view.should_compact() is False
    assert view.should_compact(force=True) is True
    assert _view(()).should_compact(force=True) is False


def test_reduction_keeps_a_whole_recent_suffix_and_reuses_stable_boundary():
    """只在压缩时推进；后续新增短轮次不会让已保存的边界滑动。"""
    turns = (_read_turn("old", 6000), _read_turn("recent", 600))
    view = _view(turns)
    assert view.should_compact() is True
    result = view.compact()
    assert result.checkpoint.reduced_through == "old"
    assert result.checkpoint.summarized_through is None
    assert result.covered_turns == 1
    assert result.tokens_after < 500
    assert result.messages[-3]["content"] == "read"
    assert result.messages[-2]["content"] == "x" * 600
    reused = _view(turns, checkpoint=result.checkpoint)
    assert reused.messages == result.messages
    assert reused.should_compact() is False
    appended = _view(
        (*turns, make_conversation_turn(user_message="tiny")), checkpoint=result.checkpoint
    )
    assert appended.messages[:-2] == result.messages[:-1]
    assert appended.should_compact() is False


def test_large_last_turn_can_leave_no_raw_suffix_and_boundary_never_recedes():
    """单个近期轮次放不下时整轮减重；预算增大后强制压缩也不回退已有游标。"""
    turns = (_read_turn("old", 6000), _read_turn("recent", 2000))
    compacted = _view(turns).compact()
    assert compacted.checkpoint.reduced_through == "recent"
    assert compacted.covered_turns == 2
    assert all("x" * 2000 not in message["content"] for message in compacted.messages)
    repeated = _view(turns, checkpoint=compacted.checkpoint, limit=10000).compact()
    assert repeated.checkpoint.reduced_through == "recent"
    assert repeated.messages == compacted.messages


@pytest.mark.parametrize("fields", [{"producer": "external"}, {"format_version": 99}])
def test_unrecognized_checkpoint_replays_original_before_recomputing(fields):
    """未知 CPU 或格式不能令原始历史静默消失。"""
    turns = (_read_turn("old", 6000),)
    checkpoint = make_compaction_checkpoint(reduced_through="old", **fields)
    view = _view(turns, checkpoint=checkpoint)
    assert view.messages[1]["content"] == "x" * 6000
    assert view.should_compact() is True
    assert view.compact().checkpoint.producer == "alice"


def test_disabled_compaction_ignores_checkpoint_and_force():
    """关闭压缩时保留全部原文，内部强制参数也不能绕过开关。"""
    turns = (_read_turn("old", 6000),)
    view = _view(
        turns,
        checkpoint=make_compaction_checkpoint(reduced_through="old"),
        config=CompactionConfig(enabled=False),
    )
    assert view.messages[1]["content"] == "x" * 6000
    assert view.should_compact(force=True) is False
