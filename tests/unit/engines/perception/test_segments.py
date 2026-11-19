"""一轮交互的切段：正文开启新段、无正文的响应并入、过长时在响应边界切、子 Agent 消息并入
其后的段、没有响应序号的轮次整轮一段；各段的事件区间按序号首尾相接，覆盖整轮。"""

from __future__ import annotations

from hivememory.core.models import TurnEvent
from hivememory.engines.perception.segments import TurnSegment, segment_turn

_USER = TurnEvent(kind="user_message", sequence=0, content="问题")


def _text(sequence: int, response: int | None, content: str = "正文") -> TurnEvent:
    return TurnEvent(
        kind="assistant_message", sequence=sequence, content=content, response_index=response
    )


def _thought(sequence: int, response: int) -> TurnEvent:
    return TurnEvent(kind="thought", sequence=sequence, content="想一想", response_index=response)


def _call(sequence: int, response: int | None) -> TurnEvent:
    return TurnEvent(
        kind="tool_call",
        sequence=sequence,
        content="",
        response_index=response,
        action_id=f"a{sequence}",
        tool_name="Bash",
    )


def _subagent(sequence: int) -> TurnEvent:
    return TurnEvent(kind="subagent_message", sequence=sequence, content="子 Agent 的报告")


def _ranges(segments: list[TurnSegment]) -> list[tuple[int, int]]:
    return [(segment.first_sequence, segment.last_sequence) for segment in segments]


def test_responses_with_text_or_thought_open_segments_and_the_rest_join():
    """第一段从用户消息开始；带正文或思考的响应开启新段，只有调用或空正文的响应并入当前段。"""
    events = [
        _USER,
        _call(1, 1),
        _text(2, 2),
        _call(3, 2),
        _call(4, 3),
        _text(5, 4, content="  "),
        _thought(6, 5),
        _call(7, 5),
    ]

    assert _ranges(segment_turn(events)) == [(0, 1), (2, 5), (6, 7)]


def test_overlong_segments_are_cut_only_at_response_boundaries():
    """一段已有上限次响应时，下一次响应开启新段；同一次响应的事件不被拆开。"""
    events = [
        _USER,
        _text(1, 1),
        _call(2, 1),
        _call(3, 2),
        _call(4, 2),
        _call(5, 3),
        _call(6, 4),
    ]

    assert _ranges(segment_turn(events, max_responses=2)) == [(0, 4), (5, 6)]


def test_subagent_messages_join_the_following_segment_or_the_last_one():
    """子 Agent 消息没有响应序号，并入其后的段；位于末尾时并入最后一段。"""
    events = [
        _USER,
        _call(1, 1),
        _subagent(2),
        _subagent(3),
        _text(4, 2),
        _subagent(5),
    ]

    assert _ranges(segment_turn(events)) == [(0, 1), (2, 5)]


def test_turns_without_response_indexes_are_one_segment():
    """来源没有记下响应边界的轮次（升级前的记录、Codex）整轮一段。"""
    events = [_USER, _text(1, None), _call(2, None), _text(3, None)]

    assert _ranges(segment_turn(events)) == [(0, 3)]
    assert _ranges(segment_turn([_USER])) == [(0, 0)]
