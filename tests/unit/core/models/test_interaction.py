"""轮次事件的单元测试：单个事件的字段规则、一轮事件集的不变量与回复正文。"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from hivememory.core.models import TurnEvent, turn_reply_text, validate_turn_events


def _user(sequence: int = 0) -> TurnEvent:
    return TurnEvent(kind="user_message", sequence=sequence, content="问题")


def _call(sequence: int, action_id: str = "a1", **fields: Any) -> TurnEvent:
    return TurnEvent(kind="tool_call", sequence=sequence, content="", action_id=action_id, **fields)


def _result(sequence: int, action_id: str = "a1") -> TurnEvent:
    return TurnEvent(
        kind="tool_result", sequence=sequence, content="ok", action_id=action_id, status="success"
    )


class TestTurnEventFields:
    def test_capability_and_external_calls_share_one_shape(self):
        """能力调用写动词与作用对象，外部工具写工具名；无法解析的 MTP 指令二者都为空。"""
        read = _call(1, verb="READ", targets=("fact_a", "fact_b"))
        bash = _call(2, "a2", tool_name="Bash", tool_args={"command": "ls"})
        unparsed = TurnEvent(kind="tool_call", sequence=3, content="⟪ READ ⟫", action_id="a3")

        assert (read.verb, read.targets, read.tool_name) == ("READ", ("fact_a", "fact_b"), None)
        assert (bash.verb, bash.tool_name, dict(bash.tool_args or {})) == (
            None,
            "Bash",
            {"command": "ls"},
        )
        assert (unparsed.verb, unparsed.tool_name, unparsed.targets) == (None, None, ())

    @pytest.mark.parametrize(
        ("kind", "fields", "match"),
        [
            ("tool_result", {"verb": "READ", "status": "success"}, "cannot carry: verb"),
            ("tool_call", {"status": "success"}, "cannot carry: status"),
            ("tool_call", {"produced_aliases": ("x",)}, "cannot carry: produced_aliases"),
            ("assistant_message", {"action_id": "a1"}, "cannot carry: action_id"),
        ],
        ids=["call-field-on-result", "status-on-call", "produced-on-call", "action-id-on-text"],
    )
    def test_fields_only_appear_on_their_event_kind(self, kind, fields, match):
        """调用一方的事实只在调用上，结局一方的事实只在结果上，正文不带配对标识。"""
        fields.setdefault("action_id", "a1")
        with pytest.raises(ValidationError, match=match):
            TurnEvent(kind=kind, sequence=1, content="", **fields)

    @pytest.mark.parametrize(
        ("fields", "match"),
        [
            ({"kind": "tool_call"}, "tool_call event requires action_id"),
            ({"kind": "tool_result", "status": "success"}, "tool_result event requires action_id"),
            ({"kind": "tool_result", "action_id": "a1"}, "tool_result event requires status"),
            (
                {"kind": "tool_call", "action_id": "a1", "verb": "READ", "tool_name": "Read"},
                "both verb and tool_name",
            ),
            (
                {"kind": "tool_call", "action_id": "a1", "targets": ("x",)},
                "without verb cannot carry targets",
            ),
        ],
        ids=[
            "call-without-action-id",
            "result-without-action-id",
            "result-without-status",
            "verb-and-tool-name",
            "targets-without-verb",
        ],
    )
    def test_tool_events_require_pairing_status_and_one_callee(self, fields, match):
        with pytest.raises(ValidationError, match=match):
            TurnEvent(sequence=1, content="", **fields)

    @pytest.mark.parametrize(
        "fields",
        [{"role": "assistant"}, {"status": "failed"}],
        ids=["removed-field", "status-outside-vocabulary"],
    )
    def test_removed_fields_and_unknown_statuses_are_rejected(self, fields):
        """事件只接受当前字段集；状态只取 success、ack、error、cancelled。"""
        with pytest.raises(ValidationError):
            TurnEvent(kind="tool_result", sequence=1, content="", action_id="a1", **fields)


class TestValidateTurnEvents:
    def test_accepts_turn_with_unfinished_call(self):
        """没有结果的调用表示调用没有完成，是合法的一轮。"""
        validate_turn_events([_user(), _call(1), _result(2), _call(3, "a2")])

    @pytest.mark.parametrize(
        ("events", "match"),
        [
            ([], "start with the user message"),
            ([_call(0)], "start with the user message"),
            ([_user(1)], "start with the user message"),
            ([_user(), _user(1)], "exactly one user message"),
            ([_user(), _call(2), _result(2)], "strictly increasing"),
            ([_user(), _call(1), _call(2)], "duplicate tool_call action_id"),
            ([_user(), _result(1)], "without an earlier tool_call"),
            ([_user(), _call(1), _result(2), _result(3)], "more than one result"),
        ],
        ids=[
            "empty",
            "missing-user-message",
            "user-message-not-at-zero",
            "second-user-message",
            "repeated-sequence",
            "duplicate-call",
            "result-before-call",
            "second-result",
        ],
    )
    def test_rejects_turns_that_break_an_invariant(self, events, match):
        with pytest.raises(ValueError, match=match):
            validate_turn_events(events)


def test_reply_text_joins_assistant_messages_in_sequence_order_with_blank_lines():
    """回复正文只取助手正文，按序号以空行拼接；没有助手正文时为空串。"""
    events = [
        _user(),
        TurnEvent(kind="assistant_message", sequence=4, content="结论"),
        TurnEvent(kind="thought", sequence=1, content="思考"),
        TurnEvent(kind="assistant_message", sequence=2, content="先读一下"),
        _call(3, verb="READ", targets=("fact_a",)),
    ]

    assert turn_reply_text(events) == "先读一下\n\n结论"
    assert turn_reply_text([_user()]) == ""
