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
        ("fields", "match"),
        [
            ({"status": "success", "result": "ok"}, "cannot carry: result, status"),
            ({"action_id": "a1"}, "cannot carry: action_id"),
        ],
        ids=["outcome-on-text", "action-id-on-text"],
    )
    def test_call_fields_only_appear_on_tool_calls(self, fields, match):
        """调用标识与结局只在工具调用上，正文不带。"""
        with pytest.raises(ValidationError, match=match):
            TurnEvent(kind="assistant_message", sequence=1, content="", **fields)

    @pytest.mark.parametrize(
        ("fields", "match"),
        [
            ({}, "tool_call event requires action_id"),
            ({"action_id": "a1", "verb": "READ", "tool_name": "Read"}, "both verb and tool_name"),
            ({"action_id": "a1", "targets": ("x",)}, "without verb cannot carry targets"),
            ({"action_id": "a1", "status": "success"}, "status and result together"),
            ({"action_id": "a1", "result": "ok"}, "status and result together"),
            (
                {"action_id": "a1", "produced_aliases": ("x",)},
                "without outcome cannot carry produced_aliases",
            ),
        ],
        ids=[
            "without-action-id",
            "verb-and-tool-name",
            "targets-without-verb",
            "status-without-result",
            "result-without-status",
            "produced-without-outcome",
        ],
    )
    def test_tool_call_requires_identity_one_callee_and_whole_outcome(self, fields, match):
        with pytest.raises(ValidationError, match=match):
            TurnEvent(kind="tool_call", sequence=1, content="", **fields)

    @pytest.mark.parametrize(
        "fields",
        [{"role": "assistant"}, {"status": "failed", "result": "x"}],
        ids=["removed-field", "status-outside-vocabulary"],
    )
    def test_removed_fields_and_unknown_statuses_are_rejected(self, fields):
        """事件只接受当前字段集；状态只取 success、ack、error、cancelled。"""
        with pytest.raises(ValidationError):
            TurnEvent(kind="tool_call", sequence=1, content="", action_id="a1", **fields)


class TestWithOutcome:
    def test_outcome_completes_the_call_in_place_of_the_original(self):
        """补全只添加结局，调用一方的字段与序号不变。"""
        call = _call(3, verb="WRITE", tool_args={"content": "x"})

        done = call.with_outcome("ack", "已登记", ("draft_a",))

        assert (done.sequence, done.action_id, done.verb, dict(done.tool_args or {})) == (
            3,
            "a1",
            "WRITE",
            {"content": "x"},
        )
        assert (done.status, done.result, done.produced_aliases) == ("ack", "已登记", ("draft_a",))

    def test_outcome_is_recorded_only_once(self):
        """已有结局的调用不能再补全，避免把两次返回记到同一个调用上。"""
        done = _call(1).with_outcome("success", "ok")

        with pytest.raises(ValueError, match="already has an outcome"):
            done.with_outcome("error", "again")


class TestValidateTurnEvents:
    def test_accepts_turn_with_call_without_outcome(self):
        """没有结局的调用表示结局没有记录，是合法的一轮。"""
        validate_turn_events([_user(), _call(1, status="success", result="ok"), _call(2, "a2")])

    @pytest.mark.parametrize(
        ("events", "match"),
        [
            ([], "start with the user message"),
            ([_call(0)], "start with the user message"),
            ([_user(1)], "start with the user message"),
            ([_user(), _user(1)], "exactly one user message"),
            ([_user(), _call(1), _call(1, "a2")], "strictly increasing"),
            ([_user(), _call(1), _call(2)], "duplicate tool_call action_id"),
        ],
        ids=[
            "empty",
            "missing-user-message",
            "user-message-not-at-zero",
            "second-user-message",
            "repeated-sequence",
            "duplicate-call",
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
