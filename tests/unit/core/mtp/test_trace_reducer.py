"""MTPTraceReducer 门面的输入适配：空输入、dict 输入与 TurnEvent 到 TraceItem 的委托链路。

逐动词的摘要规则由 TraceReducer 测试守护，事件到动作的归并由 ActionReducer 测试守护。
"""

from hivememory.core.models import TurnEvent
from hivememory.core.mtp.trace_reducer import MTPTraceReducer


def _event(kind, tool_kind=None, target=None, status=None, content="", sequence=0, action_id=None):
    normalized_role = "assistant" if kind != "tool_result" else "user"
    return TurnEvent(
        kind=kind,
        sequence=sequence,
        role=normalized_role,
        content=content,
        action_id=action_id,
        tool_kind=tool_kind,
        tool_name=target if tool_kind == "RUN" else None,
        target=target,
        status=status,
    )


def _event_dict(kind, tool_kind=None, target=None, status=None, content="", tool_args=None):
    return {
        "kind": kind,
        "sequence": 0,
        "role": "assistant" if kind != "tool_result" else "user",
        "content": content,
        "tool_kind": tool_kind,
        "tool_name": target if tool_kind == "RUN" else None,
        "target": target,
        "status": status,
        "tool_args": tool_args,
    }


class TestMTPTraceReducerInputAdaptation:
    def test_empty_list(self):
        assert MTPTraceReducer.reduce([]) == []

    def test_orphan_tool_result_produces_no_trace(self):
        events = [_event("tool_result", tool_kind="READ", status="success")]
        assert MTPTraceReducer.reduce(events) == []

    def test_dict_events_are_reduced_like_turn_events(self):
        """模拟 chat_stream SSE 序列化后以 dict 重建的场景。"""
        event_dict = _event_dict("tool_call", tool_kind="READ", target="alias_x")
        traces = MTPTraceReducer.reduce([event_dict])
        assert len(traces) == 1
        assert traces[0].action == "READ"
        assert traces[0].target == "alias_x"


class TestMTPTraceReducerMixedList:
    def test_mixed_events_order_preserved(self):
        events = [
            _event("assistant_message", sequence=0),
            _event("tool_call", tool_kind="READ", target="a1", sequence=1, action_id="read-1"),
            _event(
                "tool_result", tool_kind="READ", status="success", sequence=2, action_id="read-1"
            ),
            _event(
                "tool_call",
                tool_kind="SEARCH",
                target="*",
                content='⟪ SEARCH | * | query="test" ⟫',
                sequence=3,
                action_id="search-1",
            ),
            _event("tool_call", tool_kind="WRITE", sequence=4, action_id="write-1"),
            _event(
                "tool_call",
                tool_kind="RUN",
                target="tool_a",
                status="success",
                sequence=5,
                action_id="run-1",
            ),
        ]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 4
        assert traces[0].action == "READ"
        assert traces[1].action == "SEARCH"
        assert traces[2].action == "WRITE"
        assert traces[3].action == "RUN"
