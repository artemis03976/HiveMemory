"""
MTPTraceReducer 单测

覆盖:
- 空列表 → 空 traces
- assistant_message / tool_result 类型 → 全部过滤
- READ / SEARCH / RUN tool_call → 正确 TraceItem
- WRITE / UPDATE / CALL → 过滤
- dict 输入（模拟 SSE 序列化路径）→ 与对象输入结果相同
- 混合事件列表 → 只保留 READ/SEARCH/RUN
"""

import pytest
from hivememory.core.models import TraceItem, TurnEvent
from hivememory.patchouli.mtp.trace_reducer import MTPTraceReducer


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


class TestMTPTraceReducerEmpty:
    def test_empty_list(self):
        assert MTPTraceReducer.reduce([]) == []

    def test_all_assistant_message(self):
        events = [_event("assistant_message"), _event("assistant_message")]
        assert MTPTraceReducer.reduce(events) == []

    def test_all_tool_result(self):
        events = [_event("tool_result", tool_kind="READ", status="success")]
        assert MTPTraceReducer.reduce(events) == []


class TestMTPTraceReducerCommandMapping:
    def test_read_command(self):
        events = [_event("tool_call", tool_kind="READ", target="my_alias")]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 1
        assert traces[0].action == "READ"
        assert traces[0].target == "my_alias"

    def test_search_command_no_query(self):
        events = [_event("tool_call", tool_kind="SEARCH", target="*", content="")]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 1
        assert traces[0].action == "SEARCH"
        assert traces[0].query is None

    def test_search_command_uses_tool_args_query(self):
        events = [
            _event_dict(
                "tool_call",
                tool_kind="SEARCH",
                target="*",
                content='⟪ SEARCH | * | query="authentication flow" ⟫',
                tool_args={"query": "authentication flow"},
            )
        ]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 1
        assert traces[0].action == "SEARCH"
        assert traces[0].query == "authentication flow"

    def test_search_command_without_tool_args_no_longer_parses_content(self):
        events = [
            _event_dict(
                "tool_call",
                tool_kind="SEARCH",
                target="*",
                content='⟪ SEARCH | * | query="authentication flow" ⟫',
            )
        ]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 1
        assert traces[0].query is None

    def test_run_command(self):
        events = [_event("tool_call", tool_kind="RUN", target="git_log", status="success")]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 1
        assert traces[0].action == "RUN"
        assert traces[0].tool == "git_log"
        assert traces[0].status == "success"

    def test_run_command_no_status(self):
        events = [_event("tool_call", tool_kind="RUN", target="tool_x")]
        traces = MTPTraceReducer.reduce(events)
        assert traces[0].status == "unknown"


class TestMTPTraceReducerFiltered:
    @pytest.mark.parametrize("tool_kind", ["WRITE", "UPDATE", "CALL"])
    def test_filtered_tool_kinds(self, tool_kind):
        events = [_event("tool_call", tool_kind=tool_kind, target="something")]
        assert MTPTraceReducer.reduce(events) == []

    def test_unknown_tool_kind(self):
        events = [_event("tool_call", tool_kind="UNKNOWN")]
        assert MTPTraceReducer.reduce(events) == []


class TestMTPTraceReducerDictInput:
    """模拟 chat_stream SSE 序列化后以 dict 重建的场景"""

    def test_dict_read(self):
        event_dict = _event_dict("tool_call", tool_kind="READ", target="alias_x")
        traces = MTPTraceReducer.reduce([event_dict])
        assert len(traces) == 1
        assert traces[0].action == "READ"
        assert traces[0].target == "alias_x"

    def test_dict_run(self):
        event_dict = _event_dict("tool_call", tool_kind="RUN", target="my_tool", status="error")
        traces = MTPTraceReducer.reduce([event_dict])
        assert traces[0].action == "RUN"
        assert traces[0].status == "error"

    def test_dict_assistant_message_filtered(self):
        event_dict = {"kind": "assistant_message", "sequence": 0, "role": "assistant", "content": "hi"}
        assert MTPTraceReducer.reduce([event_dict]) == []

    def test_dict_write_filtered(self):
        event_dict = _event_dict("tool_call", tool_kind="WRITE")
        assert MTPTraceReducer.reduce([event_dict]) == []


class TestMTPTraceReducerMixedList:
    def test_mixed_events_order_preserved(self):
        events = [
            _event("assistant_message", sequence=0),
            _event("tool_call", tool_kind="READ", target="a1", sequence=1, action_id="read-1"),
            _event("tool_result", tool_kind="READ", status="success", sequence=2, action_id="read-1"),
            _event(
                "tool_call",
                tool_kind="SEARCH",
                target="*",
                content='⟪ SEARCH | * | query="test" ⟫',
                sequence=3,
                action_id="search-1",
            ),
            _event("tool_call", tool_kind="WRITE", sequence=4, action_id="write-1"),
            _event("tool_call", tool_kind="RUN", target="tool_a", status="success", sequence=5, action_id="run-1"),
        ]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 3
        assert traces[0].action == "READ"
        assert traces[1].action == "SEARCH"
        assert traces[2].action == "RUN"
