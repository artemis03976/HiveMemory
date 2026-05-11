"""
MTPTraceReducer 单测

覆盖:
- 空列表 → 空 traces
- assistant_text / mtp_result 类型 → 全部过滤
- READ / SEARCH / RUN mtp_command → 正确 TraceItem
- WRITE / UPDATE / CALL → 过滤
- dict 输入（模拟 SSE 序列化路径）→ 与对象输入结果相同
- 混合事件列表 → 只保留 READ/SEARCH/RUN
"""

import pytest
from hivememory.core.models import TraceItem, TurnEvent
from hivememory.patchouli.mtp.trace_reducer import MTPTraceReducer


def _event(kind, verb=None, target=None, status=None, content=""):
    normalized_kind = {
        "assistant_text": "assistant_message",
        "mtp_command": "tool_call",
        "mtp_result": "tool_result",
    }.get(kind, kind)
    normalized_role = "assistant" if normalized_kind != "tool_result" else "user"
    return TurnEvent(
        kind=normalized_kind,
        sequence=0,
        role=normalized_role,
        content=content,
        tool_kind=verb,
        tool_name=target if verb == "RUN" else None,
        target=target,
        status=status,
    )


class TestMTPTraceReducerEmpty:
    def test_empty_list(self):
        assert MTPTraceReducer.reduce([]) == []

    def test_all_assistant_text(self):
        events = [_event("assistant_message"), _event("assistant_message")]
        assert MTPTraceReducer.reduce(events) == []

    def test_all_mtp_result(self):
        events = [_event("tool_result", verb="READ", status="success")]
        assert MTPTraceReducer.reduce(events) == []


class TestMTPTraceReducerCommandMapping:
    def test_read_command(self):
        events = [_event("tool_call", verb="READ", target="my_alias")]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 1
        assert traces[0].action == "READ"
        assert traces[0].target == "my_alias"

    def test_search_command_no_query(self):
        # content が空なので query は None になるはず
        events = [_event("tool_call", verb="SEARCH", target="*", content="")]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 1
        assert traces[0].action == "SEARCH"
        assert traces[0].query is None

    def test_search_command_with_full_mtp_content(self):
        content = '⟪ SEARCH | * | query="authentication flow" ⟫'
        events = [_event("tool_call", verb="SEARCH", target="*", content=content)]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 1
        assert traces[0].action == "SEARCH"
        assert traces[0].query == "authentication flow"

    def test_run_command(self):
        events = [_event("tool_call", verb="RUN", target="git_log", status="success")]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 1
        assert traces[0].action == "RUN"
        assert traces[0].tool == "git_log"
        assert traces[0].status == "success"

    def test_run_command_no_status(self):
        events = [_event("tool_call", verb="RUN", target="tool_x")]
        traces = MTPTraceReducer.reduce(events)
        assert traces[0].status == "unknown"


class TestMTPTraceReducerFiltered:
    @pytest.mark.parametrize("verb", ["WRITE", "UPDATE", "CALL"])
    def test_filtered_verbs(self, verb):
        events = [_event("tool_call", verb=verb, target="something")]
        assert MTPTraceReducer.reduce(events) == []

    def test_unknown_verb(self):
        events = [_event("tool_call", verb="UNKNOWN")]
        assert MTPTraceReducer.reduce(events) == []


class TestMTPTraceReducerDictInput:
    """模拟 chat_stream SSE 序列化后以 dict 重建的场景"""

    def test_dict_read(self):
        event_dict = {
            "kind": "tool_call",
            "sequence": 0,
            "role": "assistant",
            "content": "",
            "tool_kind": "READ",
            "target": "alias_x",
            "status": None,
        }
        traces = MTPTraceReducer.reduce([event_dict])
        assert len(traces) == 1
        assert traces[0].action == "READ"
        assert traces[0].target == "alias_x"

    def test_dict_run(self):
        event_dict = {
            "kind": "tool_call",
            "sequence": 0,
            "role": "assistant",
            "content": "",
            "tool_kind": "RUN",
            "tool_name": "my_tool",
            "target": "my_tool",
            "status": "error",
        }
        traces = MTPTraceReducer.reduce([event_dict])
        assert traces[0].action == "RUN"
        assert traces[0].status == "error"

    def test_dict_assistant_text_filtered(self):
        event_dict = {"kind": "assistant_message", "sequence": 0, "role": "assistant", "content": "hi"}
        assert MTPTraceReducer.reduce([event_dict]) == []

    def test_dict_write_filtered(self):
        event_dict = {"kind": "tool_call", "sequence": 0, "role": "assistant", "content": "", "tool_kind": "WRITE"}
        assert MTPTraceReducer.reduce([event_dict]) == []


class TestMTPTraceReducerMixedList:
    def test_mixed_events_order_preserved(self):
        events = [
            _event("assistant_message"),
            _event("tool_call", verb="READ", target="a1"),
            _event("tool_result", verb="READ", status="success"),
            _event("tool_call", verb="SEARCH", target="*", content='⟪ SEARCH | * | query="test" ⟫'),
            _event("tool_call", verb="WRITE"),
            _event("tool_call", verb="RUN", target="tool_a", status="success"),
        ]
        traces = MTPTraceReducer.reduce(events)
        assert len(traces) == 3
        assert traces[0].action == "READ"
        assert traces[1].action == "SEARCH"
        assert traces[2].action == "RUN"
