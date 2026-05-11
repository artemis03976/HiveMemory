"""
TraceReducer 单元测试

覆盖:
- READ / SEARCH / RUN 的摘要提炼
- WRITE / UPDATE / CALL 过滤
- SEARCH 优先从 tool_args.query 提取
- action_id 透传到 TraceItem
"""

from hivememory.core.models import AgentAction, TraceReducer


def _action(
    action_id: str,
    tool_kind: str,
    *,
    tool_name: str = "",
    tool_args: dict | None = None,
    status: str | None = None,
) -> AgentAction:
    return AgentAction(
        action_id=action_id,
        tool_kind=tool_kind,
        tool_name=tool_name,
        tool_args=tool_args,
        status=status,
    )


def test_reduce_read_action():
    traces = TraceReducer.reduce([_action("a1", "READ", tool_name="alias_x")])
    assert len(traces) == 1
    assert traces[0].action == "READ"
    assert traces[0].action_id == "a1"
    assert traces[0].target == "alias_x"


def test_reduce_search_action_uses_tool_args_query():
    traces = TraceReducer.reduce(
        [_action("a2", "SEARCH", tool_name="*", tool_args={"query": "authentication flow"})]
    )
    assert len(traces) == 1
    assert traces[0].action == "SEARCH"
    assert traces[0].query == "authentication flow"


def test_reduce_run_action():
    traces = TraceReducer.reduce([_action("a3", "RUN", tool_name="git_log", status="success")])
    assert len(traces) == 1
    assert traces[0].action == "RUN"
    assert traces[0].tool == "git_log"
    assert traces[0].status == "success"


def test_reduce_unknown_status_defaults_for_run():
    traces = TraceReducer.reduce([_action("a4", "RUN", tool_name="tool_x")])
    assert traces[0].status == "unknown"


def test_reduce_filters_control_actions():
    traces = TraceReducer.reduce(
        [
            _action("a5", "WRITE", tool_name="memo"),
            _action("a6", "UPDATE", tool_name="memo"),
            _action("a7", "CALL", tool_name="sub_agent"),
        ]
    )
    assert traces == []
