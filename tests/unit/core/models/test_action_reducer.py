"""
ActionReducer 单元测试

覆盖:
- 空输入
- tool_call + tool_result 正常聚合，结局取自结果事件
- 能力调用以作用对象作为摘要名，外部工具保留工具名
- thought 归并到最近动作
- 非动作类事件被忽略
- 序列化后的 dict 事件与 TurnEvent 等价
"""

from hivememory.core.models import ActionReducer, TurnEvent


def _call(sequence: int, action_id: str, **fields) -> TurnEvent:
    return TurnEvent(kind="tool_call", sequence=sequence, content="", action_id=action_id, **fields)


def _result(sequence: int, action_id: str, status: str = "success") -> TurnEvent:
    return TurnEvent(
        kind="tool_result", sequence=sequence, content="result", action_id=action_id, status=status
    )


def test_reduce_empty_events():
    assert ActionReducer.reduce([]) == []


def test_reduce_single_tool_call_and_result():
    events = [_call(1, "a1", verb="READ", targets=("alias_x",)), _result(2, "a1")]

    actions = ActionReducer.reduce(events)

    assert len(actions) == 1
    action = actions[0]
    assert (action.action_id, action.tool_kind, action.tool_name) == ("a1", "READ", "alias_x")
    assert action.status == "success"
    assert [result.content for result in action.results] == ["result"]


def test_reduce_names_capability_calls_by_targets_and_external_tools_by_name():
    """能力调用没有工具名，以全部作用对象作为摘要名；外部工具保留工具名与参数。"""
    events = [
        _call(1, "read", verb="READ", targets=("fact_a", "fact_b")),
        _result(2, "read"),
        _call(3, "bash", tool_name="Bash", tool_args={"command": "ls"}),
        _result(4, "bash", status="error"),
    ]

    actions = ActionReducer.reduce(events)

    assert [(a.tool_kind, a.tool_name, a.status) for a in actions] == [
        ("READ", "fact_a, fact_b", "success"),
        ("", "Bash", "error"),
    ]
    assert dict(actions[1].tool_args or {}) == {"command": "ls"}


def test_reduce_thought_attaches_to_latest_action():
    events = [
        _call(1, "a1", verb="RUN", targets=("tool_x",)),
        TurnEvent(kind="thought", sequence=2, content="先检查参数"),
        _result(3, "a1"),
    ]

    actions = ActionReducer.reduce(events)

    assert len(actions) == 1
    assert actions[0].thought == "先检查参数"


def test_reduce_ignores_non_action_events():
    events = [
        TurnEvent(kind="user_message", sequence=0, content="hello"),
        _call(1, "a1", verb="READ", targets=("alias_x",)),
        _result(2, "a1"),
        TurnEvent(kind="assistant_message", sequence=3, content="总结"),
    ]

    actions = ActionReducer.reduce(events)

    assert [action.tool_kind for action in actions] == ["READ"]


def test_reduce_accepts_serialized_event_dicts():
    """SSE 序列化后以 dict 重建的事件按 TurnEvent 同样归并。"""
    event = _call(0, "a1", verb="READ", targets=("alias_x",))

    actions = ActionReducer.reduce([event.model_dump()])

    assert [(a.action_id, a.tool_kind, a.tool_name) for a in actions] == [("a1", "READ", "alias_x")]
