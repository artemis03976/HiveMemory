"""
ActionReducer 单元测试

覆盖:
- 空输入
- tool_call + tool_result 正常聚合
- 多个结果事件归并到同一动作
- thought 归并到最近动作
- 无 action_id 的 tool_call 使用 sequence 生成兜底键
- 非动作类事件被忽略
"""

from hivememory.core.models import ActionReducer, TurnEvent


def _ev(
    kind: str,
    sequence: int,
    *,
    content: str = "",
    action_id: str | None = None,
    tool_kind: str | None = None,
    tool_name: str | None = None,
    status: str | None = None,
) -> TurnEvent:
    role = "assistant" if kind != "tool_result" else "user"
    return TurnEvent(
        kind=kind,
        sequence=sequence,
        role=role,
        content=content,
        action_id=action_id,
        tool_kind=tool_kind,
        tool_name=tool_name,
        status=status,
    )


def test_reduce_empty_events():
    assert ActionReducer.reduce([]) == []


def test_reduce_single_tool_call_and_result():
    events = [
        _ev("tool_call", 1, content="⟪ READ | alias_x ⟫", action_id="a1", tool_kind="READ", tool_name="alias_x"),
        _ev("tool_result", 2, content="result", action_id="a1", tool_kind="READ", tool_name="alias_x", status="success"),
    ]

    actions = ActionReducer.reduce(events)

    assert len(actions) == 1
    action = actions[0]
    assert action.action_id == "a1"
    assert action.tool_kind == "READ"
    assert action.tool_name == "alias_x"
    assert action.status == "success"
    assert len(action.results) == 1
    assert action.is_complete is True


def test_reduce_multiple_results_same_action():
    events = [
        _ev("tool_call", 1, action_id="a1", tool_kind="CALL", tool_name="sub_agent"),
        _ev("tool_result", 2, content="partial", action_id="a1", tool_kind="CALL"),
        _ev("tool_result", 3, content="final", action_id="a1", tool_kind="CALL", status="success"),
    ]

    actions = ActionReducer.reduce(events)

    assert len(actions) == 1
    assert [result.content for result in actions[0].results] == ["partial", "final"]
    assert actions[0].status == "success"


def test_reduce_thought_attaches_to_latest_action():
    events = [
        _ev("tool_call", 1, action_id="a1", tool_kind="RUN", tool_name="tool_x"),
        _ev("thought", 2, content="先检查参数"),
        _ev("tool_result", 3, content="ok", action_id="a1", tool_kind="RUN", status="success"),
    ]

    actions = ActionReducer.reduce(events)

    assert len(actions) == 1
    assert actions[0].thought == "先检查参数"


def test_reduce_tool_call_without_action_id_uses_sequence_fallback():
    events = [
        _ev("tool_call", 7, tool_kind="READ", tool_name="alias_x"),
        _ev("tool_result", 8, content="orphan result"),
    ]

    actions = ActionReducer.reduce(events)

    assert len(actions) == 1
    assert actions[0].action_id == "tool_call_7"
    assert actions[0].results[0].content == "orphan result"


def test_reduce_ignores_non_action_events():
    events = [
        _ev("assistant_message", 0, content="hello"),
        _ev("tool_call", 1, action_id="a1", tool_kind="READ", tool_name="alias_x"),
        _ev("tool_result", 2, content="done", action_id="a1", tool_kind="READ"),
        _ev("assistant_message", 3, content="总结"),
    ]

    actions = ActionReducer.reduce(events)

    assert len(actions) == 1
    assert actions[0].tool_kind == "READ"
