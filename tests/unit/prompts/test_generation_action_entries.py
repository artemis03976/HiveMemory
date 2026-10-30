"""记忆生成视图动作段的单元测试。

被测对象：``render_action_entries`` 把一轮的思考、外部工具动作与 MTP 动作
渲染为动作段条目。保护的契约：
- 思考按出现顺序在事件位置单独成条；
- 外部工具动作渲染工具名、参数摘要与各条结果，出错的结果标明状态；
- 思考、参数与结果按各自上限截断，保留首尾并标出省略量；
- 超出单轮上限时保留思考、出错的动作与最近的动作，其余外部工具动作降为一行；
  降为一行仍超出时按优先级省略整条，结果不超过上限；
- 空白的思考不渲染；
- 能力动作维持一行摘要，不渲染结果，CALL/WRITE/UPDATE 与无法解析的指令不渲染；
- 只有用户消息与最终回复的轮次没有动作段。

测试中的动作与轨迹由 core 的归约器从事件得到，与感知建块的方式一致。
"""

from __future__ import annotations

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import (
    ActionReducer,
    ActorIdentity,
    TraceItem,
    TraceReducer,
    TurnEvent,
    TurnRecord,
)
from hivememory.prompts.transcript.actions import ActionEntry, render_action_entries


def _turn(events: list[TurnEvent], *, traces: list[TraceItem] | None = None) -> TurnRecord:
    actions = ActionReducer.reduce(events)
    return TurnRecord(
        identity=ActorIdentity(user_id="u1", agent_id="claude_code"),
        user_query="修一下测试",
        assistant_final_text="已修复",
        turn_events=tuple(events),
        actions=tuple(actions),
        semantic_traces=tuple(TraceReducer.reduce(actions) if traces is None else traces),
    )


def _thought(sequence: int, content: str) -> TurnEvent:
    return TurnEvent(kind="thought", sequence=sequence, content=content)


def _call(sequence: int, action_id: str, tool_name: str | None = "Bash", **kwargs) -> TurnEvent:
    return TurnEvent(
        kind="tool_call",
        sequence=sequence,
        content="",
        action_id=action_id,
        tool_name=tool_name,
        **kwargs,
    )


def _result(sequence: int, action_id: str, content: str, status: str = "success") -> TurnEvent:
    return TurnEvent(
        kind="tool_result",
        sequence=sequence,
        content=content,
        action_id=action_id,
        status=status,
    )


def _texts(entries: list[ActionEntry]) -> list[str]:
    return [entry.text for entry in entries]


DEFAULTS = GenerationTranscriptConfig()


def test_unattached_thoughts_keep_their_position_between_actions():
    """未附属的思考在事件位置单独成条，不被吸附到上一个动作的结果之后。"""
    turn = _turn(
        [
            _thought(1, "先跑测试"),
            _call(2, "toolu-1", tool_args={"command": "pytest -q"}),
            _result(3, "toolu-1", "1 failed"),
            _thought(4, "断言写反了，改一下"),
            _call(5, "toolu-2", tool_name="Edit", tool_args={"file_path": "test_a.py"}),
            _result(6, "toolu-2", "ok"),
        ]
    )

    assert _texts(render_action_entries(turn, DEFAULTS)) == [
        "THOUGHT: 先跑测试",
        'TOOL: Bash {"command": "pytest -q"}\nRESULT: 1 failed',
        "THOUGHT: 断言写反了，改一下",
        'TOOL: Edit {"file_path": "test_a.py"}\nRESULT: ok',
    ]


def test_error_results_are_marked():
    """出错的结果标明状态。"""
    turn = _turn(
        [
            _call(1, "toolu-1", tool_name="Read", tool_args={"file_path": "a.py"}),
            _result(2, "toolu-1", "permission denied", status="error"),
        ]
    )

    assert _texts(render_action_entries(turn, DEFAULTS)) == [
        'TOOL: Read {"file_path": "a.py"}\nRESULT (error): permission denied'
    ]


def test_thoughts_arguments_and_results_are_clipped_keeping_head_and_tail():
    """各部分按自身上限截断，保留首尾：长输出末尾的报错仍然可见。"""
    config = GenerationTranscriptConfig(
        max_thought_chars=10, max_args_chars=12, max_result_chars=20
    )
    turn = _turn(
        [
            _thought(1, "思" * 30),
            _call(2, "toolu-1", tool_args={"command": "x" * 50}),
            _result(3, "toolu-1", "o" * 200 + "Traceback: boom", status="error"),
        ]
    )

    thought, tool = _texts(render_action_entries(turn, config))
    tool_line, result_line = tool.split("\n")
    assert thought == "THOUGHT: 思思思思思 …[20 chars omitted]… 思思思思思"
    assert tool_line == 'TOOL: Bash {"comm …[53 chars omitted]… xxxx"}'
    assert result_line.startswith("RESULT (error): oooooooooo …[")
    assert result_line.endswith(": boom")


def test_turn_limit_keeps_thoughts_failed_and_recent_actions_before_older_ones():
    """超出单轮上限时，思考、出错的动作（即使最早）与最近的动作保留完整形态，其余降为一行。"""
    config = GenerationTranscriptConfig(max_result_chars=100, max_turn_action_chars=330)
    turn = _turn(
        [
            _thought(1, "计划"),
            _call(2, "failed", tool_args={"c": "1"}),
            _result(3, "failed", "a" * 100, status="error"),
            _call(4, "middle", tool_args={"c": "2"}),
            _result(5, "middle", "b" * 100),
            _call(6, "recent", tool_args={"c": "3"}),
            _result(7, "recent", "c" * 100),
        ]
    )

    thought, failed, middle, recent = _texts(render_action_entries(turn, config))

    assert thought == "THOUGHT: 计划"
    assert failed == 'TOOL: Bash {"c": "1"}\nRESULT (error): ' + "a" * 100
    assert middle == 'TOOL: Bash {"c": "2"}'
    assert recent == 'TOOL: Bash {"c": "3"}\nRESULT: ' + "c" * 100


def test_turn_limit_degrades_tools_before_touching_thoughts():
    """完整动作放不下时先降为一行；降级后放得下就不省略任何条目。"""
    config = GenerationTranscriptConfig(max_turn_action_chars=50)
    turn = _turn(
        [
            _thought(1, "第一步"),
            _call(2, "toolu-1", tool_args={"c": "1"}),
            _result(3, "toolu-1", "done", status="error"),
            _thought(4, "第二步"),
        ]
    )

    assert _texts(render_action_entries(turn, config)) == [
        "THOUGHT: 第一步",
        'TOOL: Bash {"c": "1"}',
        "THOUGHT: 第二步",
    ]


def test_turn_limit_is_hard_and_omits_lowest_priority_entries_first():
    """降为一行后仍超出时省略整条：先省略非出错动作，再省略出错动作，最后才是较早的思考。"""
    config = GenerationTranscriptConfig(max_turn_action_chars=200)
    turn = _turn(
        [
            _thought(1, "a" * 60),
            _call(2, "ok", tool_args={"c": "1"}),
            _result(3, "ok", "fine"),
            _call(4, "failed", tool_args={"c": "2"}),
            _result(5, "failed", "boom", status="error"),
            _thought(6, "b" * 60),
            _thought(7, "c" * 60),
        ]
    )

    entries = render_action_entries(turn, config)

    assert _texts(entries) == [
        "OMITTED: 3 entries over the per-turn limit",
        "THOUGHT: " + "b" * 60,
        "THOUGHT: " + "c" * 60,
    ]
    assert sum(len(entry.text) for entry in entries) <= 200


def test_blank_thoughts_are_not_rendered():
    """只有空白的思考（例如只有签名的思考块）不形成条目。"""
    turn = _turn(
        [
            _thought(1, "   "),
            _thought(2, ""),
            _call(3, "toolu-1", tool_args={"c": "1"}),
        ]
    )

    assert _texts(render_action_entries(turn, DEFAULTS)) == ['TOOL: Bash {"c": "1"}']


def test_capability_actions_keep_one_line_summaries_without_results():
    """能力动作只渲染一行摘要；结果、CALL/WRITE/UPDATE 与无法解析的指令不进入生成视图。

    READ 列出全部作用对象：外部 harness 经 MCP 读取与 MTP READ 写出同样的调用事件，
    摘要相同，不出现 MCP 调用名。
    """
    turn = _turn(
        [
            _call(1, "act-search", None, verb="SEARCH", tool_args={"query": "部署"}),
            _result(2, "act-search", "已有记忆正文"),
            _call(3, "act-read", None, verb="READ", targets=("fact_a", "fact_b")),
            _result(4, "act-read", "已有记忆正文"),
            _call(5, "act-run", None, verb="RUN", targets=("git_log",)),
            _result(6, "act-run", "commit abc"),
            _call(7, "act-write", None, verb="WRITE"),
            _result(8, "act-write", "已登记", status="ack"),
            _call(9, "act-unparsed", None),
            _result(10, "act-unparsed", "parse error", status="error"),
        ]
    )

    entries = render_action_entries(turn, DEFAULTS)

    assert [entry.kind for entry in entries] == ["trace", "trace", "trace"]
    assert _texts(entries) == [
        'SEARCH: "部署"',
        "READ: fact_a, fact_b",
        "RUN: git_log (success)",
    ]


def test_traces_without_events_render_in_trace_order():
    """不带结构化事件的块只有轨迹可用，按轨迹原顺序渲染。"""
    turn = _turn(
        [],
        traces=[TraceItem(action="READ", target="fact_x"), TraceItem(action="SEARCH", query="q")],
    )

    assert _texts(render_action_entries(turn, DEFAULTS)) == ["READ: fact_x", 'SEARCH: "q"']


def test_turn_with_only_messages_has_no_action_entries():
    """只有用户消息与最终回复的轮次没有动作段。"""
    turn = _turn([TurnEvent(kind="assistant_message", sequence=1, content="已修复")])

    assert render_action_entries(turn, DEFAULTS) == []
