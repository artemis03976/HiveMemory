"""一轮交互材料的解析与渲染单元测试。

被测对象：``render_turn_material`` / ``render_turn_text`` 把用户消息之后的事件
渲染为感知层唯一的材料。
保护的契约：
- 步骤按序号排列，助手正文按出现位置穿插在动作之间；
- 没有结局的调用只渲染调用，不断言它没有完成；
- 能力步骤只有一行摘要，不渲染结果，WRITE/UPDATE/CALL 与无法解析的指令不渲染；
  同一次读取无论来自 MTP 还是外部 harness 的 MCP 工具，渲染相同；
- 外部工具步骤渲染工具名、参数摘要与结果，出错的结果标明状态；思考、参数与
  结果按各自上限截断，保留首尾；
- 单轮上限只约束思考、能力摘要与外部工具步骤：超出时先降级较早的工具步骤，
  仍超出时按优先级省略整条，结果不超过上限；用户消息与助手正文不受约束；
- 空白的思考不渲染。
"""

from __future__ import annotations

from typing import Any

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import TurnEvent
from hivememory.engines.perception.material import (
    MaterialEntry,
    render_turn_material,
    render_turn_text,
)


def _turn(*events: dict[str, Any], user: str = "修一下测试") -> list[TurnEvent]:
    """以序号 0 的用户消息开头，按给出的顺序为其后的事件编号。"""
    return [
        TurnEvent(kind="user_message", sequence=0, content=user),
        *(TurnEvent(sequence=index, **fields) for index, fields in enumerate(events, start=1)),
    ]


def _text(content: str) -> dict[str, Any]:
    return {"kind": "assistant_message", "content": content}


def _thought(content: str) -> dict[str, Any]:
    return {"kind": "thought", "content": content}


def _call(
    action_id: str,
    tool_name: str | None = "Bash",
    *,
    result: str | None = None,
    status: str = "success",
    **fields: Any,
) -> dict[str, Any]:
    """一次工具调用；给出 ``result`` 时带结局，否则没有结局。"""
    event = {
        "kind": "tool_call",
        "content": fields.pop("content", ""),
        "action_id": action_id,
        "tool_name": tool_name,
        **fields,
    }
    if result is not None:
        event.update(status=status, result=result)
    return event


def _texts(entries: list[MaterialEntry]) -> list[str]:
    return [entry.text for entry in entries]


DEFAULTS = GenerationTranscriptConfig()


def test_assistant_text_is_interleaved_between_actions_in_order():
    """助手正文与动作按出现顺序穿插，不再把全部动作排在回复之前。"""
    events = _turn(
        _thought("先查记忆"),
        _call(
            "s",
            None,
            verb="SEARCH",
            tool_args={"query": "Import Bus 4xx 游标"},
            result="已有记忆正文",
        ),
        _text("找到一条相关记忆，再核对一下代码。"),
        _call("bash", tool_args={"command": "rg -n advance"}, result="238: self._advance(path)"),
        _text("连接器遇到 4xx 会推进游标。"),
        user="Import Bus 遇到 4xx 时游标怎么处理？",
    )

    assert render_turn_text(events, DEFAULTS) == "\n".join(
        [
            "[User]: Import Bus 遇到 4xx 时游标怎么处理？",
            "- THOUGHT: 先查记忆",
            '- SEARCH: "Import Bus 4xx 游标"',
            "[Assistant]: 找到一条相关记忆，再核对一下代码。",
            '- TOOL: Bash {"command": "rg -n advance"}',
            "  RESULT: 238: self._advance(path)",
            "[Assistant]: 连接器遇到 4xx 会推进游标。",
        ]
    )


def test_capability_steps_render_one_line_without_results():
    """能力步骤只有一行摘要；WRITE/UPDATE/CALL 与无法解析的指令不进入材料；没有结局的 RUN 标明。"""
    events = _turn(
        _call("search", None, verb="SEARCH", tool_args={"query": "部署"}, result="已有记忆正文"),
        _call("run", None, verb="RUN", targets=("git_log",), result="commit abc"),
        _call(
            "write",
            None,
            verb="WRITE",
            tool_args={"content": "记住"},
            result="已登记",
            status="ack",
        ),
        _call("update", None, verb="UPDATE", targets=("fact_a",), result="已登记", status="ack"),
        _call("call", None, verb="CALL", targets=("helper",), result="子 Agent 回复"),
        _call("unparsed", None, content="⟪ READ ⟫", result="parse error", status="error"),
        _call("pending", None, verb="RUN", targets=("sys_python",)),
    )

    entries = render_turn_material(events, DEFAULTS)

    assert [entry.kind for entry in entries] == ["capability"] * 3
    assert _texts(entries) == [
        '- SEARCH: "部署"',
        "- RUN: git_log (success)",
        "- RUN: sys_python (no result)",
    ]


def test_same_read_renders_the_same_from_mtp_and_mcp():
    """MTP READ 与外部 harness 经 MCP 的读取写出同形调用，材料相同，只列 alias、不含返回原文。"""
    mtp = _turn(
        _call(
            "action_2_3",
            None,
            verb="READ",
            targets=("fact_a", "fact_b"),
            content="⟪ READ ⟫",
            result='[System MTP Execution Result]\n<mtp_response status="success">',
        ),
    )
    mcp = _turn(
        _call("toolu_01", None, verb="READ", targets=("fact_a", "fact_b"), result="记忆全文"),
    )

    assert _texts(render_turn_material(mtp, DEFAULTS)) == ["- READ: fact_a, fact_b"]
    assert render_turn_text(mtp, DEFAULTS) == render_turn_text(mcp, DEFAULTS)


def test_error_results_are_marked():
    """出错的外部工具结果标明状态。"""
    events = _turn(
        _call(
            "toolu-1",
            tool_name="Read",
            tool_args={"file_path": "a.py"},
            result="permission denied",
            status="error",
        ),
    )

    assert _texts(render_turn_material(events, DEFAULTS)) == [
        '- TOOL: Read {"file_path": "a.py"}\n  RESULT (error): permission denied'
    ]


def test_thoughts_arguments_and_results_are_clipped_keeping_head_and_tail():
    """各部分按自身上限截断，保留首尾：长输出末尾的报错仍然可见。"""
    config = GenerationTranscriptConfig(
        max_thought_chars=10, max_args_chars=12, max_result_chars=20
    )
    events = _turn(
        _thought("思" * 30),
        _call(
            "toolu-1",
            tool_args={"command": "x" * 50},
            result="o" * 200 + "Traceback: boom",
            status="error",
        ),
    )

    thought, tool = _texts(render_turn_material(events, config))
    tool_line, result_line = tool.split("\n")
    assert thought == "- THOUGHT: 思思思思思 …[20 chars omitted]… 思思思思思"
    assert tool_line == '- TOOL: Bash {"comm …[53 chars omitted]… xxxx"}'
    assert result_line.startswith("  RESULT (error): oooooooooo …[")
    assert result_line.endswith(": boom")


def test_turn_limit_keeps_thoughts_failed_and_recent_tools_before_older_ones():
    """超出单轮上限时，思考、出错的工具步骤（即使最早）与最近的步骤保留完整形态，其余降为一行。"""
    config = GenerationTranscriptConfig(max_result_chars=100, max_turn_action_chars=340)
    events = _turn(
        _thought("计划"),
        _call("failed", tool_args={"c": "1"}, result="a" * 100, status="error"),
        _call("middle", tool_args={"c": "2"}, result="b" * 100),
        _call("recent", tool_args={"c": "3"}, result="c" * 100),
    )

    thought, failed, middle, recent = _texts(render_turn_material(events, config))

    assert thought == "- THOUGHT: 计划"
    assert failed == '- TOOL: Bash {"c": "1"}\n  RESULT (error): ' + "a" * 100
    assert middle == '- TOOL: Bash {"c": "2"}'
    assert recent == '- TOOL: Bash {"c": "3"}\n  RESULT: ' + "c" * 100


def test_turn_limit_degrades_tools_before_touching_thoughts():
    """完整步骤放不下时先降为一行；降级后放得下就不省略任何条目。"""
    config = GenerationTranscriptConfig(max_turn_action_chars=60)
    events = _turn(
        _thought("第一步"),
        _call("toolu-1", tool_args={"c": "1"}, result="done", status="error"),
        _thought("第二步"),
    )

    assert _texts(render_turn_material(events, config)) == [
        "- THOUGHT: 第一步",
        '- TOOL: Bash {"c": "1"}',
        "- THOUGHT: 第二步",
    ]


def test_turn_limit_is_hard_and_omits_lowest_priority_entries_first():
    """降为一行后仍超出时省略整条：先省略非出错步骤，再省略出错步骤，最后才是较早的思考。"""
    config = GenerationTranscriptConfig(max_turn_action_chars=200)
    events = _turn(
        _thought("a" * 60),
        _call("ok", tool_args={"c": "1"}, result="fine"),
        _call("failed", tool_args={"c": "2"}, result="boom", status="error"),
        _thought("b" * 60),
        _thought("c" * 60),
    )

    entries = render_turn_material(events, config)

    assert _texts(entries) == [
        "- OMITTED: 3 entries over the per-turn limit",
        "- THOUGHT: " + "b" * 60,
        "- THOUGHT: " + "c" * 60,
    ]
    assert sum(len(entry.text) for entry in entries) <= 200


def test_turn_limit_does_not_count_assistant_text():
    """助手正文不受单轮上限约束：长回复不会挤掉思考与工具步骤。"""
    config = GenerationTranscriptConfig(max_turn_action_chars=60)
    events = _turn(
        _text("长回复" * 100),
        _call("toolu-1", tool_args={"c": "1"}, result="done"),
    )

    assert _texts(render_turn_material(events, config)) == [
        "[Assistant]: " + "长回复" * 100,
        '- TOOL: Bash {"c": "1"}\n  RESULT: done',
    ]


def test_blank_thoughts_are_not_rendered():
    """只有空白的思考（例如只有签名的思考块）不形成条目。"""
    events = _turn(_thought("   "), _thought(""), _call("toolu-1", tool_args={"c": "1"}))

    assert _texts(render_turn_material(events, DEFAULTS)) == ['- TOOL: Bash {"c": "1"}']
