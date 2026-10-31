"""一轮交互材料的解析与渲染单元测试。

被测对象：``render_turn_material`` / ``render_turn_text`` 把用户消息之后的事件
渲染为感知层唯一的材料。
保护的契约：
- 步骤按序号排列，助手正文按出现位置穿插在动作之间；
- 没有结局的调用只渲染调用，不断言它没有完成；
- 能力步骤只有一行摘要，不渲染结果，WRITE/UPDATE/CALL 与无法解析的指令不渲染；
  同一次读取无论来自 MTP 还是外部 harness 的 MCP 工具，渲染相同；
- 外部工具步骤渲染工具名、完整参数与结果，出错的结果标明状态；
- 只有工具结果按上限截断，保留首尾；思考与参数原样渲染；
- 一轮的条目不受总量上限约束：条目再多也全部原样渲染，不降级、不省略；
- 空白的思考不渲染。
"""

from __future__ import annotations

from typing import Any

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import TurnEvent
from hivememory.engines.perception.material import render_turn_material, render_turn_text


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

    assert render_turn_material(events, DEFAULTS) == [
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

    assert render_turn_material(mtp, DEFAULTS) == ["- READ: fact_a, fact_b"]
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

    assert render_turn_material(events, DEFAULTS) == [
        '- TOOL: Read {"file_path": "a.py"}\n  RESULT (error): permission denied'
    ]


def test_only_tool_results_are_clipped_keeping_head_and_tail():
    """只有工具结果按上限截断并保留首尾，长输出末尾的报错仍然可见；思考与参数再长也原样渲染。"""
    config = GenerationTranscriptConfig(max_result_chars=20)
    events = _turn(
        _thought("思" * 1500),
        _call(
            "toolu-1",
            tool_args={"command": "x" * 500},
            result="o" * 200 + "Traceback: boom",
            status="error",
        ),
    )

    thought, tool = render_turn_material(events, config)

    assert thought == "- THOUGHT: " + "思" * 1500
    assert tool == "\n".join(
        [
            '- TOOL: Bash {"command": "' + "x" * 500 + '"}',
            "  RESULT (error): oooooooooo …[195 chars omitted]… back: boom",
        ]
    )


def test_turn_renders_every_entry_in_full_without_a_turn_limit():
    """一轮的条目没有总量上限：工具步骤与思考再多，也全部以完整形态按顺序渲染。"""
    steps = [
        event
        for index in range(30)
        for event in (
            _thought(f"第 {index} 步" + "想" * 300),
            _call(f"toolu-{index}", tool_args={"c": str(index)}, result="r" * 700),
        )
    ]

    entries = render_turn_material(_turn(*steps), DEFAULTS)

    assert entries == [
        entry
        for index in range(30)
        for entry in (
            f"- THOUGHT: 第 {index} 步" + "想" * 300,
            f'- TOOL: Bash {{"c": "{index}"}}\n  RESULT: ' + "r" * 700,
        )
    ]


def test_blank_thoughts_are_not_rendered():
    """只有空白的思考（例如只有签名的思考块）不形成条目。"""
    events = _turn(_thought("   "), _thought(""), _call("toolu-1", tool_args={"c": "1"}))

    assert render_turn_material(events, DEFAULTS) == ['- TOOL: Bash {"c": "1"}']
