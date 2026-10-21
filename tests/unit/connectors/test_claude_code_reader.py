"""Claude Code 会话记录读取器的单元测试（合成会话记录，不读取真实对话）。

保护的契约：一轮从 ``turnOrigin`` 为 human 的消息开始；助手文本、思考、工具
调用与工具结果映射为导入事件；子代理、元条目、压缩摘要、注入的内容、系统
条目与不认识的条目跳过；中断标记结束一轮；重复写入的块按消息与块位置去重；
只读完整行；游标之前的条目不形成轮次。
"""

from __future__ import annotations

import json

from connector.claude_code.reader import ClaudeCodeReader

from tests.unit.connectors.transcripts import (
    TranscriptWriter,
    text,
    thinking,
    tool_use,
)

reader = ClaudeCodeReader()


def test_turn_maps_thinking_tool_calls_results_and_final_text(tmp_path):
    """思考、工具调用与结果按出现顺序成为事件；最终正文取最后一次工具结果之后的助手文本。"""
    log = TranscriptWriter(tmp_path / "s.jsonl")
    log.prompt("p1", "测试挂了")
    log.assistant(thinking("先跑测试"), text("我先运行测试。"), message_id="m1")
    log.assistant(tool_use("toolu-1", "Bash", {"command": "pytest -q"}), message_id="m1")
    log.tool_result("toolu-1", "1 failed", is_error=True)
    log.assistant(tool_use("toolu-2", "Edit", {"file_path": "a.py"}), message_id="m2")
    log.tool_result("toolu-2", [{"type": "text", "text": "ok"}])
    log.assistant(text("已修复"), text("测试通过"), message_id="m3", model="claude-opus")

    [turn] = reader.read(str(log.path), 0)

    assert (turn.turn_id, turn.conversation_id, turn.user_message) == (
        "p1",
        "session-test",
        "测试挂了",
    )
    assert turn.events == [
        {"kind": "thought", "sequence": 1, "content": "先跑测试"},
        {"kind": "assistant_message", "sequence": 2, "content": "我先运行测试。"},
        {
            "kind": "tool_call",
            "sequence": 3,
            "content": "",
            "action_id": "toolu-1",
            "tool_name": "Bash",
            "tool_args": {"command": "pytest -q"},
        },
        {
            "kind": "tool_result",
            "sequence": 4,
            "content": "1 failed",
            "action_id": "toolu-1",
            "status": "error",
        },
        {
            "kind": "tool_call",
            "sequence": 5,
            "content": "",
            "action_id": "toolu-2",
            "tool_name": "Edit",
            "tool_args": {"file_path": "a.py"},
        },
        {"kind": "tool_result", "sequence": 6, "content": "ok", "action_id": "toolu-2"},
        {"kind": "assistant_message", "sequence": 7, "content": "已修复"},
        {"kind": "assistant_message", "sequence": 8, "content": "测试通过"},
    ]
    assert turn.assistant_final_text == "已修复\n测试通过"
    assert turn.model_used == "claude-opus"


def test_next_prompt_closes_the_previous_turn_at_its_line(tmp_path):
    """下一条由人发出的消息关闭上一轮，两轮的位置首尾相接。"""
    log = TranscriptWriter(tmp_path / "s.jsonl")
    log.turn("p1")
    log.turn("p2")

    first, second = reader.read(str(log.path), 0)

    assert (first.closed_by_next_turn, second.closed_by_next_turn) == (True, False)
    assert first.start_offset == log.prompt_offsets["p1"]
    assert first.end_offset == second.start_offset == log.prompt_offsets["p2"]
    assert second.end_offset == log.path.stat().st_size


def test_non_conversation_entries_are_skipped(tmp_path):
    """子代理、元条目、压缩摘要、系统条目、不认识的条目与无正文的思考都不进入轮次。"""
    log = TranscriptWriter(tmp_path / "s.jsonl")
    log.prompt("p1", "问题")
    log.prompt("side-prompt", "子代理任务", isSidechain=True)
    log.assistant(text("子代理的回复"), message_id="side", isSidechain=True)
    log.user_text("<command-name>/model</command-name>", isMeta=True)
    log.user_text("压缩摘要", isCompactSummary=True)
    log.raw(type="system", subtype="compact_boundary", uuid="sys", sessionId="session-test")
    log.raw(type="file-history-snapshot", messageId="x")
    log.raw(type="assistant", uuid="broken", message="not an object")
    log.assistant(thinking("   "), {"type": "redacted_thinking", "data": "x"}, text("回答"))

    [turn] = reader.read(str(log.path), 0)

    assert turn.events == [{"kind": "assistant_message", "sequence": 1, "content": "回答"}]


def test_interrupt_marker_ends_turn_without_opening_another(tmp_path):
    """Claude Code 写入的中断标记结束当前一轮，自身不是由人发出的消息。"""
    log = TranscriptWriter(tmp_path / "s.jsonl")
    log.prompt("p1", "写个脚本")
    log.assistant(text("好的，我先"), message_id="m1")
    log.user_text("[Request interrupted by user]")

    [turn] = reader.read(str(log.path), 0)

    assert turn.interrupted is True
    assert turn.closed_by_next_turn is False


def test_only_human_origin_prompts_open_turns(tmp_path):
    """只有 ``turnOrigin`` 为 human 的消息开启新轮：后台任务通知与注入的内容都不开启也不关闭。"""
    log = TranscriptWriter(tmp_path / "s.jsonl")
    log.prompt("p1", "问题")
    log.prompt("notice", "后台任务完成通知", turnOrigin="task_notification")
    log.user_text("技能说明正文", sourceToolUseID="toolu-skill", turnCompanion=True)

    [turn] = reader.read(str(log.path), 0)

    assert (turn.turn_id, turn.closed_by_next_turn) == ("p1", False)


def test_prompt_without_session_id_falls_back_to_transcript_name(tmp_path):
    """条目缺少 ``sessionId`` 时，外部会话标识取会话记录的文件名。"""
    log = TranscriptWriter(tmp_path / "3f2a-session.jsonl")
    log.raw(
        type="user",
        uuid="p1",
        turnOrigin="human",
        message={"role": "user", "content": "问题"},
    )

    [turn] = reader.read(str(log.path), 0)

    assert turn.conversation_id == "3f2a-session"


def test_rewritten_blocks_are_deduplicated_by_message_and_block_position(tmp_path):
    """同一消息同一块位置重复写入只形成一次事件；正文相同但位置不同的块各自保留。"""
    log = TranscriptWriter(tmp_path / "s.jsonl")
    log.prompt("p1", "问题")
    log.assistant(text("好的"), text("好的"), tool_use("toolu-1", "Read", {"file_path": "a"}))
    duplicated = json.loads(log.path.read_text(encoding="utf-8").splitlines()[1])
    log.append(duplicated)
    log.tool_result("toolu-1", "内容")
    log.tool_result("toolu-1", "内容")

    [turn] = reader.read(str(log.path), 0)

    assert [event["kind"] for event in turn.events] == [
        "assistant_message",
        "assistant_message",
        "tool_call",
        "tool_result",
    ]


def test_incomplete_last_line_is_not_read(tmp_path):
    """仍在写入的末行不读取，轮次的末尾停在最后一个完整行。"""
    log = TranscriptWriter(tmp_path / "s.jsonl")
    log.turn("p1")
    complete = log.path.stat().st_size
    with log.path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "assistant"})[:10])

    [turn] = reader.read(str(log.path), 0)

    assert turn.end_offset == complete


def test_reading_from_cursor_ignores_entries_of_earlier_turn(tmp_path):
    """从游标读取时，第一条由人发出的消息之前的条目属于更早的轮次，不形成轮次。"""
    log = TranscriptWriter(tmp_path / "s.jsonl")
    log.prompt("p1", "问题")
    cursor = log.path.stat().st_size
    log.assistant(text("迟到的条目"), message_id="m1")
    log.turn("p2")

    turns = reader.read(str(log.path), cursor)

    assert [turn.turn_id for turn in turns] == ["p2"]


def test_last_turn_id_and_start_time(tmp_path):
    """hook 以最后一条由人发出的消息定位刚结束的一轮；开始时间取第一个条目。"""
    log = TranscriptWriter(tmp_path / "s.jsonl")
    log.turn("p1")
    log.turn("p2")
    log.tool_result("toolu-x", "不是提问")

    assert reader.last_turn_id(str(log.path), 0) == "p2"
    assert reader.started_at(str(log.path)) == 1792569600.0
    assert reader.started_at(str(tmp_path / "missing.jsonl")) is None
