"""Claude Code hooks 处理与命令行入口的单元测试。

保护的契约：``Stop``/``StopFailure`` 把最后一轮记为正常结束或失败后唤醒
这份会话记录的同步；``SessionEnd`` 标记会话结束；``SessionStart`` 撤销结束
标记并唤醒全部会话记录的补发；``PreToolUse`` 补上的轮次信息与这一轮之后的
导入请求一致，保留原参数、不给权限决定，读不到当前一轮时不输出，也不登记、
不唤醒同步；hook 入口无论成败都以 0 退出，只有 ``PreToolUse`` 的输出写到
标准输出。
"""

from __future__ import annotations

import io
import json

import pytest
from connector.__main__ import main
from connector.claude_code.hooks import handle_hook, pre_tool_use
from connector.claude_code.reader import ClaudeCodeReader, plugin_tool_name
from connector.core.client import ImportClient
from connector.core.config import ConnectorConfig
from connector.core.log import ConnectorLog
from connector.core.state import StateStore
from connector.core.sync import TranscriptSync

from tests.unit.connectors.transcripts import TranscriptWriter, text, tool_use


@pytest.fixture
def setup(tmp_path):
    config = ConnectorConfig(
        url="http://hivememory.test",
        agent_id="claude_code",
        user_id=None,
        state_dir=tmp_path / "state",
    )
    store = StateStore(config.state_dir)
    store.first_run_at(0.0)
    reader = ClaudeCodeReader()
    sync = TranscriptSync(
        reader=reader,
        store=store,
        client=ImportClient(config, transport=lambda *args: (200, b"{}")),
        config=config,
    )
    spawned: list[list[str]] = []
    log = TranscriptWriter(tmp_path / "session.jsonl")

    def hook(event: str, **payload) -> None:
        handle_hook(
            event,
            {"transcript_path": str(log.path), **payload},
            sync=sync,
            reader=reader,
            store=store,
            spawn=spawned.append,
        )

    return hook, store, log, spawned


@pytest.mark.parametrize(("event", "outcome"), [("Stop", "completed"), ("StopFailure", "failed")])
def test_stop_events_record_the_last_turn_and_wake_its_sync(setup, event, outcome):
    """轮次结束的通知把最后一轮记为对应结局，并只唤醒这份会话记录的同步。"""
    hook, store, log, spawned = setup
    log.turn("p1")
    log.turn("p2")

    hook(event)

    assert store.load(str(log.path)).end_records == {"p2": outcome}
    assert spawned == [["sync", "--transcript", str(log.path)]]


def test_stop_without_an_unsubmitted_turn_is_ignored_and_logged(setup):
    """末尾一轮已按超时或会话结束提交、游标已越过它时，结束通知被忽略并写入日志。"""
    _hook, store, log, spawned = setup
    log.turn("p1")
    path = str(log.path)
    store.register(path, origin="new", offset=0)
    with store.update(path) as state:
        state.offset = log.path.stat().st_size

    config = ConnectorConfig(
        url="http://hivememory.test",
        agent_id="claude_code",
        user_id=None,
        state_dir=store.state_dir,
    )
    reader = ClaudeCodeReader()
    sync = TranscriptSync(
        reader=reader,
        store=store,
        client=ImportClient(config, transport=lambda *args: (200, b"{}")),
        config=config,
    )

    handle_hook(
        "Stop",
        {"transcript_path": path},
        sync=sync,
        reader=reader,
        store=store,
        spawn=spawned.append,
        log=ConnectorLog(store.state_dir),
    )

    assert store.load(path).end_records == {}
    assert "ignored Stop: no unsubmitted turn after cursor" in (
        store.state_dir / "connector.log"
    ).read_text(encoding="utf-8")


def test_session_end_and_start_toggle_the_session_state(setup):
    """会话结束后末尾一轮随之结束；恢复会话时撤销，并补发全部会话记录。"""
    hook, store, log, spawned = setup
    log.turn("p1")

    hook("SessionEnd", reason="other")
    ended = store.load(str(log.path)).session_ended
    hook("SessionStart", source="resume")

    assert (ended, store.load(str(log.path)).session_ended) == (True, False)
    assert spawned == [["sync", "--transcript", str(log.path)], ["sync", "--all"]]


def test_events_without_transcript_or_unknown_events_do_nothing(setup, tmp_path):
    """没有会话记录路径或不认识的事件不登记、不唤醒同步。"""
    hook, store, log, spawned = setup

    handle_hook(
        "Stop",
        {},
        sync=None,  # type: ignore[arg-type]
        reader=ClaudeCodeReader(),
        store=store,
        spawn=spawned.append,
    )
    hook("PreToolUse")

    assert spawned == []
    assert store.registered_transcripts() == []


@pytest.mark.parametrize("event", ["Stop", "PreToolUse"])
def test_hook_entry_exits_zero_and_stays_silent_on_bad_input(tmp_path, monkeypatch, capsys, event):
    """连接器的失败只进日志：无法解析的 hook 输入不打断 Claude Code，也不输出半截结果。"""
    monkeypatch.setenv("HIVEMEMORY_CONNECTOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))

    assert main(["hook", event]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert "connector failed" in (tmp_path / "state" / "connector.log").read_text(encoding="utf-8")


def test_hook_entry_registers_transcript_from_stdin(tmp_path, monkeypatch):
    """hook 入口从标准输入取得会话记录路径并登记，后台同步交给独立进程。"""
    log = TranscriptWriter(tmp_path / "session.jsonl")
    log.turn("p1")
    monkeypatch.setenv("HIVEMEMORY_CONNECTOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"transcript_path": str(log.path)})))
    spawned = []
    monkeypatch.setattr(
        "connector.__main__._spawn_detached", lambda command, config: spawned.append(command)
    )

    assert main(["hook", "Stop"]) == 0
    assert StateStore(tmp_path / "state").registered_transcripts() == [str(log.path)]
    assert spawned == [["sync", "--transcript", str(log.path)]]


_TURN_INFO_FIELDS = (
    "source",
    "external_conversation_id",
    "external_turn_id",
    "user_message",
    "recent_turns",
)


def test_pre_tool_use_adds_the_turn_info_that_the_later_import_carries(tmp_path):
    """补上的轮次信息与这一轮结束后导入请求的标识、用户消息与最近轮次一致。

    已导入的轮次只留最近几轮的上文，hook 从同一位置读起；原参数保留，模型自己
    填写的轮次信息被覆盖，不给权限决定。
    """
    config = ConnectorConfig(
        url="http://hivememory.test",
        agent_id="claude_code",
        user_id=None,
        state_dir=tmp_path / "state",
    )
    store = StateStore(config.state_dir)
    store.first_run_at(0.0)
    bodies: list[dict] = []

    def transport(url, body, headers, timeout):
        bodies.append(json.loads(body))
        return 200, b'{"status": "accepted"}'

    reader = ClaudeCodeReader()
    sync = TranscriptSync(
        reader=reader, store=store, client=ImportClient(config, transport=transport), config=config
    )
    log = TranscriptWriter(tmp_path / "session.jsonl")
    path = str(log.path)
    sync.register(path)
    for index in range(4):
        log.turn(f"p{index}", question=f"问题 {index}", answer=f"回答 {index}")
    log.prompt("p4", "记住：部署改为手动执行")
    # 下一轮开始后，此前四轮都已结束并导入。
    sync.sync(path)
    tool_input = {"content": "部署改为手动执行", "turn": {"external_turn_id": "编造的"}}
    log.assistant(
        tool_use("toolu-1", plugin_tool_name("memory_write"), tool_input), message_id="m4"
    )

    output = pre_tool_use(
        {
            "transcript_path": path,
            "tool_name": plugin_tool_name("memory_write"),
            "tool_input": tool_input,
        },
        reader=reader,
        store=store,
    )
    log.tool_result("toolu-1", "已接纳")
    log.assistant(text("记下了"), message_id="m4-reply")
    store.record_end(path, "p4", "completed")
    sync.sync(path)

    hook_output = output["hookSpecificOutput"]
    assert set(hook_output) == {"hookEventName", "updatedInput"}
    updated = hook_output["updatedInput"]
    assert updated["content"] == "部署改为手动执行"
    imported = bodies[-1]
    assert (imported["external_turn_id"], len(bodies)) == ("p4", 5)
    assert [turn["external_turn_id"] for turn in imported["recent_turns"]] == ["p1", "p2", "p3"]
    assert updated["turn"] == {field: imported[field] for field in _TURN_INFO_FIELDS}


@pytest.mark.parametrize(
    ("write_transcript", "payload"),
    [
        (lambda log: None, {"tool_input": {"content": "x"}}),
        (lambda log: log.turn("p1"), {}),
    ],
    ids=["no-human-prompt", "no-tool-input"],
)
def test_pre_tool_use_without_current_turn_or_tool_input_gives_no_output(
    tmp_path, write_transcript, payload
):
    """读不到当前一轮或拿不到工具参数时不输出，工具调用照常进行、由服务端拒绝。"""
    log = TranscriptWriter(tmp_path / "session.jsonl")
    write_transcript(log)

    output = pre_tool_use(
        {"transcript_path": str(log.path), **payload},
        reader=ClaudeCodeReader(),
        store=StateStore(tmp_path / "state"),
    )

    assert output is None


def test_pre_tool_use_entry_writes_only_the_output_without_registering(
    tmp_path, monkeypatch, capsys
):
    """PreToolUse 入口把输出作为唯一的 JSON 写到标准输出，不登记会话记录、不唤醒同步。"""
    log = TranscriptWriter(tmp_path / "session.jsonl")
    log.prompt("p1", "记住这个")
    monkeypatch.setenv("HIVEMEMORY_CONNECTOR_STATE_DIR", str(tmp_path / "state"))
    payload = {
        "transcript_path": str(log.path),
        "tool_name": plugin_tool_name("memory_update"),
        "tool_input": {"alias": "deploy_flow", "instruction": "改为手动"},
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    spawned = []
    monkeypatch.setattr(
        "connector.__main__._spawn_detached", lambda command, config: spawned.append(command)
    )

    assert main(["hook", "PreToolUse"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["updatedInput"]["turn"]["external_turn_id"] == "p1"
    assert StateStore(tmp_path / "state").registered_transcripts() == []
    assert spawned == []
