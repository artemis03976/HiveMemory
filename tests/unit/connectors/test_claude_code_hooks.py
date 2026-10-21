"""Claude Code hooks 处理与命令行入口的单元测试。

保护的契约：``Stop``/``StopFailure`` 把最后一轮记为正常结束或失败后唤醒
这份会话记录的同步；``SessionEnd`` 标记会话结束；``SessionStart`` 撤销结束
标记并唤醒全部会话记录的补发；hook 入口无论成败都以 0 退出、不向标准输出
写内容。
"""

from __future__ import annotations

import io
import json

import pytest
from connector.__main__ import main
from connector.claude_code.hooks import handle_hook
from connector.claude_code.reader import ClaudeCodeReader
from connector.core.client import ImportClient
from connector.core.config import ConnectorConfig
from connector.core.state import StateStore
from connector.core.sync import TranscriptSync

from tests.unit.connectors.transcripts import TranscriptWriter


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


def test_hook_entry_exits_zero_and_stays_silent_on_bad_input(tmp_path, monkeypatch, capsys):
    """连接器的失败只进日志：无法解析的 hook 输入不打断 Claude Code。"""
    monkeypatch.setenv("HIVEMEMORY_CONNECTOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))

    assert main(["hook", "Stop"]) == 0
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
