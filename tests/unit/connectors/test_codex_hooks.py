"""Codex hooks 处理的单元测试：登记、结束记录与唤醒同步，以及写入前补上的轮次信息。

保护的契约：子 Agent 中的事件不登记会话记录；``Stop`` 与 ``Interrupt`` 按输入中的
轮次写结局；``PreToolUse`` 的输出带 ``permissionDecision: allow`` 与覆盖模型填写
的轮次信息，子 Agent 中的写入归到派生它的那一轮，与之后导入的同名字段一致；
命令行入口写出的输出是 ASCII 转义的 JSON。
"""

from __future__ import annotations

import io
import json

import pytest
from connector.__main__ import main
from connector.codex.hooks import handle_hook, pre_tool_use
from connector.codex.reader import CodexReader, hook_tool_name
from connector.core.client import ImportClient
from connector.core.config import ConnectorConfig
from connector.core.state import StateStore
from connector.core.sync import TranscriptSync

from tests.unit.connectors.codex_rollouts import THREAD_ID, RolloutWriter

reader = CodexReader()


@pytest.fixture
def setup(tmp_path):
    config = ConnectorConfig(
        url="http://hivememory.test",
        principal_id="harness:codex",
        agent_id="codex",
        user_id=None,
        state_dir=tmp_path / "state",
    )
    store = StateStore(config.state_dir)
    store.first_run_at(0.0)
    sync = TranscriptSync(
        reader=reader,
        store=store,
        client=ImportClient(config, transport=lambda *args: (200, b"{}")),
        config=config,
    )
    spawned: list[list[str]] = []
    rollout = RolloutWriter(tmp_path)

    def hook(event: str, **payload) -> None:
        handle_hook(
            event,
            {"transcript_path": str(rollout.path), "session_id": THREAD_ID, **payload},
            sync=sync,
            store=store,
            spawn=spawned.append,
        )

    return hook, store, rollout, spawned


@pytest.mark.parametrize(("event", "outcome"), [("Stop", "completed"), ("Interrupt", "cancelled")])
def test_turn_end_events_record_the_named_turn_and_wake_its_sync(setup, event, outcome):
    """轮次结束与中断按输入中的轮次写结局，并只唤醒这份会话记录的同步。"""
    hook, store, rollout, spawned = setup
    rollout.turn("t1")
    rollout.start("t2")

    hook(event, turn_id="t2")

    assert store.load(str(rollout.path)).end_records == {"t2": outcome}
    assert spawned == [["sync", "--transcript", str(rollout.path)]]


def test_session_start_registers_and_wakes_all_syncs(setup):
    """会话开始登记会话记录、撤销已结束标记，并补发全部已登记会话记录。"""
    hook, store, rollout, spawned = setup
    rollout.turn("t1")
    hook("SessionEnd")

    hook("SessionStart", source="resume")

    assert store.registered_transcripts() == [str(rollout.path)]
    assert store.load(str(rollout.path)).session_ended is False
    assert spawned[-1] == ["sync", "--all"]


def test_events_inside_a_subagent_do_nothing(setup):
    """子 Agent 中的事件不登记子 Agent 的会话记录，也不唤醒同步：只导入顶层会话。"""
    hook, store, rollout, spawned = setup
    rollout.turn("t1")

    hook("Stop", turn_id="t1", agent_id="child-thread")

    assert (store.registered_transcripts(), spawned) == ([], [])


def _write_payload(transcript, turn_id, **extra):
    return {
        "session_id": THREAD_ID,
        "transcript_path": str(transcript),
        "turn_id": turn_id,
        "tool_name": hook_tool_name("memory_write"),
        "tool_input": {"content": "记住这个", "turn": {"source": "forged"}},
        **extra,
    }


def test_pre_tool_use_allows_the_call_with_the_current_turn_overriding_the_model(tmp_path):
    """当前一轮取输入中的轮次，之前的轮次作为最近上文；模型填写的轮次信息被覆盖。"""
    rollout = RolloutWriter(tmp_path)
    rollout.turn("t1", question="上一个问题", answer="上一个回答")
    rollout.start("t2")
    rollout.user("记住这个约定")
    store = StateStore(tmp_path / "state")

    output = pre_tool_use(_write_payload(rollout.path, "t2"), reader=reader, store=store)

    hook_output = output["hookSpecificOutput"]
    assert (hook_output["hookEventName"], hook_output["permissionDecision"]) == (
        "PreToolUse",
        "allow",
    )
    assert hook_output["updatedInput"]["content"] == "记住这个"
    assert hook_output["updatedInput"]["turn"] == {
        "source": "codex",
        "external_conversation_id": THREAD_ID,
        "external_turn_id": "t2",
        "user_message": "记住这个约定",
        "recent_turns": [
            {"user_message": "上一个问题", "assistant_text": "上一个回答", "external_turn_id": "t1"}
        ],
    }


def test_pre_tool_use_inside_a_subagent_uses_the_turn_that_spawned_it(tmp_path):
    """子 Agent 中的写入归到根会话中派生它的那一轮。"""
    root = RolloutWriter(tmp_path)
    root.start("root-turn")
    root.user("让子 Agent 去查")
    child = RolloutWriter(tmp_path, thread_id="child-thread", parent_thread_id=THREAD_ID)
    child.start("child-turn", root_turn_id="root-turn")
    child.user("查一下配置")
    store = StateStore(tmp_path / "state")
    store.register(str(root.path), origin="new", offset=0)

    output = pre_tool_use(
        _write_payload(child.path, "child-turn", agent_id="child-thread"),
        reader=reader,
        store=store,
    )

    turn = output["hookSpecificOutput"]["updatedInput"]["turn"]
    assert (turn["external_conversation_id"], turn["external_turn_id"], turn["user_message"]) == (
        THREAD_ID,
        "root-turn",
        "让子 Agent 去查",
    )


def test_pre_tool_use_without_the_named_turn_gives_no_output(tmp_path):
    """会话记录中找不到输入中的轮次时不输出，工具调用照常进行，由 HiveMemory 拒绝。"""
    rollout = RolloutWriter(tmp_path)
    rollout.turn("t1")

    assert (
        pre_tool_use(_write_payload(rollout.path, "t9"), reader=reader, store=StateStore(tmp_path))
        is None
    )


def test_pre_tool_use_entry_writes_ascii_json(tmp_path, monkeypatch, capsys):
    """命令行入口的输出是 ASCII 转义的 JSON：Windows 上标准输出按本地代码页编码，中文原样还原。"""
    rollout = RolloutWriter(tmp_path)
    rollout.start("t1")
    rollout.user("记住这个约定")
    monkeypatch.setenv("HIVEMEMORY_CONNECTOR_STATE_DIR", str(tmp_path / "state"))
    stdin = json.dumps(_write_payload(rollout.path, "t1"), ensure_ascii=False).encode("utf-8")
    monkeypatch.setattr(
        "sys.stdin",
        io.TextIOWrapper(io.BytesIO(stdin), encoding="cp1252", errors="surrogateescape"),
    )

    assert main(["codex", "hook", "PreToolUse"]) == 0

    out = capsys.readouterr().out
    assert out.isascii()
    assert (
        json.loads(out)["hookSpecificOutput"]["updatedInput"]["turn"]["user_message"]
        == "记住这个约定"
    )


def test_resumed_session_path_with_extended_prefix_is_the_same_transcript(setup):
    r"""续会话时 Codex 给出带 ``\\?\`` 前缀的路径；它与会话开始时的路径是同一份会话记录，不重复登记。

    路径是 Codex 在 Windows 上给出的形状；规整只处理字符串，测试在任何平台上都成立。
    """
    hook, store, _, _ = setup
    path = "C:\\Users\\me\\.codex\\sessions\\2026\\11\\08\\rollout-2026-11-08T10-00-00-t.jsonl"
    hook("SessionStart", source="startup", transcript_path=path)

    hook("SessionStart", source="resume", transcript_path="\\\\?\\" + path)

    assert store.registered_transcripts() == [path]
