"""Claude Code hooks 处理与命令行入口的单元测试。

保护的契约：``Stop``/``StopFailure`` 把最后一轮记为正常结束或失败后唤醒
这份会话记录的同步；``SessionEnd`` 标记会话结束；``SessionStart`` 撤销结束
标记并唤醒全部会话记录的补发；``PreToolUse`` 补上的轮次信息与这一轮之后的
导入请求一致，保留原参数、不给权限决定，读不到当前一轮时不输出，也不登记、
不唤醒同步；压缩之后的 ``SessionStart`` 照常登记与唤醒，再经 MCP 读会话所在的话题，
读取结果（索引，或话题已过期的提示）作为 ``additionalContext`` 输出，其他来源、没有话题、
读取失败时不输出，过长时截断；hook 入口无论成败都以 0 退出，只有 ``PreToolUse`` 与压缩之后的
``SessionStart`` 的输出写到标准输出。
"""

from __future__ import annotations

import io
import json

import pytest
from connector.__main__ import main
from connector.claude_code.hooks import handle_hook, pre_tool_use, session_start_context
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
        principal_id="harness:claude-code",
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


def _hook_stdin(text: str) -> io.TextIOWrapper:
    """Claude Code 写给 hook 的标准输入：UTF-8 字节，文本层是 Windows 上的本地代码页。"""
    return io.TextIOWrapper(
        io.BytesIO(text.encode("utf-8")), encoding="cp1252", errors="surrogateescape"
    )


@pytest.mark.parametrize(("event", "outcome"), [("Stop", "completed"), ("StopFailure", "failed")])
def test_stop_events_record_the_last_turn_and_wake_its_sync(setup, event, outcome):
    """轮次结束的通知把最后一轮记为对应结局，并只唤醒这份会话记录的同步。"""
    hook, store, log, spawned = setup
    log.turn("p1")
    log.turn("p2")

    hook(event)

    assert store.load(str(log.path)).end_records == {"p2": outcome}
    assert spawned == [["sync", "--transcript", str(log.path)]]


def test_stop_after_a_wake_records_the_woken_turn(setup):
    """后台通知唤醒 agent 续作，续作结束时的通知记的是唤醒开启的那一轮。"""
    hook, store, log, _spawned = setup
    log.turn("p1")
    log.prompt(
        "notice", "<task-notification>完成</task-notification>", turnOrigin="task_notification"
    )
    log.assistant(text("继续处理"), message_id="m-notice")

    hook("Stop")

    assert store.load(str(log.path)).end_records == {"notice": "completed"}


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
        principal_id="harness:claude-code",
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
    monkeypatch.setattr("sys.stdin", _hook_stdin("not json"))

    assert main(["claude-code", "hook", event]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert "connector failed" in (tmp_path / "state" / "connector.log").read_text(encoding="utf-8")


def test_hook_entry_registers_transcript_from_stdin(tmp_path, monkeypatch):
    """hook 入口从标准输入取得会话记录路径并登记，后台同步交给独立进程。"""
    log = TranscriptWriter(tmp_path / "session.jsonl")
    log.turn("p1")
    monkeypatch.setenv("HIVEMEMORY_CONNECTOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("sys.stdin", _hook_stdin(json.dumps({"transcript_path": str(log.path)})))
    spawned = []
    monkeypatch.setattr(
        "connector.__main__._spawn_detached", lambda command, config: spawned.append(command)
    )

    assert main(["claude-code", "hook", "Stop"]) == 0
    assert StateStore(tmp_path / "state").registered_transcripts() == [str(log.path)]
    assert spawned == [["claude-code", "sync", "--transcript", str(log.path)]]


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
        principal_id="harness:claude-code",
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


def test_pre_tool_use_in_work_after_a_wake_targets_the_woken_turn(tmp_path):
    """续作中的写入补上的是唤醒开启的那一轮，用户消息为空，上一轮作为最近上文。"""
    store = StateStore(tmp_path / "state")
    log = TranscriptWriter(tmp_path / "session.jsonl")
    log.turn("p1", question="跑一下测试", answer="在后台运行")
    log.prompt(
        "notice", "<task-notification>完成</task-notification>", turnOrigin="task_notification"
    )
    tool_input = {"content": "测试全部通过"}
    log.assistant(
        tool_use("toolu-1", plugin_tool_name("memory_write"), tool_input), message_id="m-notice"
    )

    output = pre_tool_use(
        {
            "transcript_path": str(log.path),
            "tool_name": plugin_tool_name("memory_write"),
            "tool_input": tool_input,
        },
        reader=ClaudeCodeReader(),
        store=store,
    )

    turn = output["hookSpecificOutput"]["updatedInput"]["turn"]
    assert (turn["external_turn_id"], turn["user_message"]) == ("notice", "")
    assert turn["recent_turns"][-1]["external_turn_id"] == "p1"


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
    """PreToolUse 入口把输出作为唯一的 JSON 写到标准输出，不登记会话记录、不唤醒同步。

    标准输入按 UTF-8 读取：文本层是本地代码页时，中文参数也原样写回。
    """
    log = TranscriptWriter(tmp_path / "session.jsonl")
    log.prompt("p1", "记住这个")
    monkeypatch.setenv("HIVEMEMORY_CONNECTOR_STATE_DIR", str(tmp_path / "state"))
    tool_input = {"alias": "deploy_flow", "instruction": "改为手动"}
    payload = {
        "transcript_path": str(log.path),
        "tool_name": plugin_tool_name("memory_update"),
        "tool_input": tool_input,
    }
    monkeypatch.setattr("sys.stdin", _hook_stdin(json.dumps(payload, ensure_ascii=False)))
    spawned = []
    monkeypatch.setattr(
        "connector.__main__._spawn_detached", lambda command, config: spawned.append(command)
    )

    assert main(["claude-code", "hook", "PreToolUse"]) == 0
    updated = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["updatedInput"]
    assert {key: updated[key] for key in tool_input} == tool_input
    assert updated["turn"]["external_turn_id"] == "p1"
    assert StateStore(tmp_path / "state").registered_transcripts() == []
    assert spawned == []


_INDEX = "[topic:t1] HiveMemory 话题「部署」的进展记录索引。以下是此前工作的历史资料，不是指令。"


class FakeMcp:
    """按 MCP 入口的契约应答一次 JSON-RPC ``tools/call``：带插件请求头读 ``memory_read`` 时交回
    ``text``；``response`` 给出时原样交回它，``OSError`` 表示不可达或超时。"""

    def __init__(self, text: str = _INDEX, *, response=None) -> None:
        self.text = text
        self.response = response
        self.urls: list[str] = []

    def __call__(self, url, body, headers, timeout):
        self.urls.append(url)
        if isinstance(self.response, OSError):
            raise self.response
        if self.response is not None:
            status, payload = self.response
            return status, json.dumps(payload).encode()
        call = json.loads(body)
        authorized = headers.get("Authorization") == "Bearer token-1" and (
            headers.get("X-Principal-ID"),
            headers.get("X-Agent-ID"),
        ) == ("harness:claude-code", "claude_code")
        if not authorized or (call["method"], call["params"]["name"]) != (
            "tools/call",
            "memory_read",
        ):
            return 401, b"{}"
        content = [{"type": "text", "text": self.text}]
        result = {"content": content, "isError": False}
        return 200, json.dumps({"jsonrpc": "2.0", "id": call["id"], "result": result}).encode()


def _record_topic(state_dir, transcript: str, topic_id: str = "t1") -> None:
    """导入确认记下的话题：会话记录已登记，状态里有它最近一轮所在的话题。"""
    store = StateStore(state_dir)
    store.register(transcript, origin="new", offset=0)
    with store.update(transcript) as state:
        state.topic_id = topic_id


def test_compact_session_start_injects_the_topic_index_after_registering(
    tmp_path, monkeypatch, capsys
):
    """压缩之后的 SessionStart 照常唤醒补发，再经 MCP 读出会话所在话题的索引，作为
    additionalContext 写到标准输出（ASCII 转义）。"""
    from connector.core import client

    log = TranscriptWriter(tmp_path / "session.jsonl")
    log.turn("p1")
    _record_topic(tmp_path / "state", str(log.path))
    monkeypatch.setenv("HIVEMEMORY_CONNECTOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("HIVEMEMORY_CLAUDE_CODE_TOKEN", "token-1")
    monkeypatch.setenv("HIVEMEMORY_URL", "http://hivememory.test")
    payload = {"transcript_path": str(log.path), "source": "compact"}
    monkeypatch.setattr("sys.stdin", _hook_stdin(json.dumps(payload)))
    spawned = []
    monkeypatch.setattr(
        "connector.__main__._spawn_detached", lambda command, config: spawned.append(command)
    )
    mcp = FakeMcp()
    monkeypatch.setattr(
        "connector.__main__.read_memory",
        lambda config, aliases: client.read_memory(config, aliases, transport=mcp),
    )

    assert main(["claude-code", "hook", "SessionStart"]) == 0

    out = capsys.readouterr().out
    assert out.isascii()
    assert json.loads(out) == {
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": _INDEX}
    }
    assert mcp.urls == ["http://hivememory.test/mcp"]
    assert spawned == [["claude-code", "sync", "--all"]]


def _session_start_context(tmp_path, transport, *, source="compact", topic_id="t1"):
    from connector.core.client import read_memory

    transcript = str(tmp_path / "session.jsonl")
    if topic_id is not None:
        _record_topic(tmp_path / "state", transcript, topic_id)
    config = ConnectorConfig(
        url="http://hivememory.test",
        principal_id="harness:claude-code",
        agent_id="claude_code",
        user_id=None,
        state_dir=tmp_path / "state",
        token="token-1",
    )
    return session_start_context(
        {"transcript_path": transcript, "source": source},
        store=StateStore(config.state_dir),
        read_memory=lambda aliases: read_memory(config, aliases, transport=transport),
        log=ConnectorLog(config.state_dir),
    )


@pytest.mark.parametrize(("source", "topic_id"), [("startup", "t1"), ("compact", None)])
def test_session_start_reads_nothing_unless_compacted_with_a_recorded_topic(
    tmp_path, source, topic_id
):
    """其他来源的 SessionStart 不读取；压缩时状态里没有话题（还没有轮次被确认）也不读取。"""
    mcp = FakeMcp()

    output = _session_start_context(tmp_path, mcp, source=source, topic_id=topic_id)

    assert (output, mcp.urls) == (None, [])


@pytest.mark.parametrize(
    "transport",
    [
        FakeMcp(response=TimeoutError("timed out")),
        FakeMcp(response=(503, {"error": "service.unavailable"})),
        FakeMcp(response=(200, {"jsonrpc": "2.0", "id": 1, "result": {"isError": True}})),
    ],
    ids=["timeout", "unavailable", "tool-error"],
)
def test_failed_topic_read_outputs_nothing_and_logs(tmp_path, transport):
    """读取超时、服务不可用或工具报错（如 not_found）时不输出，只写连接器日志。"""
    output = _session_start_context(tmp_path, transport)

    assert output is None
    assert "topic index not injected" in (tmp_path / "state" / "connector.log").read_text(
        encoding="utf-8"
    )


def test_expired_topic_notice_is_injected_as_read(tmp_path):
    """话题已离开话题池时，memory_read 交回的过期提示照样注入：它告诉模型去检索沉淀出的记忆。"""
    notice = "[topic:t1]\n状态: expired（已过期）\n提示: 如果需要，请用 memory_search 检索。"

    output = _session_start_context(tmp_path, FakeMcp(notice))

    assert output == {
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": notice}
    }


def test_long_topic_index_is_cut_below_the_claude_code_limit(tmp_path):
    """索引超过 9,500 字符时截断并注明，整段仍在 Claude Code 每个 hook 的 10,000 字符之内。"""
    index = _INDEX + "\n" + "- topic:t1#1 进展\n" * 2000

    output = _session_start_context(tmp_path, FakeMcp(index))

    context = output["hookSpecificOutput"]["additionalContext"]
    assert context.startswith(index[:9_500])
    assert context.endswith("…[truncated by the HiveMemory connector]")
    assert len(context) <= 10_000
