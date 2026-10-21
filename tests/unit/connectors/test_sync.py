"""连接器公共骨架的单元测试：游标、提交分类、结束记录、首次登记与游标失效。

使用合成的 Claude Code 会话记录与真实读取器；HTTP 以记录请求的替身传输代替。
保护的契约：确认后游标才前进；可重试错误保留游标并停止处理该会话记录；
4xx 与连续三次服务器内部错误写入拒绝记录并前进；结局先于提交写入、重发
沿用原结局；首次见到的旧会话不回填；游标失效时从最后一个已确认的轮次之后
继续，找不到该轮时停止处理并写入拒绝记录；并发读改写不损坏状态。
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from connector.claude_code.reader import ClaudeCodeReader
from connector.core.client import ImportClient
from connector.core.config import ConnectorConfig
from connector.core.state import StateStore
from connector.core.sync import (
    FLUSH_GRACE_SECONDS,
    RECENT_TEXT_LIMIT,
    STALE_SECONDS,
    TranscriptSync,
)
from connector.core.transcript import complete_size

from tests.unit.connectors.transcripts import TranscriptWriter, text, tool_use

STARTED_AT = 1792569600.0  # 合成会话记录第一个条目的时间


class FakeTransport:
    """记录提交的请求体；按顺序返回预设的响应，``OSError`` 表示网络不可达。"""

    def __init__(self, *responses: tuple[int, dict] | OSError) -> None:
        self.responses = list(responses)
        self.bodies: list[dict] = []
        self.headers: list[dict] = []

    def __call__(self, url, body, headers, timeout):
        self.bodies.append(json.loads(body))
        self.headers.append(headers)
        response = self.responses.pop(0) if self.responses else (200, {"status": "accepted"})
        if isinstance(response, OSError):
            raise response
        status, payload = response
        return status, json.dumps(payload).encode()

    def turn_ids(self) -> list[str]:
        return [body["external_turn_id"] for body in self.bodies]


def _harness(
    tmp_path: Path,
    transport: FakeTransport,
    *,
    first_run_at: float = STARTED_AT - 60,
    clock: Callable[[], float] | None = None,
    user_id: str | None = None,
    reader: ClaudeCodeReader | None = None,
) -> tuple[TranscriptSync, StateStore]:
    config = ConnectorConfig(
        url="http://hivememory.test",
        agent_id="claude_code",
        user_id=user_id,
        state_dir=tmp_path / "state",
    )
    store = StateStore(config.state_dir)
    store.first_run_at(first_run_at)
    sync = TranscriptSync(
        reader=reader or ClaudeCodeReader(),
        store=store,
        client=ImportClient(config, transport=transport),
        config=config,
        clock=clock or (lambda: os.path.getmtime(tmp_path) + 1),
        sleep=lambda _seconds: None,
    )
    return sync, store


def _log(tmp_path: Path) -> TranscriptWriter:
    return TranscriptWriter(tmp_path / "session.jsonl")


def test_cursor_advances_only_after_confirmation(tmp_path):
    """确认的一轮前进游标；网络不可达时保留游标，下次唤醒从未确认的一轮继续。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.turn("p2")
    log.turn("p3")  # 仍在进行
    transport = FakeTransport((200, {"status": "accepted"}), OSError("connection refused"))
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))

    first = sync.sync(str(log.path))
    state_after_failure = store.load(str(log.path))
    second = sync.sync(str(log.path))

    assert (first.confirmed, first.stopped) == (1, "retryable")
    assert state_after_failure.offset == log.prompt_offsets["p2"]
    assert state_after_failure.last_turn_id == "p1"
    assert (second.confirmed, second.stopped) == (1, None)
    assert transport.turn_ids() == ["p1", "p2", "p2"]
    assert store.load(str(log.path)).offset == log.prompt_offsets["p3"]


def test_service_unavailable_keeps_cursor_and_stops_the_transcript(tmp_path):
    """503 是可重试错误：不前进游标，也不继续提交后面的轮次。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.turn("p2")
    log.turn("p3")
    transport = FakeTransport(
        (503, {"error": "import_bus.unavailable", "reason": "queue_unavailable"})
    )
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))

    report = sync.sync(str(log.path))

    assert report.stopped == "retryable"
    assert transport.turn_ids() == ["p1"]
    assert store.load(str(log.path)).offset == 0
    assert store.rejections() == []


def test_client_error_is_recorded_without_content_and_cursor_moves_on(tmp_path):
    """4xx 重发不会改变结果：写入只含标识与状态的拒绝记录，继续提交下一轮。"""
    log = _log(tmp_path)
    log.turn("p1", question="很长的提问正文")
    log.turn("p2")
    log.turn("p3")
    transport = FakeTransport((413, {"error": "request.body_too_large"}))
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))

    report = sync.sync(str(log.path))

    assert (report.confirmed, report.rejected) == (1, 1)
    assert transport.turn_ids() == ["p1", "p2"]
    [rejection] = store.rejections()
    assert {key: rejection[key] for key in ("turn_id", "status", "error")} == {
        "turn_id": "p1",
        "status": 413,
        "error": "request.body_too_large",
    }
    assert "很长的提问正文" not in json.dumps(rejection, ensure_ascii=False)


@pytest.mark.parametrize("status", [400, 403, 404])
def test_configuration_errors_keep_cursor_instead_of_dropping_turns(tmp_path, status):
    """身份声明不合法、未获准入或地址不对与具体轮次无关：保留游标，修正配置后补发。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.turn("p2")
    transport = FakeTransport((status, {"error": "workspace.admission_denied"}))
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))

    report = sync.sync(str(log.path))

    assert report.stopped == "misconfigured"
    assert store.load(str(log.path)).offset == 0
    assert store.rejections() == []


def test_third_consecutive_server_error_rejects_the_turn(tmp_path):
    """同一轮连续三次服务器内部错误后按不可重试处理，避免一轮阻塞整份会话。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.turn("p2")
    log.turn("p3")
    transport = FakeTransport((500, {}), (500, {}), (500, {}))
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))

    reports = [sync.sync(str(log.path)) for _ in range(3)]

    assert [report.stopped for report in reports] == ["server_error", "server_error", None]
    assert transport.turn_ids() == ["p1", "p1", "p1", "p2"]
    assert [rejection["turn_id"] for rejection in store.rejections()] == ["p1"]


def test_outcome_is_recorded_before_submission_and_reused_on_resend(tmp_path):
    """结局先于提交写入；会话恢复后末尾一轮重新算作进行中，重发仍沿用原结局。"""
    log = _log(tmp_path)
    log.turn("p1")
    transport = FakeTransport(OSError("offline"))
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))
    store.set_session_ended(str(log.path), True)

    sync.sync(str(log.path))
    recorded = dict(store.load(str(log.path)).end_records)
    store.set_session_ended(str(log.path), False)
    sync.sync(str(log.path))

    assert recorded == {"p1": "cancelled"}
    assert [body["outcome"] for body in transport.bodies] == ["cancelled", "cancelled"]
    assert store.load(str(log.path)).end_records == {}


def test_outcome_written_by_a_hook_first_is_the_one_submitted(tmp_path):
    """同步判定结局之后、写入之前 hook 已写入结局时，提交的是已保存的结局。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.turn("p2")

    class HookRacingReader(ClaudeCodeReader):
        """读取之后、同步写入结局之前，模拟 hook 先为第一轮写入结局。"""

        def read(self, path, offset):
            turns = super().read(path, offset)
            store.record_end(path, "p1", "failed")
            return turns

    transport = FakeTransport()
    sync, store = _harness(tmp_path, transport, reader=HookRacingReader())
    sync.register(str(log.path))

    sync.sync(str(log.path))

    assert [(body["external_turn_id"], body["outcome"]) for body in transport.bodies] == [
        ("p1", "failed")
    ]


def test_deleted_transcript_is_forgotten_without_rejection(tmp_path):
    """会话记录被删除（例如过期清理）时移除它的状态，不写拒绝记录、不再补发。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.turn("p2")
    transport = FakeTransport()
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))
    sync.sync(str(log.path))
    log.path.unlink()

    report = sync.sync(str(log.path))

    assert report.stopped == "missing"
    assert store.registered_transcripts() == []
    assert store.rejections() == []


def test_complete_size_stops_at_the_last_full_line_across_read_chunks(tmp_path):
    """末尾仍在写入的不完整行不计入，即使它跨越多个向前读取的块。"""
    path = tmp_path / "partial.jsonl"
    path.write_bytes(b'{"a":1}\n' + b"x" * 200_000)

    assert complete_size(str(path)) == len(b'{"a":1}\n')
    assert complete_size(str(tmp_path / "missing.jsonl")) == 0


def test_in_progress_turn_waits_until_it_ends(tmp_path):
    """末尾仍在进行的一轮不提交；收到结束记录后才提交，结局取自结束记录。"""
    log = _log(tmp_path)
    log.turn("p1")
    transport = FakeTransport()
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))

    sync.sync(str(log.path))
    store.record_end(str(log.path), "p1", "completed")
    sync.sync(str(log.path))

    assert [(body["external_turn_id"], body["outcome"]) for body in transport.bodies] == [
        ("p1", "completed")
    ]


def test_trailing_turn_ended_before_its_last_lines_are_written_waits_for_them(tmp_path):
    """结束通知先于最后几行写入时不提交半截内容；写完后提交完整的一轮。"""
    log = _log(tmp_path)
    log.prompt("p1", "修一下登录")
    log.assistant(text("我先看看"), tool_use("t1", "Read", {"path": "a.py"}), message_id="m1")
    log.tool_result("t1", "源码")
    transport = FakeTransport()
    sync, store = _harness(tmp_path, transport, clock=lambda: os.path.getmtime(log.path) + 1)
    sync.register(str(log.path))
    store.record_end(str(log.path), "p1", "completed")

    sync.sync(str(log.path))
    submitted_early = list(transport.bodies)
    log.assistant(text("修好了"), message_id="m2")
    sync.sync(str(log.path))

    assert submitted_early == []
    [body] = transport.bodies
    assert body["assistant_final_text"] == "我先看看\n\n修好了"


def test_quiet_transcript_submits_ended_turn_without_a_clean_finish(tmp_path):
    """已结束、内容没有正常收尾的一轮，在会话记录静止超过等待时限后按读到的内容提交。"""
    log = _log(tmp_path)
    log.prompt("p1", "跑测试")
    log.assistant(tool_use("t1", "Bash", {"command": "pytest"}), message_id="m1")
    transport = FakeTransport()
    sync, store = _harness(
        tmp_path,
        transport,
        clock=lambda: os.path.getmtime(log.path) + FLUSH_GRACE_SECONDS + 1,
    )
    sync.register(str(log.path))
    store.record_end(str(log.path), "p1", "failed")

    sync.sync(str(log.path))

    assert [(body["external_turn_id"], body["outcome"]) for body in transport.bodies] == [
        ("p1", "failed")
    ]


def test_turn_that_grows_while_the_previous_one_submits_is_reread(tmp_path):
    """提交前一轮期间末尾一轮写完并收到结束记录：重新读取后提交完整内容，不用旧内容。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.prompt("p2", "加个测试")
    log.assistant(text("开始写"), tool_use("t2", "Write", {"path": "t.py"}), message_id="m2")
    sync_ref: list[StateStore] = []

    class GrowingTransport(FakeTransport):
        def __call__(self, url, body, headers, timeout):
            if not self.bodies:
                # p1 提交期间，p2 的剩余内容写入会话记录，Stop 写下结束记录。
                log.tool_result("t2", "ok")
                log.assistant(text("测试已加上"), message_id="m3")
                sync_ref[0].record_end(str(log.path), "p2", "completed")
            return super().__call__(url, body, headers, timeout)

    transport = GrowingTransport()
    sync, store = _harness(tmp_path, transport, clock=lambda: os.path.getmtime(log.path) + 1)
    sync_ref.append(store)
    sync.register(str(log.path))

    sync.sync(str(log.path))

    assert transport.turn_ids() == ["p1", "p2"]
    second = transport.bodies[1]
    assert second["assistant_final_text"] == "开始写\n\n测试已加上"
    assert [event["kind"] for event in second["events"]] == [
        "assistant_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]


def test_interrupted_and_stale_turns_end_as_cancelled(tmp_path):
    """中断标记立即结束一轮；没有会话结束通知的会话末尾超过 30 分钟未更新时按中断结束。"""
    interrupted = TranscriptWriter(tmp_path / "interrupted.jsonl")
    interrupted.turn("p1")
    interrupted.user_text("[Request interrupted by user for tool use]")
    stale = TranscriptWriter(tmp_path / "stale.jsonl")
    stale.turn("q1")
    transport = FakeTransport()
    sync, _ = _harness(
        tmp_path,
        transport,
        clock=lambda: os.path.getmtime(stale.path) + STALE_SECONDS + 1,
    )
    for log in (interrupted, stale):
        sync.register(str(log.path))
        sync.sync(str(log.path))

    assert [(body["external_turn_id"], body["outcome"]) for body in transport.bodies] == [
        ("p1", "cancelled"),
        ("q1", "cancelled"),
    ]


def test_existing_session_is_not_backfilled(tmp_path):
    """连接器首次运行之前开始的会话从当前末尾开始，只提交之后的轮次。"""
    log = _log(tmp_path)
    log.turn("old-1")
    log.turn("old-2")
    transport = FakeTransport()
    sync, store = _harness(tmp_path, transport, first_run_at=STARTED_AT + 60)
    sync.register(str(log.path))
    log.turn("new-1")
    log.turn("new-2")

    sync.sync(str(log.path))

    assert transport.turn_ids() == ["new-1"]
    assert store.load(str(log.path)).origin == "existing"


def test_recent_turns_come_from_confirmed_turns_before_the_cursor(tmp_path):
    """最近轮次取游标之前已确认的至多三轮，跨同步保持。"""
    log = _log(tmp_path)
    for index in range(1, 5):
        log.turn(f"p{index}", question=f"问题{index}", answer=f"回答{index}")
    log.turn("p5")
    transport = FakeTransport()
    sync, _ = _harness(tmp_path, transport)
    sync.register(str(log.path))
    sync.sync(str(log.path))
    log.turn("p6")

    sync.sync(str(log.path))

    assert transport.turn_ids() == ["p1", "p2", "p3", "p4", "p5"]
    assert transport.bodies[3]["recent_turns"] == [
        {
            "user_message": f"问题{index}",
            "assistant_text": f"回答{index}",
            "external_turn_id": f"p{index}",
        }
        for index in (1, 2, 3)
    ]
    assert [turn["user_message"] for turn in transport.bodies[4]["recent_turns"]] == [
        "问题2",
        "问题3",
        "问题4",
    ]


def test_recent_turn_text_is_truncated_for_routing_context(tmp_path):
    """最近轮次只供路由理解上文：过长的正文截断，避免拖着后续轮次超过请求体上限。"""
    log = _log(tmp_path)
    log.turn("p1", question="长" * (RECENT_TEXT_LIMIT + 100))
    log.turn("p2")
    log.turn("p3")
    transport = FakeTransport()
    sync, _ = _harness(tmp_path, transport)
    sync.register(str(log.path))

    sync.sync(str(log.path))

    [recent] = transport.bodies[1]["recent_turns"]
    assert recent["user_message"] == "长" * RECENT_TEXT_LIMIT


def test_rewritten_transcript_resumes_after_last_confirmed_turn(tmp_path):
    """会话记录被重写、游标不再落在行首时，从最后一个已确认的轮次之后继续。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.turn("p2")
    transport = FakeTransport()
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))
    sync.sync(str(log.path))
    original = log.path.read_text(encoding="utf-8")
    log.path.write_text('{"type":"summary","summary":"x"}\n' + original, encoding="utf-8")
    log.turn("p3")

    sync.sync(str(log.path))

    assert transport.turn_ids() == ["p1", "p2"]
    assert store.load(str(log.path)).last_turn_id == "p2"


def test_lost_cursor_blocks_the_transcript_instead_of_resending_it(tmp_path):
    """找不到最后一个已确认的轮次时停止处理这份会话记录并写入拒绝记录。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.turn("p2")
    transport = FakeTransport()
    sync, store = _harness(tmp_path, transport)
    sync.register(str(log.path))
    sync.sync(str(log.path))
    log.path.write_text("", encoding="utf-8")
    log.turn("x1")
    log.turn("x2")

    report = sync.sync(str(log.path))

    assert report.stopped == "blocked"
    assert transport.turn_ids() == ["p1"]
    assert store.load(str(log.path)).blocked is True
    assert store.rejections()[-1]["error"] == "cursor_lost"


def test_request_carries_identity_claims_and_import_fields(tmp_path):
    """请求带来源、外部标识、Agent 与可选的用户请求头。"""
    log = _log(tmp_path)
    log.turn("p1")
    log.turn("p2")
    transport = FakeTransport()
    sync, _ = _harness(tmp_path, transport, user_id="alice")
    sync.register(str(log.path))

    sync.sync(str(log.path))

    [body] = transport.bodies
    assert {key: body[key] for key in ("source", "external_conversation_id", "agent_id")} == {
        "source": "claude-code",
        "external_conversation_id": "session-test",
        "agent_id": "claude_code",
    }
    assert transport.headers[0]["X-User-ID"] == "alice"


def test_concurrent_updates_keep_state_consistent(tmp_path):
    """多个写入方同时读改写同一份状态时，每个写入都保留，状态文件始终完整可读。

    每个线程各自打开锁文件，与多个 hook 进程一样经文件锁互斥。
    """
    workers = 8
    log = _log(tmp_path)
    store = StateStore(tmp_path / "state")
    store.register(str(log.path), origin="new", offset=0)

    unreadable: list[str] = []

    def record(worker: int) -> None:
        for turn in range(25):
            store.record_end(str(log.path), f"w{worker}-t{turn}", "completed")
            # 线程内的断言不会让测试失败：先收集，join 之后统一断言。
            if store.load(str(log.path)) is None:
                unreadable.append(f"w{worker}-t{turn}")

    threads = [threading.Thread(target=record, args=(worker,)) for worker in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert unreadable == []
    assert len(store.load(str(log.path)).end_records) == workers * 25


def test_request_body_satisfies_the_import_bus_contract(tmp_path):
    """连接器构造的请求体通过服务端请求模型的校验：两侧的字段不漂移。"""
    from hivememory.server.models.import_bus import ImportTurnRequest
    from tests.unit.connectors.transcripts import text, thinking, tool_use

    log = _log(tmp_path)
    log.turn("p0")
    log.prompt("p1", "修一下测试")
    log.assistant(thinking("先跑测试"), tool_use("toolu-1", "Bash", {"command": "pytest"}))
    log.tool_result("toolu-1", "1 failed", is_error=True)
    log.assistant(text("已修复"), message_id="m2")
    log.turn("p2")
    transport = FakeTransport()
    sync, _ = _harness(tmp_path, transport)
    sync.register(str(log.path))

    sync.sync(str(log.path))

    requests = [ImportTurnRequest.model_validate(body) for body in transport.bodies]
    assert [request.external_turn_id for request in requests] == ["p0", "p1"]
    assert [event.kind for event in requests[1].events] == [
        "thought",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]
