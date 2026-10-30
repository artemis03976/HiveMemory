"""同步循环：读取游标之后已经结束的轮次，按顺序提交，确认之后游标才前进。

规则：

1. **结束的判定**：一轮在其后出现下一轮、harness 写入中断标记、已有结束
   记录（例如 hook 报告的正常结束或失败）或会话已结束时才算结束；会话记录
   末尾仍在进行的轮次不提交。没有会话结束通知的崩溃会话，末尾超过 30 分钟
   没有新条目时按中断结束。
   末尾一轮已判定结束、但读到的内容还没有正常收尾时（结束通知先于最后几行
   写入），短暂等待会话记录写完再提交；读取之后会话记录又有新内容时重新
   读取，不以读取时的旧内容提交。
2. **结局先于提交写入**：结局写入状态后才提交，发送失败后的重发沿用原结局。
3. **游标只在确认后前进**：可重试错误（网络错误、服务暂不可用）与连接器配置
   错误停止处理这份会话记录并保留游标，等下次唤醒；因这一轮内容被拒绝的
   4xx 写入拒绝记录后前进；同一轮连续三次服务器内部错误也按不可重试处理，
   避免一轮阻塞整份会话。
4. **首次见到的会话记录**：在连接器首次运行之后开始的从头读取；更早开始的
   （接入之前已有、后来被恢复的会话）游标放在当前末尾，不回填。
5. **游标失效**（会话记录被重写或缩短，游标不在行首或最后确认的一轮不再从
   记录的位置开始）：从头扫描，在最后一个已确认的轮次之后继续；找不到该轮
   时停止处理这份会话记录并写入拒绝记录，不重发整个会话。

重发由服务端按推导出的交互标识去重：在服务端查重的保留窗口内（同一服务
进程、最近接纳的若干交互），确认丢失后的重发不会形成重复交互。会话记录
被删除时移除它的状态，不再补发。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from connector.core.client import ImportClient, SubmitResult
from connector.core.config import ConnectorConfig
from connector.core.log import ConnectorLog
from connector.core.state import (
    RECENT_TURNS,
    Outcome,
    StateStore,
    TranscriptOrigin,
    TranscriptState,
)
from connector.core.transcript import (
    TranscriptReader,
    TranscriptTurn,
    complete_size,
    is_line_start,
)

STALE_SECONDS = 30 * 60
# 末尾一轮已判定结束但内容尚未收尾时，等待会话记录写完的时限与轮询间隔；
# 会话记录静止超过这个时限后，按读到的内容提交。
FLUSH_GRACE_SECONDS = 10.0
FLUSH_POLL_SECONDS = 0.5
# 等待按轮询次数封顶，不依赖时钟前进。
MAX_FLUSH_POLLS = int(FLUSH_GRACE_SECONDS / FLUSH_POLL_SECONDS)
# 最近轮次只供 Gateway 理解上文，每段正文截断到这个长度，避免一轮过大的
# 内容拖着后续轮次一起超过服务端的请求体上限。
RECENT_TEXT_LIMIT = 2000
MAX_SERVER_ERRORS = 3
# 每轮同步结束后重新读取一次，接住同步期间新结束的轮次；有上限以免无限循环。
MAX_ROUNDS = 10


@dataclass
class SyncReport:
    """一次同步的结果：确认与拒绝的轮数，以及提前停止的原因。"""

    confirmed: int = 0
    rejected: int = 0
    stopped: str | None = None
    # 末尾一轮已判定结束、内容尚未收尾，需要等会话记录写完后重新读取。
    awaiting_flush: bool = False
    # 读取之后会话记录有新内容，需要重新读取。
    reread: bool = False


class TranscriptSync:
    """一份会话记录的登记与同步。"""

    def __init__(
        self,
        *,
        reader: TranscriptReader,
        store: StateStore,
        client: ImportClient,
        config: ConnectorConfig,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._reader = reader
        self._store = store
        self._client = client
        self._config = config
        self._clock = clock
        self._sleep = sleep
        self._log = ConnectorLog(store.state_dir)

    def register(self, path: str) -> None:
        """首次见到会话记录时登记游标：接入之前开始的会话不回填。"""
        if self._store.load(path) is not None:
            return
        started = self._reader.started_at(path)
        first_run = self._store.first_run_at(self._clock())
        origin: TranscriptOrigin = "new" if started is None or started >= first_run else "existing"
        offset = 0 if origin == "new" else complete_size(path)
        if self._store.register(path, origin=origin, offset=offset):
            self._log.write(f"registered transcript={path} origin={origin} offset={offset}")

    def sync(self, path: str) -> SyncReport:
        """提交游标之后已经结束的轮次；同一份会话记录同时只有一个同步在运行。"""
        report = SyncReport()
        with self._store.sync_lock(path):
            flush_polls = 0
            rounds = 0
            while rounds < MAX_ROUNDS:
                state = self._store.load(path)
                if state is None:
                    return report
                if not os.path.exists(path):
                    self._store.forget(path)
                    self._log.write(f"forgot deleted transcript={path}")
                    report.stopped = "missing"
                    return report
                if state.blocked:
                    report.stopped = "blocked"
                    return report
                if not self._cursor_valid(path, state):
                    if not self._relocate(path, state):
                        report.stopped = "blocked"
                        return report
                    rounds += 1
                    continue
                report.awaiting_flush = False
                report.reread = False
                progressed = self._submit_ended_turns(path, state, report)
                if report.stopped is not None:
                    return report
                if progressed or report.reread:
                    rounds += 1
                    continue
                if report.awaiting_flush and flush_polls < MAX_FLUSH_POLLS:
                    flush_polls += 1
                    self._sleep(FLUSH_POLL_SECONDS)
                    continue
                return report
        return report

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _submit_ended_turns(self, path: str, state: TranscriptState, report: SyncReport) -> bool:
        # 先取得完整行的长度再读取：之后长度变化说明读取之后又写入了内容。
        read_size = complete_size(path)
        turns = self._reader.read(path, state.read_offset)
        context = [turn for turn in turns if turn.start_offset < state.offset][-RECENT_TURNS:]
        pending = [turn for turn in turns if turn.start_offset >= state.offset]
        progressed = False
        for index, turn in enumerate(pending):
            outcome = self._outcome(path, state, turn)
            if outcome is None:
                break
            if not self._content_settled(path, state, turn, read_size, report):
                break
            # 先写入再提交，并以实际保存的结局提交：hook 可能已为这一轮写入结局。
            outcome = self._store.record_end(path, turn.turn_id, outcome)
            result = self._client.submit(self._body(turn, outcome, context))
            next_offset = (
                pending[index + 1].start_offset if index + 1 < len(pending) else turn.end_offset
            )
            if not self._settle(path, turn, next_offset, result, report):
                return progressed
            context = [*context, turn][-RECENT_TURNS:]
            progressed = True
            # 提交期间 hook 可能写入了新的结束记录，下一轮按最新状态判定。
            refreshed = self._store.load(path)
            if refreshed is None:
                return progressed
            state = refreshed
        return progressed

    def _content_settled(
        self,
        path: str,
        state: TranscriptState,
        turn: TranscriptTurn,
        read_size: int,
        report: SyncReport,
    ) -> bool:
        """读到的内容是否就是该轮的全部内容，可以据此提交。

        其后已有下一轮或中断标记的轮次，内容到下一行为止已经完整。末尾一轮
        只在读取之后会话记录没有新内容、且该轮已经正常收尾或会话记录已静止
        一段时间（会话已结束、超时或写入已停止）时才提交；否则交给调用方重新
        读取或稍后再读。
        """
        if turn.closed_by_next_turn or turn.interrupted:
            return True
        if complete_size(path) != read_size:
            report.reread = True
            return False
        if turn.looks_complete or state.session_ended or self._quiet_for(path, FLUSH_GRACE_SECONDS):
            return True
        report.awaiting_flush = True
        return False

    def _quiet_for(self, path: str, seconds: float) -> bool:
        try:
            return self._clock() - os.path.getmtime(path) > seconds
        except OSError:
            return False

    def _outcome(self, path: str, state: TranscriptState, turn: TranscriptTurn) -> Outcome | None:
        """一轮的结局；仍在进行时返回 ``None``。"""
        recorded = state.end_records.get(turn.turn_id)
        if recorded is not None:
            return recorded
        if turn.closed_by_next_turn or turn.interrupted:
            return "cancelled"
        if state.session_ended or self._is_stale(path):
            return "cancelled"
        return None

    def _settle(
        self,
        path: str,
        turn: TranscriptTurn,
        next_offset: int,
        result: SubmitResult,
        report: SyncReport,
    ) -> bool:
        """按提交结果处理游标；返回是否继续处理这份会话记录的后续轮次。"""
        if result.kind == "confirmed":
            self._advance(path, turn, next_offset)
            report.confirmed += 1
            self._log.write(
                f"confirmed turn={turn.turn_id} status={result.import_status} transcript={path}"
            )
            return True
        if result.kind == "rejected":
            self._reject(path, turn, next_offset, result, report)
            return True
        if result.kind == "server_error":
            with self._store.update(path) as state:
                failures = state.failures.get(turn.turn_id, 0) + 1
                state.failures[turn.turn_id] = failures
            if failures >= MAX_SERVER_ERRORS:
                self._reject(path, turn, next_offset, result, report)
                return True
        report.stopped = result.kind
        self._log.write(
            f"stopped turn={turn.turn_id} kind={result.kind} status={result.status} "
            f"reason={result.reason} transcript={path}"
        )
        return False

    def _advance(self, path: str, turn: TranscriptTurn, next_offset: int) -> None:
        with self._store.update(path) as state:
            state.advance(
                turn_id=turn.turn_id, turn_start=turn.start_offset, next_offset=next_offset
            )

    def _reject(
        self,
        path: str,
        turn: TranscriptTurn,
        next_offset: int,
        result: SubmitResult,
        report: SyncReport,
    ) -> None:
        self._store.append_rejection(
            {
                "at": self._clock(),
                "transcript": path,
                "conversation_id": turn.conversation_id,
                "turn_id": turn.turn_id,
                "status": result.status,
                "error": result.error,
                "reason": result.reason,
            }
        )
        self._advance(path, turn, next_offset)
        report.rejected += 1
        self._log.write(
            f"rejected turn={turn.turn_id} status={result.status} error={result.error} "
            f"transcript={path}"
        )

    def _cursor_valid(self, path: str, state: TranscriptState) -> bool:
        """游标仍指向原来的内容：位置都在行首，且最后确认的一轮仍从记录的位置开始。

        只检查行首不够：重写后的文件恰好在原位置换行时，游标会悄悄落到别的
        内容上，因此以最后一个已确认轮次的起点核对内容。
        """
        if not all(
            is_line_start(path, offset) for offset in [state.offset, *state.context_offsets]
        ):
            return False
        if state.last_turn_id is None or not state.context_offsets:
            return True
        return self._reader.turn_id_at(path, state.context_offsets[-1]) == state.last_turn_id

    def _relocate(self, path: str, state: TranscriptState) -> bool:
        """游标失效后从头扫描，在最后一个已确认的轮次之后继续。"""
        turns = self._reader.read(path, 0)
        if state.last_turn_id is None:
            offset = 0 if state.origin == "new" else complete_size(path)
            with self._store.update(path) as current:
                current.offset = offset
                current.context_offsets = []
            self._log.write(f"cursor reset offset={offset} transcript={path}")
            return True
        index = next(
            (i for i, turn in enumerate(turns) if turn.turn_id == state.last_turn_id), None
        )
        if index is None:
            self._store.append_rejection(
                {
                    "at": self._clock(),
                    "transcript": path,
                    "conversation_id": None,
                    "turn_id": state.last_turn_id,
                    "status": None,
                    "error": "cursor_lost",
                    "reason": "last confirmed turn not found after transcript rewrite",
                }
            )
            with self._store.update(path) as current:
                current.blocked = True
            self._log.write(f"blocked transcript={path} reason=cursor_lost")
            return False
        next_offset = (
            turns[index + 1].start_offset if index + 1 < len(turns) else turns[index].end_offset
        )
        with self._store.update(path) as current:
            current.offset = next_offset
            current.context_offsets = [
                turn.start_offset for turn in turns[max(0, index + 1 - RECENT_TURNS) : index + 1]
            ]
        self._log.write(f"cursor relocated offset={next_offset} transcript={path}")
        return True

    def _is_stale(self, path: str) -> bool:
        try:
            return self._clock() - os.path.getmtime(path) > STALE_SECONDS
        except OSError:
            return False

    def _body(
        self, turn: TranscriptTurn, outcome: Outcome, context: list[TranscriptTurn]
    ) -> dict[str, Any]:
        return {
            "source": self._reader.source,
            "external_conversation_id": turn.conversation_id,
            "external_turn_id": turn.turn_id,
            "agent_id": self._config.agent_id,
            "outcome": outcome,
            "user_message": turn.user_message,
            "events": turn.events,
            "model_used": turn.model_used,
            "recent_turns": [
                {
                    "user_message": previous.user_message[:RECENT_TEXT_LIMIT],
                    "assistant_text": (previous.assistant_final_text or "")[:RECENT_TEXT_LIMIT],
                    # 服务端据此查出上一轮实际落位的话题，话题路由失败时续接。
                    "external_turn_id": previous.turn_id,
                }
                for previous in context
            ],
        }
