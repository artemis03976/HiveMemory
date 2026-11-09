"""连接器状态：每份会话记录的游标、结束记录与失败计数，以及拒绝记录。

状态按会话记录分文件保存在状态目录中。多个 hook 进程与同步进程会同时运行，
因此每次读改写都持有该会话记录的状态锁，并以写临时文件再替换的方式原子
写入；状态锁只在短暂的读改写期间持有，网络提交期间不持有，hook 因此不会
被同步进程阻塞。同一份会话记录的同步由另一把同步锁串行化。

状态与拒绝记录只保存标识、位置与状态码，不保存会话正文。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import fcntl
except ImportError:  # Windows 没有 fcntl，改用 msvcrt 的字节锁。
    fcntl = None  # type: ignore[assignment]
    import msvcrt

Outcome = Literal["completed", "cancelled", "failed"]
TranscriptOrigin = Literal["new", "existing"]

# 最近若干轮的起点位置供 Gateway 理解上文；数量与服务端截取的最近轮数一致。
RECENT_TURNS = 3


@dataclass
class TranscriptState:
    """一份会话记录的连接器状态。

    ``offset`` 是游标：下一次从这里读取，它之前的轮次都已得到确认；
    ``last_turn_id`` 是最后一个已确认轮次的标识，游标位置失效时据此重新定位；
    ``context_offsets`` 是最近已确认轮次的起点，用于给 Gateway 附上上文。
    ``end_records`` 记录各轮的结局，先于提交写入，重发沿用同一结局；
    ``failures`` 记录各轮连续得到服务器内部错误的次数。
    """

    path: str
    origin: TranscriptOrigin
    offset: int = 0
    last_turn_id: str | None = None
    context_offsets: list[int] = field(default_factory=list)
    end_records: dict[str, Outcome] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    session_ended: bool = False
    blocked: bool = False

    @property
    def read_offset(self) -> int:
        """读取的起点：包含最近已确认轮次，以便为下一轮附上上文。"""
        return min([self.offset, *self.context_offsets])

    def advance(self, *, turn_id: str, turn_start: int, next_offset: int) -> None:
        """一轮得到确认（或不可重试地被拒绝）后前进游标，并清理该轮的记录。"""
        self.offset = next_offset
        self.last_turn_id = turn_id
        self.context_offsets = [*self.context_offsets, turn_start][-RECENT_TURNS:]
        self.end_records.pop(turn_id, None)
        self.failures.pop(turn_id, None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptState:
        return cls(**data)


class StateStore:
    """状态目录的读写入口。"""

    def __init__(self, state_dir: Path) -> None:
        self._dir = state_dir
        self._transcripts = state_dir / "transcripts"

    @property
    def state_dir(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # 会话记录状态
    # ------------------------------------------------------------------

    def load(self, transcript: str) -> TranscriptState | None:
        """读取一份会话记录的状态；尚未登记时返回 ``None``。"""
        with self._lock(self._state_path(transcript).with_suffix(".lock")):
            return self._read(transcript)

    @contextmanager
    def update(self, transcript: str) -> Iterator[TranscriptState]:
        """持状态锁读改写一份已登记会话记录的状态，退出时原子写回。"""
        with self._lock(self._state_path(transcript).with_suffix(".lock")):
            state = self._read(transcript)
            if state is None:
                raise KeyError(f"transcript is not registered: {transcript}")
            yield state
            self._write(state)

    def register(self, transcript: str, *, origin: TranscriptOrigin, offset: int) -> bool:
        """首次见到一份会话记录时登记游标；已登记时不改动，返回是否新登记。"""
        with self._lock(self._state_path(transcript).with_suffix(".lock")):
            if self._read(transcript) is not None:
                return False
            self._write(TranscriptState(path=transcript, origin=origin, offset=offset))
            return True

    def record_end(self, transcript: str, turn_id: str, outcome: Outcome) -> Outcome:
        """写入一轮的结局并返回实际保存的结局；已有结局时保留原结局。

        同步与 hook 可能同时为同一轮写结局，先写入的生效；提交方使用返回值，
        首次提交与重发因此始终发送同一结局。
        """
        with self.update(transcript) as state:
            return state.end_records.setdefault(turn_id, outcome)

    def forget(self, transcript: str) -> None:
        """会话记录已被删除时移除它的状态，``--all`` 补发不再扫描它。"""
        with self._lock(self._state_path(transcript).with_suffix(".lock")):
            self._state_path(transcript).unlink(missing_ok=True)

    def set_session_ended(self, transcript: str, ended: bool) -> None:
        """会话结束时末尾未结束的轮次随之结束；会话恢复时撤销这一标记。"""
        with self.update(transcript) as state:
            state.session_ended = ended

    def registered_transcripts(self) -> list[str]:
        """全部已登记会话记录的路径。"""
        if not self._transcripts.is_dir():
            return []
        paths = []
        for state_file in sorted(self._transcripts.glob("*.json")):
            try:
                paths.append(json.loads(state_file.read_text(encoding="utf-8"))["path"])
            except (OSError, ValueError, KeyError):
                continue
        return paths

    @contextmanager
    def sync_lock(self, transcript: str) -> Iterator[None]:
        """串行化同一份会话记录的同步；与状态锁分开，提交期间不阻塞 hook。"""
        with self._lock(self._state_path(transcript).with_suffix(".sync.lock")):
            yield

    # ------------------------------------------------------------------
    # 全局记录
    # ------------------------------------------------------------------

    def first_run_at(self, now: float | None = None) -> float:
        """连接器首次运行的时间；会话记录早于它开始的不回填。"""
        marker = self._dir / "connector.json"
        with self._lock(self._dir / "connector.lock"):
            try:
                return float(json.loads(marker.read_text(encoding="utf-8"))["first_run_at"])
            except (OSError, ValueError, KeyError):
                first_run = time.time() if now is None else now
                self._atomic_write(marker, {"first_run_at": first_run})
                return first_run

    def append_rejection(self, record: dict[str, Any]) -> None:
        """追加一条拒绝记录（只含标识、状态码与原因，不含正文）。"""
        with self._lock(self._dir / "rejections.lock"):
            with (self._dir / "rejections.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def rejections(self) -> list[dict[str, Any]]:
        path = self._dir / "rejections.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _state_path(self, transcript: str) -> Path:
        digest = hashlib.sha256(os.path.abspath(transcript).encode("utf-8")).hexdigest()
        return self._transcripts / f"{digest[:24]}.json"

    def _read(self, transcript: str) -> TranscriptState | None:
        path = self._state_path(transcript)
        if not path.exists():
            return None
        return TranscriptState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _write(self, state: TranscriptState) -> None:
        self._atomic_write(self._state_path(state.path), asdict(state))

    @staticmethod
    def _atomic_write(path: Path, data: dict[str, Any]) -> None:
        """写临时文件后替换：并发读取方只会看到完整的旧版本或新版本。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    @contextmanager
    def _lock(path: Path) -> Iterator[None]:
        """跨进程互斥锁；锁文件常驻，只用于加锁，不承载内容。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            else:
                _windows_lock(handle)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                else:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _windows_lock(handle: Any) -> None:
    """msvcrt 的阻塞加锁约 10 秒后放弃；同步锁会跨越网络提交，因此持续重试直到取得。"""
    while True:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return
        except OSError:
            continue
