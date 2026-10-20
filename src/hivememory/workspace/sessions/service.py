"""会话服务：同步管理容量、认领与历史，不依赖任务进程或资源库。"""

from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.models import WorkspaceIdentity
from hivememory.core.protocol.models import (
    ConversationTurn,
    InteractionPayload,
    SessionCompactionCheckpoint,
)
from hivememory.utils.time import utc_now
from hivememory.workspace.contracts.errors import (
    SessionBusyError,
    SessionCapacityError,
    SessionCheckpointError,
    SessionClosedError,
    SessionNotFoundError,
)


class SessionSummary(BaseModel):
    """会话列表的轻量快照，不复制历史正文。"""

    session_id: str
    belong_to: WorkspaceIdentity
    title: str
    created_at: datetime
    updated_at: datetime
    turn_count: int = 0
    active_process_id: str | None = None

    model_config = ConfigDict(frozen=True)


class ConversationSession(SessionSummary):
    """会话资源快照；历史按封口顺序排列，任意 Agent 都可继续该会话。"""

    turns: tuple[ConversationTurn, ...] = ()
    compaction_checkpoint: SessionCompactionCheckpoint | None = None


class ConversationSessionService:
    """单事件循环内的会话真相源，认领、追加与释放均不跨越 await。

    内部状态不外泄：写入与读取都复制 payload，避免可变交互模型被调用方
    修改。容量在认领前预留，同一会话至多有一个进程持有写入资格。
    """

    def __init__(
        self,
        config: WorkspaceSessionConfig,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._config = config
        self._clock = clock
        self._sessions: dict[str, ConversationSession] = {}
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise SessionClosedError("会话服务已关闭")

    def _get(self, belong_to: WorkspaceIdentity, session_id: str) -> ConversationSession:
        self._require_open()
        session = self._sessions.get(session_id)
        # 不泄露其他 Workspace 的会话是否存在。
        if session is None or session.belong_to != belong_to:
            raise SessionNotFoundError("会话不存在")
        return session

    def create(self, belong_to: WorkspaceIdentity, title: str = "") -> ConversationSession:
        """创建空会话；配额用尽时显式拒绝，不淘汰旧会话。"""
        self._require_open()
        if (
            sum(s.belong_to == belong_to for s in self._sessions.values())
            >= self._config.max_sessions_per_workspace
        ):
            raise SessionCapacityError("Workspace 会话数已达上限")
        now = self._clock()
        session = ConversationSession(
            session_id=f"session_{uuid4().hex}",
            belong_to=belong_to,
            title=title.strip(),
            created_at=now,
            updated_at=now,
        )
        self._sessions[session.session_id] = session
        return session.model_copy(deep=True)

    def list_sessions(self, belong_to: WorkspaceIdentity) -> tuple[SessionSummary, ...]:
        """按最近更新时间列出指定 Workspace 的会话。"""
        self._require_open()
        return tuple(
            SessionSummary(**s.model_dump(exclude={"turns", "compaction_checkpoint"}))
            for s in sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)
            if s.belong_to == belong_to
        )

    def get(self, belong_to: WorkspaceIdentity, session_id: str) -> ConversationSession:
        """返回完整历史快照，修改返回值不会污染已封口的会话。"""
        return self._get(belong_to, session_id).model_copy(deep=True)

    def rename(
        self, belong_to: WorkspaceIdentity, session_id: str, title: str
    ) -> ConversationSession:
        """重命名会话，不改变历史或运行中的认领。"""
        session = self._get(belong_to, session_id).model_copy(
            update={"title": title.strip(), "updated_at": self._clock()}
        )
        self._sessions[session_id] = session
        return session.model_copy(deep=True)

    def delete(self, belong_to: WorkspaceIdentity, session_id: str) -> None:
        """删除空闲会话；不级联删除 Topic 或 Memory。"""
        session = self._get(belong_to, session_id)
        if session.active_process_id is not None:
            raise SessionBusyError("会话仍有运行中的进程")
        del self._sessions[session_id]

    def claim(self, belong_to: WorkspaceIdentity, session_id: str, process_id: str) -> None:
        """同步认领会话并检查轮数容量；同一进程也不能重复认领。"""
        session = self._get(belong_to, session_id)
        if session.active_process_id is not None:
            raise SessionBusyError("会话仍有运行中的进程")
        if session.turn_count >= self._config.max_turns_per_session:
            raise SessionCapacityError("会话轮数已达上限，请新建会话")
        self._sessions[session_id] = session.model_copy(update={"active_process_id": process_id})

    def release(self, belong_to: WorkspaceIdentity, session_id: str, process_id: str) -> None:
        """只释放对应进程的认领，迟到关闭不会清除后继进程的认领。"""
        session = self._get(belong_to, session_id)
        if session.active_process_id == process_id:
            self._sessions[session_id] = session.model_copy(update={"active_process_id": None})

    def append(
        self,
        belong_to: WorkspaceIdentity,
        session_id: str,
        *,
        process_id: str,
        agent_id: str,
        payload: InteractionPayload,
        topic_id: str,
    ) -> None:
        """认领者同步追加已封口的一轮；重复封口视为接线错误。"""
        session = self._get(belong_to, session_id)
        if session.active_process_id != process_id:
            raise SessionBusyError("进程未持有会话认领")
        if any(t.interaction_id == process_id for t in session.turns):
            raise ValueError("同一进程不能重复写入会话")
        if session.turn_count >= self._config.max_turns_per_session:
            raise SessionCapacityError("会话轮数已达上限")
        now = self._clock()
        turn = ConversationTurn(
            interaction_id=process_id,
            agent_id=agent_id,
            payload=payload.model_copy(deep=True),
            topic_id=topic_id,
            created_at=now,
        )
        self._sessions[session_id] = session.model_copy(
            update={
                "turns": (*session.turns, turn),
                "turn_count": session.turn_count + 1,
                "title": session.title or payload.user_message.strip()[:80],
                "updated_at": now,
            }
        )

    def accept_checkpoint(
        self,
        belong_to: WorkspaceIdentity,
        session_id: str,
        *,
        process_id: str,
        checkpoint: SessionCompactionCheckpoint,
    ) -> None:
        """认领者同步提交旧轮次的检查点；完整校验后才替换已有视图。"""
        session = self._get(belong_to, session_id)
        if session.active_process_id != process_id:
            raise SessionBusyError("进程未持有会话认领")
        positions = {turn.interaction_id: index for index, turn in enumerate(session.turns)}

        def position(cursor: str | None) -> int:
            if cursor is None:
                return -1
            if cursor not in positions:
                raise SessionCheckpointError("检查点游标必须指向本会话已有轮次")
            return positions[cursor]

        reduced = position(checkpoint.reduced_through)
        summarized = position(checkpoint.summarized_through)
        if reduced < summarized:
            raise SessionCheckpointError("减重边界不能早于摘要覆盖")
        previous = session.compaction_checkpoint
        # 游标只在同一产生方、同一格式版本之间可比：换用新格式的 CPU 会忽略旧视图并
        # 从头重建，若仍要求不回退，新视图会被永久拒绝。
        if (
            previous is not None
            and (previous.producer, previous.format_version)
            == (checkpoint.producer, checkpoint.format_version)
            and (
                reduced < position(previous.reduced_through)
                or summarized < position(previous.summarized_through)
            )
        ):
            raise SessionCheckpointError("检查点游标不能后退")
        self._sessions[session_id] = session.model_copy(
            update={
                "compaction_checkpoint": checkpoint.model_copy(deep=True),
                "updated_at": self._clock(),
            }
        )

    def close_and_clear(self) -> None:
        """进程全部关闭后清空会话，后续读写显式失败。"""
        self._closed = True
        self._sessions.clear()
