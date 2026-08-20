"""System 应用层持有的 Chat Run 运行时控制句柄。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models import (
    IdentityScope,
    require_workspace_access_context,
)


class ChatRunPhase(str, Enum):
    """Chat Run 当前所在的编排阶段。"""

    CREATED = "created"
    GATEWAY = "gateway"
    PREPARE = "prepare"
    ALICE = "alice"
    FINALIZE = "finalize"
    TERMINAL = "terminal"


class ChatRunOutcome(str, Enum):
    """Chat Run 的持久终态事实。"""

    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class StopResult:
    """一次 stop 请求的即时判定。"""

    accepted: bool
    reason: str


@dataclass(frozen=True)
class CancelResult:
    """Stop API 对外保持的结构化结果。"""

    generation_id: str
    cancelled: bool
    status: str
    reason: str


@dataclass(frozen=True)
class ChatRunStatusSnapshot:
    """通过 scoped control plane 暴露的 Chat Run 状态。"""

    generation_id: str
    phase: str
    status: str
    reason: str | None


@dataclass
class ChatGenerationRun:
    """一次 Chat Run 的阶段引用与终态事实。

    ``generation_id`` 只是 ``interaction_id`` 的只读兼容投影，进程内不再保存
    第二份生成事实；registry 的稳定键即为 ``interaction_id``。
    """

    identity_scope: IdentityScope
    interaction_id: str
    phase: ChatRunPhase = ChatRunPhase.CREATED
    outcome: ChatRunOutcome = ChatRunOutcome.RUNNING
    stop_reason: str | None = None
    active_task: asyncio.Task[object] | None = None

    @property
    def generation_id(self) -> str:
        """旧 Chat 控制 API 的兼容句柄，只读派生自 interaction_id。"""
        return self.interaction_id

    @property
    def scope_fingerprint(self) -> str:
        """保存注册时冻结的完整身份作用域指纹。"""
        return self.identity_scope.scope_fingerprint

    def bind_phase(self, phase: ChatRunPhase, task: asyncio.Task[object]) -> None:
        """绑定当前可被 stop 中断的阶段 task。"""
        self.phase = phase
        self.active_task = task

    def unbind_phase(self, task: asyncio.Task[object]) -> None:
        """仅按 task 身份解绑，避免旧 task 清空新阶段引用。"""
        if self.active_task is task:
            self.active_task = None

    def enter_phase(self, phase: ChatRunPhase) -> None:
        """进入没有可中断 task 的阶段或阶段交接窗口。"""
        self.phase = phase
        self.active_task = None

    def try_enter_finalizing(self) -> bool:
        """同步进入 finalize；已接受 stop 时拒绝进入。"""
        if self.outcome in {ChatRunOutcome.STOP_REQUESTED, ChatRunOutcome.CANCELLED}:
            return False
        if self.phase is ChatRunPhase.TERMINAL:
            return False
        self.phase = ChatRunPhase.FINALIZE
        self.active_task = None
        return True

    def mark_cancelled(self) -> None:
        """记录 Chat-level cancelled 终态。"""
        self.outcome = ChatRunOutcome.CANCELLED
        self.phase = ChatRunPhase.TERMINAL
        self.active_task = None

    def mark_completed(self) -> None:
        """记录 Chat-level completed 终态。"""
        self.outcome = ChatRunOutcome.COMPLETED
        self.phase = ChatRunPhase.TERMINAL
        self.active_task = None

    def mark_failed(self) -> None:
        """记录 Chat-level failed 终态。"""
        self.outcome = ChatRunOutcome.FAILED
        self.phase = ChatRunPhase.TERMINAL
        self.active_task = None

    def request_stop(self, reason: str = "user_requested") -> StopResult:
        """同步记录 stop，并取消当前唯一的可中断阶段 task。"""
        if self.outcome in {ChatRunOutcome.STOP_REQUESTED, ChatRunOutcome.CANCELLED}:
            return StopResult(
                accepted=True,
                reason=self.stop_reason or reason,
            )

        if self.phase in {ChatRunPhase.FINALIZE, ChatRunPhase.TERMINAL}:
            return StopResult(
                accepted=False,
                reason=(
                    "already_finalizing"
                    if self.phase is ChatRunPhase.FINALIZE
                    else "already_terminal"
                ),
            )

        self.outcome = ChatRunOutcome.STOP_REQUESTED
        self.stop_reason = reason

        task = self.active_task
        if task is not None and not task.done():
            task.cancel()

        return StopResult(
            accepted=True,
            reason=reason,
        )


class ChatGenerationRunRegistry:
    """进程内 Chat Run 注册表与 stop API 控制面。"""

    def __init__(self) -> None:
        self._runs: dict[str, ChatGenerationRun] = {}

    def register(self, run: ChatGenerationRun) -> None:
        require_workspace_access_context(run.identity_scope)
        existing = self._runs.get(run.interaction_id)
        if existing is not None:
            raise WorkspaceDomainError(
                "interaction_id 已被注册，拒绝覆盖现有 Chat Run",
                details={"interaction_id": run.interaction_id},
            )
        self._runs[run.interaction_id] = run

    def get(
        self,
        generation_id: str,
        identity_scope: IdentityScope,
    ) -> ChatGenerationRun | None:
        """只向同 owner/workspace 的控制请求暴露 run。"""
        identity_scope = require_workspace_access_context(identity_scope)
        run = self._runs.get(generation_id)
        if run is None or not self._same_resource_scope(run.identity_scope, identity_scope):
            return None
        return run

    def cancel(
        self,
        generation_id: str,
        identity_scope: IdentityScope,
        reason: str = "user_requested",
    ) -> CancelResult:
        run = self.get(generation_id, identity_scope)
        if run is None:
            return CancelResult(
                generation_id=generation_id,
                cancelled=False,
                status="not_found",
                reason=reason,
            )

        result = run.request_stop(reason)
        return CancelResult(
            generation_id=generation_id,
            cancelled=result.accepted,
            status=run.outcome.value,
            reason=result.reason,
        )

    def status(
        self,
        generation_id: str,
        identity_scope: IdentityScope,
    ) -> ChatRunStatusSnapshot | None:
        """查询 scoped 状态；跨 scope 与不存在统一返回 ``None``。"""
        run = self.get(generation_id, identity_scope)
        if run is None:
            return None
        return ChatRunStatusSnapshot(
            generation_id=run.generation_id,
            phase=run.phase.value,
            status=run.outcome.value,
            reason=run.stop_reason,
        )

    def close(self, run: ChatGenerationRun) -> None:
        """移除已由 Chat application 记录终态的 run。"""
        if self._runs.get(run.interaction_id) is run:
            self._runs.pop(run.interaction_id, None)

    @staticmethod
    def _same_resource_scope(
        registered: IdentityScope,
        requested: IdentityScope,
    ) -> bool:
        return registered.workspace_identity == requested.workspace_identity


__all__ = [
    "CancelResult",
    "ChatGenerationRun",
    "ChatGenerationRunRegistry",
    "ChatRunStatusSnapshot",
    "ChatRunOutcome",
    "ChatRunPhase",
    "StopResult",
]
