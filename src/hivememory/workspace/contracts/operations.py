"""执行者的操作请求、线程句柄与单一操作入口契约。

请求只描述操作参数，不携带身份或目标 Workspace；CPU 驱动以执行线程绑定
提交函数，MTP 与 CALL 适配器只消费这个函数。CALL 的派生与子线程的结束是
线程的生命周期动作，经入口的专用方法提交；context 的派生、检查与撤销始终
由 workspace 管理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.errors import ThreadEndedError
from hivememory.core.models.memory import MemoryAtom
from hivememory.core.models.pending import PendingAtom, WriteFocus
from hivememory.core.models.query import QueryFilters
from hivememory.core.models.reference import ReferenceResolution
from hivememory.workspace.contracts.threads import ExecutionThread, ThreadDescriptor


@dataclass(frozen=True)
class OperationRequest[R]:
    """请求的结果类型契约：actor 请求一一对应能力方法。"""


@dataclass(frozen=True)
class SubmitWriteIntentRequest(OperationRequest[PendingAtom]):
    """提交 WRITE，生成队列接纳后返回 workspace 已登记的意图作为 ACK。"""

    focus: WriteFocus


@dataclass(frozen=True)
class SubmitUpdateIntentRequest(OperationRequest[PendingAtom]):
    """提交 UPDATE，由能力层解析基础引用并验证其可更新性。"""

    base_alias: str
    instruction: str
    content: str | None = None


@dataclass(frozen=True)
class ResolveReferencesRequest(OperationRequest[list[ReferenceResolution]]):
    """READ：按请求顺序解析引用，每个引用均获得中立解析结果。"""

    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ResolveRunTargetRequest(OperationRequest[ReferenceResolution]):
    """RUN：解析单个执行目标，独立检查 memory.run 并记录正式引用。"""

    alias: str


@dataclass(frozen=True)
class RetrieveRequest(OperationRequest[list[MemoryAtom]]):
    """检索可见记忆，过滤与数量参数由能力层映射到 canonical 读取。"""

    semantic_query: str
    keywords: tuple[str, ...] = ()
    # 缺省时由检索引擎配置的默认条数决定。
    top_k: int | None = None
    filters: QueryFilters | None = None


@dataclass(frozen=True)
class DeriveCallThreadRequest:
    """CALL 的派生请求：以调用方可见的图纸确定被调用方，派生一个子线程。

    不是 actor 操作请求：入口的 ``execute`` 只接受五项记忆请求，派生只经
    ``derive_child_thread`` 提交。``agent.call`` 授权与图纸解析是其中的
    actor 操作，派生本身是线程的生命周期动作。
    """

    agent_alias: str | None = None


class OperationEntry(Protocol):
    """workspace 的单一操作入口：只接受存活线程的 context。

    ``execute`` 分派五项记忆请求；派生与结束子线程是线程的生命周期动作，
    由两个专用方法提交，结果中的 context 只交给 CPU 驱动。
    """

    async def execute[R](
        self, request: OperationRequest[R], *, access: WorkspaceAccessContext
    ) -> R: ...

    async def derive_child_thread(
        self, request: DeriveCallThreadRequest, *, access: WorkspaceAccessContext
    ) -> ExecutionThread:
        """以调用方的线程授权 ``agent.call`` 并解析图纸，再派生子线程。"""
        ...

    def end_thread(self, *, access: WorkspaceAccessContext) -> None:
        """同步结束一个子线程；已结束的线程是空操作，主线程以接线缺陷拒绝。"""
        ...


class OperationSubmitter(Protocol):
    """CPU 侧唯一的线程句柄：线程描述、提交函数与子线程的派生和结束。

    执行适配器、frame 与 CALL 编排只持有句柄，读不出 context；帧持有所属
    线程的句柄，帧与线程的对应关系由此建立。
    """

    @property
    def thread(self) -> ThreadDescriptor:
        """本线程的描述：标识、观测标签、图纸与操作快照。"""
        ...

    async def __call__[R](self, request: OperationRequest[R]) -> R:
        """以本线程提交一项记忆请求。"""
        ...

    async def derive_child(self, request: DeriveCallThreadRequest) -> OperationSubmitter:
        """以本线程提交 CALL 派生，返回子线程的句柄。"""
        ...

    def end(self) -> None:
        """同步结束本线程；可以在取消收尾中调用，不引入新的挂起点。

        只用于子线程：主线程由进程在 Actor 阶段收尾时结束，CPU 结束主线程
        以接线缺陷拒绝。
        """
        ...


class ThreadOperationSubmitter:
    """CPU 驱动以执行线程绑定的线程句柄，供任何 CPU 驱动复用。

    句柄只交出线程描述，context 留在句柄内部；派生结果中的子线程 context
    在这里绑定成子线程的句柄后才交出。
    """

    def __init__(self, entry: OperationEntry, thread: ExecutionThread) -> None:
        self._entry = entry
        self._descriptor = thread.descriptor
        self._access = thread.access

    @property
    def thread(self) -> ThreadDescriptor:
        """本线程的描述，不含 context。"""
        return self._descriptor

    async def __call__[R](self, request: OperationRequest[R]) -> R:
        """记忆请求透传到入口，由入口判定线程存活并分派。"""
        return await self._entry.execute(request, access=self._access)

    async def derive_child(self, request: DeriveCallThreadRequest) -> ThreadOperationSubmitter:
        """以本线程提交 CALL 派生，把子线程绑定成它自己的句柄。"""
        child = await self._entry.derive_child_thread(request, access=self._access)
        return ThreadOperationSubmitter(self._entry, child)

    def end(self) -> None:
        """同步结束本线程；重复结束是空操作。"""
        self._entry.end_thread(access=self._access)


__all__ = [
    "ThreadEndedError",
    "OperationEntry",
    "OperationRequest",
    "OperationSubmitter",
    "DeriveCallThreadRequest",
    "ResolveReferencesRequest",
    "ResolveRunTargetRequest",
    "RetrieveRequest",
    "SubmitUpdateIntentRequest",
    "SubmitWriteIntentRequest",
    "ThreadOperationSubmitter",
]
