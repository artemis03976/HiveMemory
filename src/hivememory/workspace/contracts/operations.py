"""执行者的操作请求、线程 context 与单一操作入口契约。

请求只描述操作参数，不携带身份或目标 Workspace；CPU 用独立的线程 context
绑定提交函数，MTP 与 CALL 适配器只消费这个函数。子线程暂时沿用主线程
的提交函数，线程 context 的派生、检查与撤销始终由 workspace 管理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.errors import ThreadContextRevokedError
from hivememory.core.models.agent import AgentProfile
from hivememory.core.models.memory import MemoryAtom
from hivememory.core.models.pending import PendingAtom, WriteFocus
from hivememory.core.models.query import QueryFilters
from hivememory.core.models.reference import ReferenceResolution


@dataclass(frozen=True)
class OperationRequest[R]:
    """请求的结果类型契约：actor 请求一一对应能力方法，补偿请求由入口处理。"""


@dataclass(frozen=True)
class SubmitWriteIntentRequest(OperationRequest[PendingAtom]):
    """提交 WRITE，返回 workspace 已登记的意图作为 ACK。"""

    focus: WriteFocus


@dataclass(frozen=True)
class SubmitUpdateIntentRequest(OperationRequest[PendingAtom]):
    """提交 UPDATE，由能力层解析基础引用并验证其可更新性。"""

    base_alias: str
    instruction: str
    content: str | None = None


@dataclass(frozen=True)
class CancelIntentsRequest(OperationRequest[list[str]]):
    """执行单元内部补偿：撤回当前进程仍为 PENDING 的指定意图。

    不属于 actor 操作；入口校验线程 context 后只按绑定的进程关联撤回，不检查
    actor 操作许可，也不能撤回其他进程或已经认领的意图。
    """

    aliases: tuple[str, ...]


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
    top_k: int = 5
    filters: QueryFilters | None = None


@dataclass(frozen=True)
class ResolveCallTargetRequest(OperationRequest[AgentProfile]):
    """CALL：读取目标 Profile，内置与自定义图纸均经过 agent.call 授权。"""

    agent_alias: str | None = None


class OperationEntry(Protocol):
    """workspace 的单一操作入口，只接受有效的线程 context。"""

    async def execute[R](
        self, request: OperationRequest[R], *, access: WorkspaceAccessContext
    ) -> R: ...


class OperationSubmitter(Protocol):
    """CPU 绑定线程 context 后的提交函数，适配器只提交操作参数。"""

    async def __call__[R](self, request: OperationRequest[R]) -> R: ...


__all__ = [
    "CancelIntentsRequest",
    "ThreadContextRevokedError",
    "OperationEntry",
    "OperationRequest",
    "OperationSubmitter",
    "ResolveCallTargetRequest",
    "ResolveReferencesRequest",
    "ResolveRunTargetRequest",
    "RetrieveRequest",
    "SubmitUpdateIntentRequest",
    "SubmitWriteIntentRequest",
]
