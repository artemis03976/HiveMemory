"""执行者的操作请求、执行凭据与单一操作入口契约。

请求只描述操作参数，不携带身份或目标 Workspace；CPU 用独立的执行凭据
绑定提交函数，MTP 与 CALL 适配器只消费这个函数。子线程暂时沿用主线程
的提交函数，凭据的签发、兑现与吊销始终由 workspace 管理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, Protocol

from hivememory.core.models.agent import AgentProfile
from hivememory.core.models.memory import MemoryAtom
from hivememory.core.models.pending import PendingAtom, WriteFocus
from hivememory.core.models.query import QueryFilters
from hivememory.core.models.reference import ReferenceResolution


class ExecutionCredentialRevokedError(RuntimeError):
    """执行凭据未知或已吊销，操作入口拒绝继续分派。"""


class ExecutionCredential:
    """按对象身份兑现的不透明凭据，不保存或序列化身份数据。"""

    __slots__ = ()

    def __reduce__(self) -> NoReturn:
        """凭据仅供进程内传递，不能复制为可持久化的身份载体。"""
        raise TypeError("Execution credentials cannot be serialized")


@dataclass(frozen=True)
class OperationRequest[R]:
    """操作请求的结果类型契约，具体请求与能力方法一一对应。"""


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
    """撤回当前进程仍为 PENDING 的指定意图。"""

    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ResolveReferencesRequest(OperationRequest[list[ReferenceResolution]]):
    """按请求顺序解析引用，每个引用均获得中立解析结果。"""

    aliases: tuple[str, ...]


@dataclass(frozen=True)
class RetrieveRequest(OperationRequest[list[MemoryAtom]]):
    """检索可见记忆，过滤与数量参数由能力层映射到 canonical 读取。"""

    semantic_query: str
    keywords: tuple[str, ...] = ()
    top_k: int = 5
    filters: QueryFilters | None = None


@dataclass(frozen=True)
class GetAgentProfileRequest(OperationRequest[AgentProfile]):
    """读取 Agent Profile，内置与自定义图纸均经过 profile.read 授权。"""

    agent_alias: str | None = None


class OperationEntry(Protocol):
    """workspace 的单一操作入口，凭据只在入口兑现为访问 context。"""

    async def execute[R](
        self, request: OperationRequest[R], *, credential: ExecutionCredential
    ) -> R: ...


class OperationSubmitter(Protocol):
    """CPU 绑定凭据后的提交函数，适配器只提交操作参数。"""

    async def __call__[R](self, request: OperationRequest[R]) -> R: ...


__all__ = [
    "CancelIntentsRequest",
    "ExecutionCredential",
    "ExecutionCredentialRevokedError",
    "GetAgentProfileRequest",
    "OperationEntry",
    "OperationRequest",
    "OperationSubmitter",
    "ResolveReferencesRequest",
    "RetrieveRequest",
    "SubmitUpdateIntentRequest",
    "SubmitWriteIntentRequest",
]
