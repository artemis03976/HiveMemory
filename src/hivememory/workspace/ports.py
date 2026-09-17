"""Workspace 资源平面端口：访问准入、资源读取、领域提交/结果、派生失效。

端口协议按父计划 5.7.1 冻结的四类责任区分，不得合并成任意 action
envelope；参数类型不得出现 ``PendingAtom``、Alice frame、MTP parser 或
外部 wire model。所有端口方法只接受经 admission 签发的
``WorkspaceAccessContext``，拒绝裸 ``IdentityScope``。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.workspace.access import (
    CallerPrincipal,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.workspace.projections import (
    CanonicalResourceChange,
    DomainHandle,
    DomainResult,
    DomainSubmission,
    InteractionApplyRequest,
    InteractionApplyResult,
    MemoryIntentRequest,
    MemorySnapshot,
    ProfileSnapshot,
)


@runtime_checkable
class WorkspaceAdmissionPort(Protocol):
    """admission 边界：把受信 principal + 待验证 claim 换成访问上下文。"""

    async def admit(
        self,
        principal: CallerPrincipal,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        operation: WorkspaceOperation,
    ) -> WorkspaceAccessContext:
        """校验 principal/Actor/Workspace 映射与 operation 授权后签发上下文。"""
        ...


@runtime_checkable
class WorkspaceResourcePort(Protocol):
    """正式资源读取端口（scope-aware，含缓存与授权重验，WRX-2/3 落地）。"""

    async def read_memory(
        self,
        access: WorkspaceAccessContext,
        memory_id: str,
    ) -> MemorySnapshot:
        """按 canonical UUID 读取当前授权下的 Memory 快照。"""
        ...

    async def read_memory_by_alias(
        self,
        access: WorkspaceAccessContext,
        alias: str,
    ) -> MemorySnapshot:
        """按 Workspace 分区 alias 读取当前授权下的 Memory 快照。"""
        ...

    async def search_memory(
        self,
        access: WorkspaceAccessContext,
        query: str,
        *,
        top_k: int = 8,
    ) -> tuple[MemorySnapshot, ...]:
        """语义检索当前 Workspace 内对当前 Actor 可见的 Memory 快照。"""
        ...

    async def read_profile(
        self,
        access: WorkspaceAccessContext,
        agent_alias: str | None,
    ) -> ProfileSnapshot:
        """读取当前授权下的 Agent Profile 快照（Root/CALL 共用入口）。"""
        ...

    async def acquire_asset(
        self,
        access: WorkspaceAccessContext,
        asset_ref: str,
    ) -> object:
        """获取 READY representation 的读取 lease（Asset facade 于 WRX-2 落地）。"""
        ...


@runtime_checkable
class DomainMutationPort(Protocol):
    """领域提交端口：把明确意图交给 Patchouli，由 Patchouli 决定结果。"""

    async def apply_interaction(self, request: InteractionApplyRequest) -> InteractionApplyResult:
        """提交已发生的交互载荷（幂等键 interaction_id）。"""
        ...

    async def submit_memory_intent(self, request: MemoryIntentRequest) -> DomainSubmission:
        """提交记忆意图；Patchouli 决定生成、更新、合并或拒绝。"""
        ...


@runtime_checkable
class DomainResultPort(Protocol):
    """领域结果端口：只投影真实领域结果，按归属信息做跨 scope 授权。"""

    async def get_submission_result(
        self,
        access: WorkspaceAccessContext,
        handle: DomainHandle,
    ) -> DomainResult:
        """查询提交结果的当前投影；越权/跨 scope 查询明确拒绝。"""
        ...


@runtime_checkable
class ResourceInvalidationPort(Protocol):
    """派生 cache 失效端口：canonical mutation 提交路径同步调用。

    正确性边界（父计划 6.2 节）：失效必须在 mutation 提交的同一同步边界
    完成；进程内事件只作观测，投递失败不得影响 cache 正确性。canonical
    mutation 的内部失效通知不做 Actor admission。
    """

    async def invalidate(self, change: CanonicalResourceChange) -> None:
        """使指定 canonical 变更涉及的派生视图失效。"""
        ...


__all__ = [
    "WorkspaceAdmissionPort",
    "WorkspaceResourcePort",
    "DomainMutationPort",
    "DomainResultPort",
    "ResourceInvalidationPort",
]
