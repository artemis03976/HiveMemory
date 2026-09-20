"""System 统一 Actor Authentication 网关：唯一对外认证入口。

A1 计划（docs/plans/v0.7.0-a1-workspace-access-boundary.md 第 3/3.1 节）
确立的认证编排：所有调用侧只调用一个网关，输入受信接入信息、待解析的
Actor 身份和目标 Workspace；网关内部顺序完成两项认证，全部成功后返回
一个可在有效区间内复用的 ``WorkspaceAccessContext``。

    1. Principal authentication —— 匹配 System 接入登记与 adapter，
       确认 CallerPrincipal 和 ActorIdentity；
    2. Workspace authentication / admission —— 核对 Workspace Actor
       准入记录（含 W0 owner 约束），取得 allowed_operations 的可信关联。

认证入口不要求提供待执行的 operation，也不向调用方返回"只完成第一项
认证"的可访问上下文。网关统一流程不改变数据所有权：它查询 System 接入
登记和 Workspace 访问登记（后者由 ``workspace.registry`` 持有），但不
执行 search/read/submit，不接受任意 action 代执行业务，也不替代全局
总线路由——认证成功后，调用侧沿既有 adapter/service/bridge 发起业务调用。

阶段拒绝语义（A1 第 3.4 节，均以稳定 reason 区分）：

- 接入未登记/禁用（不区分，避免泄漏配置）→ ``unknown_principal``；
- adapter 不匹配 → ``adapter_mismatch``；
- principal 身份解析规则不允许该用户 → ``actor_not_allowed_for_principal``；
- actor user ≠ workspace owner（W0 兼容基线）→ ``actor_not_owner``；
- Workspace 无有效 Actor 访问记录 → ``actor_not_admitted``。

以上均为 ``AdmissionDeniedError``（第一、二层失败）；缺少行为许可发生在
后续每次动作的共享行为检查（``workspace.access.WorkspaceAccessGuard``），
属 ``OperationDeniedError``，不在本网关判断。
"""

from __future__ import annotations

import time
from typing import Callable

from hivememory.core.errors import AdmissionDeniedError, OwnerMismatchError
from hivememory.core.models import ActorIdentity, IdentityScope, WorkspaceIdentity
from hivememory.workspace.registry import WorkspaceActorAccessRegistry

from hivememory.system.access.credentials import (
    AccessContextValidity,
    issue_access_context,
)
from hivememory.system.access.principal import CallerPrincipal
from hivememory.system.access.registry import SystemActorAccessRegistry

__all__ = [
    "ActorAuthenticationGateway",
]

_DEFAULT_ISSUED_BY = "system-actor-authentication"


class ActorAuthenticationGateway:
    """统一 Actor Authentication 网关（System 接入层）。

    一次 :meth:`authenticate` 调用完成 Principal authentication 与
    Workspace authentication；两项均通过才签发 ``WorkspaceAccessContext``。
    网关不执行业务、不转发路由；adapter 负责协议解析与接入证据，网关
    负责按登记规则统一验证并作出认证结论。

    生命周期（A1 第 3.4 节）：context 仅在本网关（其所在运行实例）内
    复用；``context_ttl_seconds`` 声明认证有效区间上限，``None`` 表示
    不设固定 TTL、随网关关闭一并失效；:meth:`close` 后旧 context 一律
    拒绝使用，调用侧必须重新认证。本类不提供配置热更新——首版本地
    配置在运行实例内不可变，修改经重启生效。
    """

    def __init__(
        self,
        *,
        system_registry: SystemActorAccessRegistry,
        workspace_registry: WorkspaceActorAccessRegistry,
        issued_by: str = _DEFAULT_ISSUED_BY,
        context_ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if context_ttl_seconds is not None and context_ttl_seconds <= 0:
            raise ValueError("context_ttl_seconds 必须为正数或 None")
        self._system_registry = system_registry
        self._workspace_registry = workspace_registry
        self._issued_by = issued_by
        self._context_ttl_seconds = context_ttl_seconds
        self._clock = clock
        # 有效期锚点随签发写入每份 grant；close() 使其签发的全部 context
        # 一并失效（``workspace.access.AccessContextValidity``）。
        self._validity = AccessContextValidity()

    @property
    def is_closed(self) -> bool:
        """网关是否已关闭；关闭后不再签发 context，旧 context 一并失效。"""
        return self._validity.closed

    def close(self) -> None:
        """关闭网关：已签发 context 随有效期锚点一并失效（A1 第 3.4 节）。"""
        self._validity.close()

    async def authenticate(
        self,
        *,
        adapter: str,
        principal: CallerPrincipal,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
    ) -> WorkspaceAccessContext:
        """完成两项认证并签发访问上下文；任一失败即拒绝，无部分成功。

        ``adapter`` 是调用来源实际使用的接入方式标识；``principal`` 由
        受信 adapter 依据其接入证据构造。认证成功返回的 context 与单次
        operation 解耦，可在此后各次获准动作中复用。
        """
        if self._validity.closed:
            raise AdmissionDeniedError(
                message="认证网关已关闭，拒绝新的认证请求",
                details={"reason": "authentication_gateway_closed"},
            )
        if not isinstance(adapter, str) or not adapter.strip():
            raise TypeError("adapter 必须是非空字符串")
        if not isinstance(principal, CallerPrincipal):
            raise TypeError("principal 必须是 CallerPrincipal")
        if not isinstance(actor, ActorIdentity):
            raise TypeError("actor 必须是 ActorIdentity")
        if not isinstance(workspace, WorkspaceIdentity):
            raise TypeError("workspace 必须是 WorkspaceIdentity")

        # ---- 1. Principal authentication：System 接入登记与 adapter 匹配 ----
        entry = self._system_registry.entry_for(principal.principal_id)
        # 未登记与已禁用统一按未知 principal 拒绝，不区分，避免泄漏配置。
        if entry is None or not entry.enabled:
            raise AdmissionDeniedError(
                message="调用来源未获准接入",
                details={
                    "principal_id": principal.principal_id,
                    "reason": "unknown_principal",
                },
            )
        if adapter not in entry.adapters:
            raise AdmissionDeniedError(
                message="调用来源与接入方式不匹配",
                details={
                    "principal_id": principal.principal_id,
                    "reason": "adapter_mismatch",
                },
            )
        if (
            entry.allowed_user_ids is not None
            and actor.user_id not in entry.allowed_user_ids
        ):
            raise AdmissionDeniedError(
                message="接入登记的身份解析规则不允许该 Actor 用户",
                details={
                    "principal_id": principal.principal_id,
                    "reason": "actor_not_allowed_for_principal",
                },
            )

        # ---- 2. Workspace authentication / admission ----
        try:
            identity_scope = IdentityScope(
                actor_identity=actor,
                workspace_identity=workspace,
            )
        except OwnerMismatchError as exc:
            # W0 兼容基线：actor user ≠ workspace owner 时按 admission 拒绝，
            # 而不是把矛盾坐标放行到资源层。
            raise AdmissionDeniedError(
                message="actor 与 workspace owner 不一致，admission 拒绝",
                details={
                    "principal_id": principal.principal_id,
                    "actor_user_id": actor.user_id,
                    "owner_user_id": workspace.owner_user_id,
                    "reason": "actor_not_owner",
                },
            ) from exc

        access_record = self._workspace_registry.record_for(workspace, actor)
        if access_record is None or not access_record.enabled:
            # 相同 owner 也不表示自动获准进入；缺失或禁用的访问记录是
            # Workspace 准入失败，不是身份认证失败。
            raise AdmissionDeniedError(
                message="该 Actor 在目标 Workspace 没有有效的访问登记",
                details={
                    "principal_id": principal.principal_id,
                    "reason": "actor_not_admitted",
                },
            )

        expires_at = (
            self._clock() + self._context_ttl_seconds
            if self._context_ttl_seconds is not None
            else None
        )
        return issue_access_context(
            principal,
            identity_scope,
            access_record=access_record,
            validity=self._validity,
            issued_by=self._issued_by,
            expires_at=expires_at,
        )
