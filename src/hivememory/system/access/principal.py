"""System 侧 Principal authentication：按接入登记确认调用来源。

实现 ``core.access.PrincipalAuthenticator`` 端口，供 workspace 认证入口
（``workspace.authentication.ActorAuthenticationGateway``）在 Workspace 准入前
调用。接入登记（``system.access.registry``）属于安装级配置，由 System 持有；
本类只回答"哪个已登记的调用来源在发起请求、能否服务该 Actor"，不授予任何
Workspace operation。
"""

from __future__ import annotations

from hivememory.core.access import CallerPrincipal
from hivememory.core.errors import AdmissionDeniedError
from hivememory.core.models import ActorIdentity
from hivememory.system.access.registry import SystemActorAccessRegistry

__all__ = [
    "SystemPrincipalAuthenticator",
]


class SystemPrincipalAuthenticator:
    """按 System 接入登记完成 Principal authentication。

    拒绝语义（均为 ``AdmissionDeniedError``，以稳定 reason 区分）：

    - 接入未登记/禁用（不区分，避免泄漏配置）→ ``unknown_principal``；
    - adapter 不匹配 → ``adapter_mismatch``；
    - principal 身份解析规则不允许该用户 → ``actor_not_allowed_for_principal``。
    """

    def __init__(self, registry: SystemActorAccessRegistry) -> None:
        self._system_registry = registry

    def authenticate_principal(
        self,
        *,
        adapter: str,
        principal: CallerPrincipal,
        actor: ActorIdentity,
    ) -> None:
        """确认调用来源已登记、adapter 匹配且可服务该 Actor 用户。"""
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
        if entry.allowed_user_ids is not None and actor.user_id not in entry.allowed_user_ids:
            raise AdmissionDeniedError(
                message="接入登记的身份解析规则不允许该 Actor 用户",
                details={
                    "principal_id": principal.principal_id,
                    "reason": "actor_not_allowed_for_principal",
                },
            )
