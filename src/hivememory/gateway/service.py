"""GatewayService：Gateway 子系统业务入口。"""

from __future__ import annotations

from hivememory.core.models import IdentityScope, require_identity_scope
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    GatewayProcessResult,
    GatewaySessionTurn,
)
from hivememory.gateway.runtime import GatewayRuntime


class GatewayService:
    """
    Gateway 子系统业务门面。

    调用方只通过 process 进入 Gateway Workflow。
    """

    def __init__(self, runtime: GatewayRuntime) -> None:
        self._runtime = runtime

    async def process(
        self,
        message: str,
        *,
        identity_scope: IdentityScope,
        ingress_mode: GatewayIngressMode,
        request_timeout_ms: int | None = None,
        session_turns: tuple[GatewaySessionTurn, ...] = (),
        designated_topic_id: str | None = None,
    ) -> GatewayProcessResult:
        """把一次 Gateway 请求完整委托给 Runtime 持有的 workflow。

        ``identity_scope`` 是调用方冻结的执行坐标：主动链路与 Import Bus 都由
        系统以 root 授权组装，Gateway 只使用其中的 Workspace。Gateway 不接收
        工作 actor，也不做授权判断。``designated_topic_id`` 是用户为这一轮指定的
        话题：在候选中时话题路由直接取用，不在候选中时这一轮以受控错误拒绝。
        """

        configured_timeout_ms = self._runtime.config.workflow.default_request_timeout_ms
        effective_timeout_ms = (
            configured_timeout_ms
            if request_timeout_ms is None
            else min(request_timeout_ms, configured_timeout_ms)
        )

        identity_scope = require_identity_scope(identity_scope)
        return await self._runtime.workflow.run(
            message,
            identity_scope=identity_scope,
            ingress_mode=ingress_mode,
            session_turns=session_turns,
            designated_topic_id=designated_topic_id,
            request_timeout_ms=effective_timeout_ms,
        )


__all__ = ["GatewayService"]
