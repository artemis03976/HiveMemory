"""一轮外部交互的话题路由：root 交互 API 与 root 话题 API 共用。

外部 harness 的一轮在 HiveMemory 没有任务进程，它的话题由系统以 Gateway 的
PASSIVE_MEMORY 模式路由。路由只是 root 方法内部的一步，调用方已在方法入口
完成 root 授权，这里沿用同一个 scope，不另行授权。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hivememory.core.protocol.gateway import GatewayIngressMode, GatewaySessionTurn
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.models import IdentityScope
    from hivememory.core.protocol.gateway import GatewayDecision


@dataclass(frozen=True)
class PassiveTurnRouting:
    """一轮待路由的外部交互：用户消息与最近几轮。

    最近一轮所在的话题供话题路由失败时续接：已知时直接写在最后一轮的
    ``topic_id`` 上；不知道而最近一轮有交互标识时给出
    ``previous_interaction_id``，路由前查它的进度取得应用后实际所在的话题。
    """

    message: str
    session_turns: tuple[GatewaySessionTurn, ...] = ()
    previous_interaction_id: str | None = None


async def route_passive_turn(
    global_bus: GlobalSystemBus,
    scope: IdentityScope,
    routing: PassiveTurnRouting,
    *,
    request_timeout_ms: int,
) -> GatewayDecision:
    """以 PASSIVE_MEMORY 模式取得这一轮的话题路由与新话题的标题和摘要。

    Gateway 的异常原样传出，是否可恢复由调用方判断；PASSIVE_MEMORY 返回
    命令结果属于契约错误。
    """
    session_turns = routing.session_turns
    if routing.previous_interaction_id is not None and session_turns:
        previous = await global_bus.request(
            PatchouliRoutes.INTERACTION_STATUS,
            identity_scope=scope,
            interaction_id=routing.previous_interaction_id,
        )
        if previous is not None and previous.topic_id is not None:
            last = session_turns[-1].model_copy(update={"topic_id": previous.topic_id})
            session_turns = (*session_turns[:-1], last)
    result = await global_bus.request(
        GatewayPublicRoutes.PROCESS,
        message=routing.message,
        identity_scope=scope,
        ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
        request_timeout_ms=request_timeout_ms,
        session_turns=session_turns,
    )
    if result.kind != "decision":
        raise RuntimeError("PASSIVE_MEMORY 不得返回命令结果")
    return result.decision


__all__ = ["PassiveTurnRouting", "route_passive_turn"]
