"""被动接入的 memory context 准备逻辑。

把 Gateway 决策与 Patchouli retrieval 封装为单一协作对象，
可恢复失败在此收敛为 `MemoryContextAttempt(degraded=True)`，
不暴露给上层路由策略。

设计 §6：Gateway 或 retrieval 的可恢复失败不得阻止当前 user 进入
buffer；契约违约与装配缺陷不降级，直接向上抛出。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

from hivememory.core.models import WorkspaceAccessContext
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayIngressMode,
    RetrievalMode,
)
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.services.passive.events import PassiveIngressEventEmitter
from hivememory.system.services.passive.exceptions import (
    PassiveIngressContractError,
    is_recoverable_ingress_error,
)
from hivememory.system.services.passive.models import (
    PassiveConversationKey,
    PassiveIngressEvent,
)

logger = logging.getLogger(__name__)

MemoryContextStage = Literal["gateway", "retrieval"]


@dataclass(frozen=True)
class MemoryContextAttempt:
    """一次 memory context 准备的结果。

    `degraded=True` 表示 Gateway 或 retrieval 发生可恢复失败：本轮没有
    memory context，但 user 事件仍必须进入 buffer 并在 turn 完成后提交。
    """

    decision: GatewayDecision | None = None
    retrieval_result: RetrievalResponse | None = None
    failed_stage: MemoryContextStage | None = None
    error_class: str | None = None

    @property
    def degraded(self) -> bool:
        return self.failed_stage is not None

    @property
    def memory_ref_count(self) -> int:
        if self.retrieval_result is None:
            return 0
        return len(self.retrieval_result.memories)


class MemoryContextProvider:
    """请求 Gateway decision 与 Patchouli retrieval 并发布观测事件。

    调用 `prepare()` 取得本轮 `MemoryContextAttempt`；
    失败处理与事件发布集中在此，上层只消费结果对象。
    """

    def __init__(
        self,
        bus: GlobalSystemBus,
        *,
        gateway_request_timeout_ms: int = 8000,
        events: PassiveIngressEventEmitter,
    ) -> None:
        self._bus = bus
        self._gateway_request_timeout_ms = gateway_request_timeout_ms
        self._events = events

    async def prepare(
        self,
        event: PassiveIngressEvent,
        access_context: WorkspaceAccessContext,
        key: PassiveConversationKey,
    ) -> MemoryContextAttempt:
        """请求 Gateway decision 与 Patchouli retrieval，可恢复失败则降级。

        retrieval 失败时保留已获得的 decision，使最终 submission 仍能路由到
        正确的 topic，只是缺少 memory context。
        """
        started_at = time.perf_counter()
        attempt = await self._attempt(event.content, access_context, key)

        self._events.memory_context_prepared(
            key=key,
            external_event_id=event.external_event_id,
            turn_id=event.turn_id,
            duration_ms=(time.perf_counter() - started_at) * 1000,
            memory_ref_count=attempt.memory_ref_count,
            degraded=attempt.degraded,
            failed_stage=attempt.failed_stage,
            error_class=attempt.error_class,
            topic_id=(
                attempt.decision.target_topic_id if attempt.decision else None
            ),
            workspace_id=access_context.workspace_identity.workspace_id,
        )
        return attempt

    async def _attempt(
        self,
        content: str,
        access_context: WorkspaceAccessContext,
        key: PassiveConversationKey,
    ) -> MemoryContextAttempt:
        try:
            decision = await self._request_gateway_decision(content, access_context)
        except Exception as exc:
            if not is_recoverable_ingress_error(exc):
                raise
            logger.warning(
                "Passive Gateway 决策可恢复失败，降级为无 memory context: "
                "conversation=%s, error=%s",
                key.label,
                type(exc).__name__,
                exc_info=True,
            )
            return MemoryContextAttempt(
                failed_stage="gateway",
                error_class=type(exc).__name__,
            )

        try:
            retrieval_result = await self._retrieve_for_decision(decision, access_context)
        except Exception as exc:
            if not is_recoverable_ingress_error(exc):
                raise
            logger.warning(
                "Passive retrieval 可恢复失败，降级为无 memory context: "
                "conversation=%s, error=%s",
                key.label,
                type(exc).__name__,
                exc_info=True,
            )
            return MemoryContextAttempt(
                decision=decision,
                failed_stage="retrieval",
                error_class=type(exc).__name__,
            )

        return MemoryContextAttempt(decision=decision, retrieval_result=retrieval_result)

    async def _request_gateway_decision(
        self,
        content: str,
        access_context: WorkspaceAccessContext,
    ) -> GatewayDecision:
        gateway_result = await self._bus.request(
            GlobalRoutes.GATEWAY_PROCESS,
            message=content,
            access_context=access_context,
            ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
            request_timeout_ms=self._gateway_request_timeout_ms,
        )
        if gateway_result.kind != "decision":
            raise PassiveIngressContractError(
                "PASSIVE_MEMORY 不得返回 command outcome"
            )
        return gateway_result.decision

    async def _retrieve_for_decision(
        self,
        decision: GatewayDecision,
        access_context: WorkspaceAccessContext,
    ) -> RetrievalResponse:
        if (
            decision.retrieval_plan.mode == RetrievalMode.SKIP
            or decision.retrieval_plan.top_k == 0
        ):
            return RetrievalResponse()

        return await self._bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
            request=RetrievalRequest(
                semantic_query=decision.rewritten_query,
                keywords=list(decision.search_keywords),
                access_context=access_context,
                top_k=decision.retrieval_plan.top_k,
            ),
        )


__all__ = ["MemoryContextAttempt", "MemoryContextProvider", "MemoryContextStage"]
