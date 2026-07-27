from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hivememory.core.constants import DEFAULT_AGENT_ID
from hivememory.core.models import Identity
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.system.services.passive import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveMessageIngressor,
)
from hivememory.system.services.passive.models import (
    DEFAULT_EXTERNAL_CONVERSATION_ID,
    DEFAULT_PASSIVE_SOURCE,
)
from hivememory.system.services.passive.outbox import SealedTurn
from hivememory.system.config.memory_compiler import FullContextStrategyConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.events import RuntimeEventSink
from hivememory.system.runtime.scheduler.models import MaintenanceTaskSpec

if TYPE_CHECKING:
    from hivememory.system.config import HiveMemoryConfig
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
    from hivememory.system.runtime.scheduler.async_scheduler import (
        AsyncMaintenanceScheduler,
    )


class PassiveIngressService:
    """顶层被动接入应用服务 — 持有独立的被动消息编排器。"""

    _MAINTENANCE_OWNER = "system.passive_ingress"

    def __init__(
        self,
        bus: GlobalSystemBus,
        config: HiveMemoryConfig,
        scheduler: AsyncMaintenanceScheduler,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._bus = bus
        self._config = config
        self._scheduler = scheduler
        self._ingressor = PassiveMessageIngressor(
            bus=bus,
            submit_sealed_turn=self._submit_sealed_turn,
            gateway_request_timeout_ms=(
                config.gateway.workflow.default_request_timeout_ms
            ),
            config=config.passive_ingress,
            runtime_events=runtime_events,
        )
        self._maintenance_registered = False
        self._configure_idle_flush()

    @property
    def ingressor(self) -> PassiveMessageIngressor:
        return self._ingressor

    def _configure_idle_flush(self) -> None:
        tasks_config = self._config.scheduler.tasks
        self._ingressor.configure_idle_flush(
            timeout_seconds=tasks_config.observer_idle_flush_timeout_seconds,
        )

    def _register_maintenance_tasks(self) -> bool:
        if not self._config.scheduler.enabled:
            return False
        tasks_config = self._config.scheduler.tasks
        self._scheduler.register(
            MaintenanceTaskSpec(
                owner=self._MAINTENANCE_OWNER,
                name="observer_idle_flush",
                interval_seconds=tasks_config.observer_idle_flush_interval_seconds,
                enabled=tasks_config.enable_observer_idle_flush,
            ),
            self._ingressor.scan_idle_conversations_once,
        )
        return True

    def _unregister_maintenance_tasks(self) -> int:
        return self._scheduler.unregister_owner(self._MAINTENANCE_OWNER)

    async def _submit_sealed_turn(self, sealed: SealedTurn) -> str | None:
        """把 sealed turn 提交给 Patchouli。

        抛出异常即代表提交失败，由 Ingressor 保留 outbox item 供重试。

        Returns:
            Patchouli 落定的真实 topic_id，供观测事件关联。
        """
        return await self._bus.request(
            GlobalRoutes.PATCHOULI_SUBMIT_INTERACTION,
            payload=sealed.payload,
            target_topic=sealed.target_topic or "NEW_TOPIC",
        )

    async def start(self) -> None:
        if self._maintenance_registered:
            return
        self._maintenance_registered = self._register_maintenance_tasks()

    async def stop(self) -> None:
        if not self._maintenance_registered:
            return
        self._unregister_maintenance_tasks()
        self._maintenance_registered = False

    async def shutdown_drain(self) -> dict[str, Any]:
        await self.stop()
        result = await self._ingressor.shutdown_drain()
        return {
            "success": result["outbox_pending"] == 0,
            "observer_payloads_submitted": result["submitted_turns"],
            "observer_payloads_pending": result["outbox_pending"],
        }

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    async def ingest_event(
        self,
        event: PassiveIngressEvent,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """接收单个外部事件。

        公共响应只包含外部调用方实际需要的接收状态与 memory context，
        不暴露 Gateway 内部 execution state、runtime event 或 fallback 细节。

        `session_id` 是过渡期兼容入参：当事件未显式携带
        `external_conversation_id` 时，用它作为外部会话 ID。
        新调用方应直接在事件上设置 `source` 与 `external_conversation_id`。
        """
        if session_id and event.external_conversation_id == (
            DEFAULT_EXTERNAL_CONVERSATION_ID
        ):
            event = event.model_copy(
                update={"external_conversation_id": session_id}
            )

        identity = Identity(
            user_id=user_id,
            agent_id=agent_id,
            session_id=event.external_conversation_id,
        )
        outcome = await self._ingressor.route_event(event=event, identity=identity)

        if outcome.kind == "duplicate":
            return {
                "status": "duplicate",
                "external_event_id": event.external_event_id,
                "memory": None,
            }

        if outcome.kind == "user":
            return {
                "status": "accepted",
                "external_event_id": event.external_event_id,
                "memory": self._compile_memory_context(outcome.retrieval_result),
            }

        if outcome.kind == "buffered":
            return {
                "status": "buffered",
                "external_event_id": event.external_event_id,
                "memory": None,
            }

        return {
            "status": "ignored",
            "external_event_id": event.external_event_id,
            "memory": None,
        }

    def _compile_memory_context(self, retrieval_result) -> str | None:
        if retrieval_result is None or not retrieval_result.memories:
            return None
        return MemoryCompiler().compile(
            retrieval_result.memories,
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            MemoryCompileOptions(
                retrieval_strategy_config=FullContextStrategyConfig()
            ),
        ).text

    async def flush_conversation(
        self,
        source: str,
        external_conversation_id: str,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> bool:
        """显式 seal 并提交指定外部会话的当前 turn。

        Returns:
            True 表示至少有一个 sealed turn 被成功提交。
        """
        identity = Identity(
            user_id=user_id,
            agent_id=agent_id,
            session_id=external_conversation_id,
        )
        key = PassiveConversationKey.build(
            source=source,
            external_conversation_id=external_conversation_id,
            identity=identity,
        )
        submitted = await self._ingressor.flush_conversation(key, identity)
        return submitted > 0

    async def flush_ingressor(
        self,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        session_id: str | None = None,
        source: str = DEFAULT_PASSIVE_SOURCE,
    ) -> bool:
        """过渡期兼容入口：按 session_id 显式 flush。

        新调用方应改用 `flush_conversation(source, external_conversation_id, ...)`。
        """
        return await self.flush_conversation(
            source=source,
            external_conversation_id=session_id or DEFAULT_EXTERNAL_CONVERSATION_ID,
            user_id=user_id,
            agent_id=agent_id,
        )
