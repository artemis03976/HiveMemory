from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from hivememory.core.models import Identity
from hivememory.system.application.passive_message_ingressor import PassiveMessageIngressor
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime import MaintenanceTaskSpec, SystemAsyncScheduler

if TYPE_CHECKING:
    from hivememory.patchouli.config import HiveMemoryConfig
    from hivememory.patchouli.passive_ingest import PassiveIngressEvent
    from hivememory.system.runtime.global_bus import GlobalSystemBus


class PassiveIngressService:
    """顶层被动接入应用服务 — 持有独立的被动消息编排器。"""

    def __init__(
        self,
        bus: GlobalSystemBus,
        config: HiveMemoryConfig,
    ) -> None:
        self._bus = bus
        self._config = config
        self._ingressor = PassiveMessageIngressor(bus=bus)
        self._scheduler_started = False
        sched_config = config.scheduler
        self._scheduler = SystemAsyncScheduler(
            tick_seconds=sched_config.tick_seconds,
            shutdown_wait_seconds=sched_config.shutdown_wait_seconds,
        )
        self._setup_maintenance_tasks()

    def _setup_maintenance_tasks(self) -> None:
        tasks_config = self._config.scheduler.tasks
        self._ingressor.configure_idle_flush(
            timeout_seconds=tasks_config.observer_idle_flush_timeout_seconds,
            on_flush_callback=self._observer_idle_flush_callback,
        )
        self._scheduler.register(
            MaintenanceTaskSpec(
                name="observer_idle_flush",
                interval_seconds=tasks_config.observer_idle_flush_interval_seconds,
                enabled=tasks_config.enable_observer_idle_flush,
            ),
            self._ingressor.scan_idle_sessions_once,
        )

    async def _observer_idle_flush_callback(
        self,
        payload,
        target_topic=None,
    ) -> None:
        await self._submit_interaction(payload, target_topic=target_topic)

    async def _submit_interaction(
        self,
        payload,
        target_topic: Optional[str] = None,
    ) -> None:
        await self._bus.request(
            GlobalRoutes.PATCHOULI_SUBMIT_INTERACTION,
            payload=payload,
            target_topic=target_topic or "NEW_TOPIC",
        )

    async def start(self) -> None:
        if self._scheduler_started or not self._config.scheduler.enabled:
            return
        self._scheduler.start()
        self._scheduler_started = True

    async def stop(self) -> None:
        if self._scheduler_started:
            await self._scheduler.stop()
            self._scheduler_started = False

    async def shutdown_drain(self) -> Dict[str, Any]:
        await self.stop()
        flushed_rounds = self._ingressor.flush_all_pending_sessions()
        for payload, target_topic in flushed_rounds:
            await self._submit_interaction(payload, target_topic=target_topic)
        return {
            "success": True,
            "observer_payloads_submitted": len(flushed_rounds),
        }

    async def ingest_event(
        self,
        event: PassiveIngressEvent,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        identity = Identity(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        outcome = await self._ingressor.route_event(
            event=event,
            identity=identity,
        )

        if outcome.flushed:
            flushed_payload, flushed_target_topic = outcome.flushed
            await self._submit_interaction(
                flushed_payload,
                target_topic=flushed_target_topic,
            )

        if outcome.kind == "user" and outcome.hot_result is not None:
            hot_result = outcome.hot_result
            return {
                "intent": hot_result.intent,
                "rewritten": hot_result.rewritten,
                "keywords": hot_result.keywords,
                "worth_saving": hot_result.worth_saving,
                "memory": hot_result.rendered_memory_context,
            }

        if outcome.kind == "buffered":
            return {
                "intent": "buffered",
                "rewritten": None,
                "keywords": [],
                "worth_saving": True,
                "memory": None,
            }

        return {
            "intent": "ignored",
            "rewritten": None,
            "keywords": [],
            "worth_saving": False,
            "memory": None,
        }

    async def flush_observer_session(
        self,
        user_id: str,
        agent_id: str = "omni_doll",
        session_id: Optional[str] = None,
    ) -> bool:
        identity = Identity(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        flushed = self._ingressor.flush_session(identity)
        if not flushed:
            return False
        payload, target_topic = flushed
        await self._submit_interaction(payload, target_topic=target_topic)
        return True
