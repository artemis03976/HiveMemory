from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from hivememory.core.models import Identity
from hivememory.system.application.passive import (
    PassiveIngressEvent,
    PassiveMessageIngressor,
)
from hivememory.system.contracts.routes import GlobalRoutes
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
    _OBSERVER_IDLE_FLUSH_TASK = "observer_idle_flush"

    def __init__(
        self,
        bus: GlobalSystemBus,
        config: HiveMemoryConfig,
        scheduler: AsyncMaintenanceScheduler,
    ) -> None:
        self._bus = bus
        self._config = config
        self._scheduler = scheduler
        self._ingressor = PassiveMessageIngressor(bus=bus)
        self._maintenance_registered = False
        self._configure_idle_flush()

    def _configure_idle_flush(self) -> None:
        tasks_config = self._config.scheduler.tasks
        self._ingressor.configure_idle_flush(
            timeout_seconds=tasks_config.observer_idle_flush_timeout_seconds,
            on_flush_callback=self._observer_idle_flush_callback,
        )

    def _register_maintenance_tasks(self) -> bool:
        if not self._config.scheduler.enabled:
            return False
        tasks_config = self._config.scheduler.tasks
        self._scheduler.register(
            MaintenanceTaskSpec(
                owner=self._MAINTENANCE_OWNER,
                name=self._OBSERVER_IDLE_FLUSH_TASK,
                interval_seconds=tasks_config.observer_idle_flush_interval_seconds,
                enabled=tasks_config.enable_observer_idle_flush,
            ),
            self._ingressor.scan_idle_sessions_once,
        )
        return True

    def _unregister_maintenance_tasks(self) -> int:
        return self._scheduler.unregister_owner(self._MAINTENANCE_OWNER)

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
        if self._maintenance_registered:
            return
        self._maintenance_registered = self._register_maintenance_tasks()

    async def stop(self) -> None:
        if not self._maintenance_registered:
            return
        self._unregister_maintenance_tasks()
        self._maintenance_registered = False

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
