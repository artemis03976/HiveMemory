from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from hivememory.core.models import IdentityScope
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmissionQueue,
)
from hivememory.system.config.memory_compiler import FullContextStrategyConfig
from hivememory.system.runtime.events import RuntimeEventSink
from hivememory.system.runtime.scheduler.models import MaintenanceTaskSpec
from hivememory.system.services.passive import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveMessageIngressor,
)
from hivememory.system.services.passive.models import (
    DEFAULT_EXTERNAL_CONVERSATION_ID,
    DEFAULT_PASSIVE_SOURCE,
)

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
        interaction_queue: InteractionSubmissionQueue,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._config = config
        self._scheduler = scheduler
        self._interaction_queue = interaction_queue
        self._ingressor = PassiveMessageIngressor(
            bus=bus,
            interaction_queue=interaction_queue,
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
        # PatchouliSystem.stop 会停止 claim；先在这里等待 passive 已移交的 work。
        await self._interaction_queue.drain_all()
        queue_pending = await self._interaction_queue.pending_count()
        return {
            "success": queue_pending == 0,
            "observer_payloads_submitted": result["accepted_submissions"],
            "observer_payloads_pending": queue_pending,
        }

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    async def ingest_event(
        self,
        event: PassiveIngressEvent,
        identity_scope: IdentityScope,
    ) -> dict[str, Any]:
        """接收单个外部事件。

        公共响应只包含外部调用方实际需要的接收状态与 memory context，
        不暴露 Gateway 内部 execution state、runtime event 或 fallback 细节。

        身份入口约定（v0.6.2 收敛）：connector 侧的 ``user_id + agent_id``
        选择由 server 边界一次性冻结为 ``identity_scope``，本层不再解析。
        """
        outcome = await self._ingressor.route_event_scoped(
            event=event,
            identity_scope=identity_scope,
            interaction_id=f"passive_{uuid4().hex}",
        )

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
        identity_scope: IdentityScope,
    ) -> bool:
        """显式把指定外部会话的当前 turn 移交 submission queue。

        会话分桶键由 ``source + external_conversation_id`` 与 scope 的
        actor 三元组共同构成（共享 infra 命名键，不解释 scope 对象）。

        Returns:
            True 表示当前 turn 已被 queue 接收。
        """
        key = PassiveConversationKey.build(
            source=source,
            external_conversation_id=external_conversation_id,
            identity_scope=identity_scope,
        )
        submitted = await self._ingressor.flush_conversation(key)
        return submitted > 0

    async def flush_ingressor(
        self,
        identity_scope: IdentityScope,
        session_id: str | None = None,
        source: str = DEFAULT_PASSIVE_SOURCE,
    ) -> bool:
        """过渡期兼容入口：按 session_id 显式 flush。

        新调用方应改用 `flush_conversation(source, external_conversation_id, ...)`。
        """
        return await self.flush_conversation(
            source=source,
            external_conversation_id=session_id or DEFAULT_EXTERNAL_CONVERSATION_ID,
            identity_scope=identity_scope,
        )
