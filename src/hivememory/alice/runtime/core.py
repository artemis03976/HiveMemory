from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import Any, AsyncGenerator, Optional

from hivememory.core.models import MemoryAtom
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    AgentRunStatus,
)

from hivememory.alice.contracts.local_routes import AliceLocalRoutes
from hivememory.alice.runtime.agent.runtime import AgentRuntime
from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler
from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
from hivememory.alice.runtime.orchestrator import AgentOrchestrator
from hivememory.alice.runtime.bus import AliceBus
from hivememory.agent_runtime.cache import KoakumaAtomCache
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.resolver import RuntimeAliasResolver
from hivememory.agent_runtime.mtp.mtp_executor import KoakumaMTPExecutor
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import AliceConfig, MemoryCompilerConfig, SharedConfig
from hivememory.system.contracts.events import GlobalEvents
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink

logger = logging.getLogger(__name__)


class StreamExitReason(str, Enum):
    RUNNING = "running"
    TERMINAL = "terminal"
    FAILED = "failed"
    CLOSED = "closed"
    MISSING_DONE = "missing_done"


class AliceRuntime:
    """Alice 子系统的 runtime 聚合根。"""

    def __init__(
        self,
        alice_config: AliceConfig,
        shared_config: SharedConfig,
        memory_compiler_config: MemoryCompilerConfig,
        global_bus: Optional[GlobalSystemBus] = None,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._alice_config = alice_config
        self._shared_config = shared_config
        self._memory_compiler_config = memory_compiler_config
        self._global_bus = global_bus
        self._runtime_events = runtime_events or NullRuntimeEventSink()
        self._local_bus = AliceBus()
        self._local_routes_registered = False
        self._global_events_registered = False

        # ---- 引擎层 (agent_runtime)：单 Agent 执行能力 ----
        self._pending_runtime = PendingAtomRuntime()
        self._atom_cache = KoakumaAtomCache()
        self._alias_resolver = RuntimeAliasResolver(
            pending_runtime=self._pending_runtime,
            atom_cache=self._atom_cache,
            bus=self._local_bus,
        )
        self._koakuma = KoakumaRuntime(
            bus=self._local_bus,
            config=alice_config.koakuma,
            alias_resolver=self._alias_resolver,
            memory_compiler_config=memory_compiler_config,
        )
        self._mtp_executor = KoakumaMTPExecutor(self._koakuma)
        self._agent_runtime = AgentRuntime(
            mtp_executor=self._mtp_executor,
            alice_config=alice_config,
            shared_config=shared_config,
            pending_runtime=self._pending_runtime,
        )

        # ---- 编排层 (alice)：多 Agent 编排，拿门面跑单 Agent ----
        self._prompt_assembler = AgentPromptAssembler(
            alice_config.koakuma,
        )
        self._orchestrator = AgentOrchestrator(
            agent_runtime=self._agent_runtime,
            frame_scheduler=FrameScheduler(prompt_assembler=self._prompt_assembler),
            agent_profile_resolver=AgentProfileResolver(local_bus=self._local_bus),
            alias_resolver=self._alias_resolver,
        )

        logger.info("AliceRuntime 初始化完成")

    def register_preretrieval_aliases(self, memories: list[MemoryAtom]) -> None:
        self._atom_cache.ingest_atoms(memories)
        if memories:
            logger.debug(
                f"预检索记忆缓存完成: {len(memories)} 条记忆已缓存到 Koakuma"
            )

    async def _on_pending_atom_settled(self, *, settlement) -> None:
        """Handle settlement event from Patchouli generation pipeline."""
        self._pending_runtime.settle(settlement)
        await self._refresh_l1_cache_for_settlement(settlement)
        logger.info(
            f"Settlement applied: {settlement.pending_alias} -> "
            f"{settlement.resolution.value} (canonical={settlement.canonical_alias})"
        )

    async def _on_pending_atom_failed(self, *, pending_alias: str) -> None:
        """Handle generation failure event — mark atom as FAILED to unblock lifecycle."""
        self._agent_runtime.mark_task_failed(pending_alias)
        logger.warning(f"PendingAtom marked FAILED: {pending_alias}")

    async def _on_pending_atom_cancelled(self, *, pending_alias: str) -> None:
        """Handle generation cancellation event — mark atom as CANCELLED."""
        self._agent_runtime.mark_task_cancelled(pending_alias)
        logger.warning(f"PendingAtom marked CANCELLED: {pending_alias}")

    async def _refresh_l1_cache_for_settlement(self, settlement) -> None:
        """Refresh L1 atom cache after a pending atom points to a canonical atom."""
        canonical_alias = settlement.canonical_alias
        if not canonical_alias:
            return

        self._atom_cache.invalidate_alias(canonical_alias)

        try:
            retrieval_response = await self._local_bus.request(
                GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
                aliases=[canonical_alias],
            )
        except Exception as exc:
            logger.warning(
                f"Failed to refresh L1 cache for settled atom "
                f"'{canonical_alias}': {exc}"
            )
            return

        memories = getattr(retrieval_response, "memories", []) or []
        memory = memories[0] if memories else None
        if memory is None:
            logger.debug(
                f"No canonical atom returned while refreshing L1 cache: "
                f"alias='{canonical_alias}'"
            )
            return

        self._atom_cache.ingest_atom(memory)

    def mount_local_routes(self) -> None:
        if self._local_routes_registered:
            return

        self._local_bus.register(
            AliceLocalRoutes.RUN_AGENT,
            self.run_agent,
        )
        self._local_bus.register(
            AliceLocalRoutes.RUN_AGENT_STREAM,
            self.run_agent_stream,
        )

        if self._global_bus is not None:
            self._local_bus.register(
                GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
                self._request_patchouli_memory_retrieve,
            )
            self._local_bus.register(
                GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
                self._request_patchouli_memory_retrieve_by_aliases,
            )
            self._local_bus.register(
                GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
                self._request_patchouli_get_agent_profile,
            )
            self._local_bus.register(
                GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION,
                self._request_patchouli_record_memory_citation,
            )
            if not self._global_events_registered:
                self._global_bus.subscribe(
                    GlobalEvents.PENDING_ATOM_SETTLED,
                    self._on_pending_atom_settled,
                )
                self._global_bus.subscribe(
                    GlobalEvents.PENDING_ATOM_FAILED,
                    self._on_pending_atom_failed,
                )
                self._global_bus.subscribe(
                    GlobalEvents.PENDING_ATOM_CANCELLED,
                    self._on_pending_atom_cancelled,
                )
                self._global_events_registered = True

        self._local_routes_registered = True

    def unmount_local_routes(self) -> None:
        if not self._local_routes_registered:
            return

        for route in AliceLocalRoutes.ALL:
            self._local_bus.unregister(route)
        if self._global_bus is not None:
            self._local_bus.unregister(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE)
            self._local_bus.unregister(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES)
            self._local_bus.unregister(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE)
            self._local_bus.unregister(GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION)
            if self._global_events_registered:
                self._global_bus.unsubscribe(
                    GlobalEvents.PENDING_ATOM_SETTLED,
                    self._on_pending_atom_settled,
                )
                self._global_bus.unsubscribe(
                    GlobalEvents.PENDING_ATOM_FAILED,
                    self._on_pending_atom_failed,
                )
                self._global_bus.unsubscribe(
                    GlobalEvents.PENDING_ATOM_CANCELLED,
                    self._on_pending_atom_cancelled,
                )
                self._global_events_registered = False
        self._local_routes_registered = False

    async def _request_patchouli_memory_retrieve(self, *args: Any, **kwargs: Any) -> Any:
        if self._global_bus is None:
            raise KeyError(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
            *args,
            **kwargs,
        )

    async def _request_patchouli_memory_retrieve_by_aliases(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._global_bus is None:
            raise KeyError(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
            *args,
            **kwargs,
        )

    async def _request_patchouli_get_agent_profile(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._global_bus is None:
            raise KeyError(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
            *args,
            **kwargs,
        )

    async def _request_patchouli_record_memory_citation(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._global_bus is None:
            raise KeyError(GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION,
            *args,
            **kwargs,
        )

    def health(self) -> dict[str, Any]:
        return {
            "local_routes_registered": self._local_routes_registered,
            "agent_runtime": self._agent_runtime.health(),
            "koakuma_runtime": {
                "status": "ok",
            },
            "profile_cache": {
                "status": "ok",
            },
        }

    @property
    def local_bus(self) -> AliceBus:
        return self._local_bus

    @property
    def local_routes_registered(self) -> bool:
        return self._local_routes_registered

    async def run_agent(
        self,
        agent_run_context: AgentRunContext,
        generation_options: Optional[dict[str, Any]] = None,
        cancel_event=None,
    ) -> AgentRunResult:
        agent_run_id = f"agent_run_{uuid.uuid4().hex}"
        self._emit_agent_event(
            RuntimeEventType.AGENT_RUN_STARTED,
            agent_run_context,
            agent_run_id=agent_run_id,
            status="started",
        )
        self.register_preretrieval_aliases(agent_run_context.retrieval_result.memories)
        messages = self._prompt_assembler.build_main_agent_messages(agent_run_context)
        try:
            result = await self._orchestrator.run_agent(
                messages=messages,
                identity=agent_run_context.identity,
                topic_id=agent_run_context.topic_id,
                generation_options=generation_options,
                agent_profile=agent_run_context.agent_profile,
                cancel_event=cancel_event,
            )
            self._emit_agent_terminal(agent_run_context, agent_run_id, result)
            return result
        except Exception:
            self._emit_agent_event(
                RuntimeEventType.AGENT_RUN_FAILED,
                agent_run_context,
                agent_run_id=agent_run_id,
                status="failed",
                severity="error",
                message="Agent run failed.",
            )
            raise

    async def run_agent_stream(
        self,
        agent_run_context: AgentRunContext,
        generation_options: Optional[dict[str, Any]] = None,
        cancel_event=None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        agent_run_id = f"agent_run_{uuid.uuid4().hex}"
        self._emit_agent_event(
            RuntimeEventType.AGENT_RUN_STARTED,
            agent_run_context,
            agent_run_id=agent_run_id,
            status="started",
        )
        self.register_preretrieval_aliases(agent_run_context.retrieval_result.memories)
        messages = self._prompt_assembler.build_main_agent_messages(agent_run_context)
        exit_reason = StreamExitReason.RUNNING
        try:
            async for event in self._orchestrator.run_agent_stream(
                messages=messages,
                identity=agent_run_context.identity,
                topic_id=agent_run_context.topic_id,
                generation_options=generation_options,
                agent_profile=agent_run_context.agent_profile,
                cancel_event=cancel_event,
            ):
                if event.get("event") == "done":
                    self._emit_agent_terminal(
                        agent_run_context,
                        agent_run_id,
                        AgentRunResult(**event["data"]),
                    )
                    exit_reason = StreamExitReason.TERMINAL
                yield event
            if exit_reason != StreamExitReason.TERMINAL:
                exit_reason = StreamExitReason.MISSING_DONE
                self._emit_agent_event(
                    RuntimeEventType.AGENT_RUN_FAILED,
                    agent_run_context,
                    agent_run_id=agent_run_id,
                    status="failed",
                    severity="error",
                    message="Agent stream ended without done event.",
                )
                raise RuntimeError("Agent stream ended without done event")
        except Exception:
            if exit_reason not in (
                StreamExitReason.TERMINAL,
                StreamExitReason.MISSING_DONE,
            ):
                exit_reason = StreamExitReason.FAILED
                self._emit_agent_event(
                    RuntimeEventType.AGENT_RUN_FAILED,
                    agent_run_context,
                    agent_run_id=agent_run_id,
                    status="failed",
                    severity="error",
                    message="Agent stream run failed.",
                )
            raise
        finally:
            if exit_reason == StreamExitReason.RUNNING:
                exit_reason = StreamExitReason.CLOSED
                if cancel_event is not None:
                    cancel_event.set()
                self._emit_agent_event(
                    RuntimeEventType.AGENT_RUN_CANCELLED,
                    agent_run_context,
                    agent_run_id=agent_run_id,
                    status="cancelled",
                    message="Agent stream closed before terminal event.",
                    data={"close_reason": "stream_closed"},
                )

    def _emit_agent_terminal(
        self,
        agent_run_context: AgentRunContext,
        agent_run_id: str,
        result: AgentRunResult,
    ) -> None:
        event_type = (
            RuntimeEventType.AGENT_RUN_CANCELLED
            if result.status == AgentRunStatus.CANCELLED.value
            else RuntimeEventType.AGENT_RUN_COMPLETED
        )
        self._emit_agent_event(
            event_type,
            agent_run_context,
            agent_run_id=agent_run_id,
            status=str(result.status),
            data={
                "mtp_iterations": result.mtp_iterations,
                "total_iterations": result.total_iterations,
                "materialize_task_count": len(result.materialize_tasks),
            },
        )

    def _emit_agent_event(
        self,
        event_type: RuntimeEventType,
        agent_run_context: AgentRunContext,
        *,
        agent_run_id: str,
        status: str,
        severity: str = "info",
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._runtime_events.emit(
            RuntimeEvent(
                event_type=event_type,
                task_type="foreground",
                agent_run_id=agent_run_id,
                topic_id=agent_run_context.topic_id,
                agent_id=agent_run_context.identity.agent_id,
                status=status,
                severity=severity,  # type: ignore[arg-type]
                message=message,
                data=data or {},
            )
        )


__all__ = ["AliceRuntime"]
