from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from hivememory.agent_runtime.cache import KoakumaAtomCache
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.mtp.mtp_executor import KoakumaMTPExecutor
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.agent_runtime.products import RuntimeProducts
from hivememory.agent_runtime.resolver import RuntimeAliasResolver
from hivememory.alice.contracts.local_routes import AliceLocalRoutes
from hivememory.alice.runtime.agent.call_coordinator import CallCoordinator
from hivememory.alice.runtime.agent.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.runtime.agent.profile_resolver import AgentProfileResolver
from hivememory.alice.runtime.agent.run_scheduler import RunScheduler
from hivememory.alice.runtime.agent.run_session import RunSession
from hivememory.alice.runtime.agent.runtime import AgentRuntime
from hivememory.alice.runtime.bus import AliceBus
from hivememory.alice.runtime.route_bindings import build_alice_route_bindings
from hivememory.core.models import OMNI_DOLL_PROFILE, AgentProfile, Identity, MemoryAtom
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    AgentRunStatus,
)
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import AliceConfig, MemoryCompilerConfig, SharedConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.model_registry import ModelRegistry
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
        runtime_events: RuntimeEventSink | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._alice_config = alice_config
        self._shared_config = shared_config
        self._memory_compiler_config = memory_compiler_config
        self._runtime_events = runtime_events or NullRuntimeEventSink()
        self._local_bus = AliceBus()
        self._local_routes_registered = False

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
            pending_runtime=self._pending_runtime,
            model_registry=model_registry,  # 传入注册表，用于逐帧模型解析
        )

        # ---- 编排层 (alice)：每个 run 独立构造 root frame 与调度状态机 ----
        self._prompt_assembler = AgentPromptAssembler(
            alice_config.koakuma,
        )
        self._frame_factory = FrameFactory()
        self._agent_profile_resolver = AgentProfileResolver(local_bus=self._local_bus)
        self._call_coordinator = CallCoordinator(
            self._agent_runtime,
            self._agent_profile_resolver,
            self._alias_resolver,
            frame_factory=self._frame_factory,
            prompt_assembler=self._prompt_assembler,
        )

        logger.info("AliceRuntime 初始化完成")

    def register_preretrieval_aliases(self, memories: list[MemoryAtom]) -> None:
        self._atom_cache.ingest_atoms(memories)
        if memories:
            logger.debug(f"预检索记忆缓存完成: {len(memories)} 条记忆已缓存到 Koakuma")

    async def on_pending_atom_settled(self, *, settlement) -> None:
        """Handle settlement event from Patchouli generation pipeline."""
        self._pending_runtime.settle(settlement)
        await self._refresh_l1_cache_for_settlement(settlement)
        logger.info(
            f"Settlement applied: {settlement.pending_alias} -> "
            f"{settlement.resolution.value} (canonical={settlement.canonical_alias})"
        )

    async def on_pending_atom_failed(self, *, pending_alias: str) -> None:
        """Handle generation failure event — mark atom as FAILED to unblock lifecycle."""
        self._agent_runtime.mark_task_failed(pending_alias)
        logger.warning(f"PendingAtom marked FAILED: {pending_alias}")

    async def on_pending_atom_cancelled(self, *, pending_alias: str) -> None:
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
                f"Failed to refresh L1 cache for settled atom " f"'{canonical_alias}': {exc}"
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

        for route, handler in build_alice_route_bindings(self):
            self._local_bus.register(route, handler)
        self._local_routes_registered = True

    def unmount_local_routes(self) -> None:
        if not self._local_routes_registered:
            return

        for route in AliceLocalRoutes.ALL:
            self._local_bus.unregister(route)
        self._local_routes_registered = False

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
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        generation_id: str | None = None,
    ) -> AgentRunResult:
        session = self._create_run_session(
            generation_id=generation_id,
            cancel_event=cancel_event,
        )
        self._emit_agent_event(
            RuntimeEventType.AGENT_RUN_STARTED,
            agent_run_context,
            agent_run_id=session.agent_run_id,
            status="started",
        )
        self.register_preretrieval_aliases(agent_run_context.retrieval_result.memories)
        messages = self._prompt_assembler.build_main_agent_messages(agent_run_context)
        try:
            frame = self._create_root_frame(
                messages=messages,
                identity=agent_run_context.identity,
                topic_id=agent_run_context.topic_id,
                session=session,
                agent_profile=agent_run_context.agent_profile,
            )
            scheduler = RunScheduler(
                agent_runtime=self._agent_runtime,
                session=session,
                call_coordinator=self._call_coordinator,
            )
            engine_result = await scheduler.run(
                frame,
                generation_options=generation_options,
            )
            result = self._assemble_agent_run_result(
                frame,
                engine_result,
                scheduler.runtime_products or RuntimeProducts(),
            )
            self._emit_agent_terminal(agent_run_context, session.agent_run_id, result)
            return result
        except Exception:
            self._emit_agent_event(
                RuntimeEventType.AGENT_RUN_FAILED,
                agent_run_context,
                agent_run_id=session.agent_run_id,
                status="failed",
                severity="error",
                message="Agent run failed.",
            )
            raise

    async def run_agent_stream(
        self,
        agent_run_context: AgentRunContext,
        generation_options: dict[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        generation_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        session = self._create_run_session(
            generation_id=generation_id,
            cancel_event=cancel_event,
        )
        self._emit_agent_event(
            RuntimeEventType.AGENT_RUN_STARTED,
            agent_run_context,
            agent_run_id=session.agent_run_id,
            status="started",
        )
        self.register_preretrieval_aliases(agent_run_context.retrieval_result.memories)
        messages = self._prompt_assembler.build_main_agent_messages(agent_run_context)
        exit_reason = StreamExitReason.RUNNING
        scheduler_stream: AsyncGenerator[dict[str, Any], None] | None = None
        try:
            frame = self._create_root_frame(
                messages=messages,
                identity=agent_run_context.identity,
                topic_id=agent_run_context.topic_id,
                session=session,
                agent_profile=agent_run_context.agent_profile,
            )
            scheduler = RunScheduler(
                agent_runtime=self._agent_runtime,
                session=session,
                call_coordinator=self._call_coordinator,
            )
            event_metadata = self._event_metadata_for_frame(frame)
            scheduler_stream = scheduler.run_stream(
                frame,
                generation_options=generation_options,
                event_metadata=event_metadata,
            )
            async for event in scheduler_stream:
                yield event

            terminal_result = scheduler.terminal_result
            if terminal_result is None:
                exit_reason = StreamExitReason.MISSING_DONE
                self._emit_agent_event(
                    RuntimeEventType.AGENT_RUN_FAILED,
                    agent_run_context,
                    agent_run_id=session.agent_run_id,
                    status="failed",
                    severity="error",
                    message="Agent stream ended without done event.",
                )
                raise RuntimeError("Agent stream ended without done event")
            result = self._assemble_agent_run_result(
                frame,
                terminal_result,
                scheduler.runtime_products or RuntimeProducts(),
            )
            self._emit_agent_terminal(agent_run_context, session.agent_run_id, result)
            exit_reason = StreamExitReason.TERMINAL
            yield {
                "event": "done",
                "data": {
                    **result.model_dump(),
                    **event_metadata,
                    "stream_sequence": scheduler.next_stream_sequence,
                },
            }
        except Exception:
            if exit_reason not in (
                StreamExitReason.TERMINAL,
                StreamExitReason.MISSING_DONE,
            ):
                exit_reason = StreamExitReason.FAILED
                self._emit_agent_event(
                    RuntimeEventType.AGENT_RUN_FAILED,
                    agent_run_context,
                    agent_run_id=session.agent_run_id,
                    status="failed",
                    severity="error",
                    message="Agent stream run failed.",
                )
            raise
        finally:
            if exit_reason == StreamExitReason.RUNNING:
                exit_reason = StreamExitReason.CLOSED
                session.cancel_event.set()
                self._emit_agent_event(
                    RuntimeEventType.AGENT_RUN_CANCELLED,
                    agent_run_context,
                    agent_run_id=session.agent_run_id,
                    status="cancelled",
                    message="Agent stream closed before terminal event.",
                    data={"close_reason": "stream_closed"},
                )
            if scheduler_stream is not None:
                await scheduler_stream.aclose()

    def _create_root_frame(
        self,
        *,
        messages: list[dict[str, str]],
        identity: Identity,
        topic_id: str,
        agent_profile: AgentProfile | None,
        session: RunSession,
    ) -> ExecutionFrame:
        """为当前 run 创建并登记唯一 root frame。"""
        profile = agent_profile or OMNI_DOLL_PROFILE
        policy = FrameExecutionPolicy.from_profile(
            profile,
            max_iterations=getattr(self._agent_runtime, "max_iterations", None),
        )
        frame = self._frame_factory.create(
            FrameSpec(
                runtime_scope=self._frame_factory.scope(run_id=session.agent_run_id),
                profile=profile,
                identity=identity,
                messages=messages,
                topic_id=topic_id or "",
                execution_policy=policy,
            )
        )
        session.register_root_frame(frame)
        return frame

    @staticmethod
    def _assemble_agent_run_result(
        frame: ExecutionFrame,
        engine_result: FrameExecutionResult,
        runtime_products: RuntimeProducts,
    ) -> AgentRunResult:
        """把执行层终态与产品投影为稳定的 Alice 公共结果。"""
        if engine_result.status == FrameExecutionStatus.CANCELLED:
            run_status = AgentRunStatus.CANCELLED
        elif engine_result.status == FrameExecutionStatus.COMPLETED:
            run_status = AgentRunStatus.COMPLETED
        else:
            run_status = AgentRunStatus.FAILED
        progress = frame.progress
        return AgentRunResult(
            status=run_status,
            final_text="".join(progress.text_segments),
            mtp_iterations=max(0, progress.iteration - 1),
            total_iterations=progress.iteration,
            turn_events=progress.turn_events,
            materialize_tasks=list(runtime_products.materialize_tasks),
            model_used=progress.model_used,
        )

    @staticmethod
    def _event_metadata_for_frame(frame: ExecutionFrame) -> dict[str, Any]:
        agent_id = getattr(frame.agent_profile, "alias", None) or frame.identity.agent_id
        return {
            "agent_run_id": frame.runtime_scope.run_id,
            "action_id": None,
            "scope": "main",
            "depth": 0,
            "agent_id": agent_id,
            "frame_id": frame.runtime_scope.frame_id,
        }

    @staticmethod
    def _create_run_session(
        *,
        generation_id: str | None,
        cancel_event: asyncio.Event | None,
    ) -> RunSession:
        return RunSession(
            agent_run_id=f"agent_run_{uuid.uuid4().hex}",
            generation_id=generation_id,
            cancel_event=cancel_event if cancel_event is not None else asyncio.Event(),
        )

    def _emit_agent_terminal(
        self,
        agent_run_context: AgentRunContext,
        agent_run_id: str,
        result: AgentRunResult,
    ) -> None:
        if result.status == AgentRunStatus.CANCELLED.value:
            event_type = RuntimeEventType.AGENT_RUN_CANCELLED
        elif result.status == AgentRunStatus.FAILED.value:
            event_type = RuntimeEventType.AGENT_RUN_FAILED
        else:
            event_type = RuntimeEventType.AGENT_RUN_COMPLETED
        self._emit_agent_event(
            event_type,
            agent_run_context,
            agent_run_id=agent_run_id,
            status=str(result.status),
            severity=("error" if result.status == AgentRunStatus.FAILED.value else "info"),
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
