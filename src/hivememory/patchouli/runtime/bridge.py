"""Patchouli cross-system bus bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from hivememory.system.contracts.events import GlobalEvents
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


@dataclass(frozen=True)
class PatchouliPublicApi:
    """Public Patchouli API surface mounted by PatchouliBridge."""

    chat: PatchouliService
    memory: Any
    memory_tasks: Any
    agent_profiles: Any
    topics: Any
    readiness: Any


class PatchouliBridge:
    """Bridge Patchouli local capabilities and events to system-level buses."""


    def __init__(
        self,
        *,
        local_bus: PatchouliBus | None = None,
        public_api: PatchouliPublicApi,
        global_bus: GlobalSystemBus | None = None,
    ) -> None:
        if local_bus is None:
            raise ValueError("PatchouliBridge requires a PatchouliBus")
        self._local_bus = local_bus
        self._public_api = public_api
        self._global_bus = global_bus
        self._public_routes_registered = False
        self._local_events_registered = False

    @property
    def public_routes_registered(self) -> bool:
        return self._public_routes_registered

    @property
    def local_events_registered(self) -> bool:
        return self._local_events_registered

    def mount(self) -> None:
        if not self._local_events_registered:
            self._register_local_event_bridges()
            self._local_events_registered = True

        if self._global_bus is not None and not self._public_routes_registered:
            self._register_public_routes()
            self._public_routes_registered = True

    def unmount(self) -> None:
        if self._global_bus is not None and self._public_routes_registered:
            self._unregister_public_routes()
            self._public_routes_registered = False

        if self._local_events_registered:
            self._unregister_local_event_bridges()
            self._local_events_registered = False

    def _public_route_bindings(self) -> list[tuple[str, Any]]:
        return [
            (
                PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE,
                self._public_api.chat.analyze_and_retrieve,
            ),
            (
                PatchouliRoutes.SUBMIT_INTERACTION,
                self._public_api.chat.submit_interaction,
            ),
            (
                PatchouliRoutes.MEMORY_CREATE,
                self._public_api.memory.create_memory,
            ),
            (
                PatchouliRoutes.MEMORY_LIST,
                self._public_api.memory.list_memories,
            ),
            (
                PatchouliRoutes.MEMORY_GET,
                self._public_api.memory.get_memory,
            ),
            (
                PatchouliRoutes.MEMORY_UPDATE,
                self._public_api.memory.update_memory,
            ),
            (
                PatchouliRoutes.MEMORY_DELETE,
                self._public_api.memory.delete_memory,
            ),
            (
                PatchouliRoutes.MEMORY_RECORD_FEEDBACK,
                self._public_api.memory.record_feedback,
            ),
            (
                PatchouliRoutes.MEMORY_TASK_LIST,
                self._public_api.memory_tasks.list_memory_tasks,
            ),
            (
                PatchouliRoutes.MEMORY_TASK_GET,
                self._public_api.memory_tasks.get_memory_task,
            ),
            (
                PatchouliRoutes.MEMORY_TASK_CANCEL,
                self._public_api.memory_tasks.cancel_memory_task,
            ),
            (
                PatchouliRoutes.AGENT_PROFILE_CREATE,
                self._public_api.agent_profiles.create_agent_profile,
            ),
            (
                PatchouliRoutes.AGENT_PROFILE_LIST,
                self._public_api.agent_profiles.list_agent_profiles,
            ),
            (
                PatchouliRoutes.TOPIC_LIST_ACTIVE,
                self._public_api.topics.list_active_topics,
            ),
            (
                PatchouliRoutes.MEMORY_RETRIEVE,
                self._public_api.memory.retrieve,
            ),
            (
                PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES,
                self._public_api.memory.retrieve_by_aliases,
            ),
            (
                PatchouliRoutes.GET_AGENT_PROFILE,
                self._public_api.agent_profiles.get_agent_profile,
            ),
            (
                PatchouliRoutes.PREPARE_AGENT_RUN,
                self._public_api.chat.prepare_agent_run,
            ),
            (
                PatchouliRoutes.FINALIZE_AGENT_RUN,
                self._public_api.chat.finalize_agent_run,
            ),
            (
                PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN,
                self._public_api.chat.cleanup_prepared_agent_run,
            ),
            (
                PatchouliRoutes.MANUAL_SETTLE_TOPIC,
                self._public_api.topics.settle_topic,
            ),
            (
                PatchouliRoutes.EVICT_TOPIC,
                self._public_api.topics.evict_topic,
            ),
            (
                PatchouliRoutes.RECORD_MEMORY_CITATION,
                self._public_api.chat.record_memory_citation,
            ),
            (
                PatchouliRoutes.WARMUP_MODELS,
                self._public_api.readiness.warmup_models,
            ),
            (
                PatchouliRoutes.MODELS_READY,
                self._public_api.readiness.is_models_ready,
            ),
        ]

    def _register_public_routes(self) -> None:
        if self._global_bus is None:
            return
        for route, handler in self._public_route_bindings():
            self._global_bus.register(route, handler)

    def _unregister_public_routes(self) -> None:
        if self._global_bus is None:
            return
        for route, _ in self._public_route_bindings():
            self._global_bus.unregister(route)

    def _register_local_event_bridges(self) -> None:
        self._local_bus.subscribe(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            self._forward_pending_atom_settled,
        )
        self._local_bus.subscribe(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            self._forward_pending_atom_failed,
        )
        self._local_bus.subscribe(
            PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
            self._forward_pending_atom_cancelled,
        )

    def _unregister_local_event_bridges(self) -> None:
        self._local_bus.unsubscribe(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            self._forward_pending_atom_settled,
        )
        self._local_bus.unsubscribe(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            self._forward_pending_atom_failed,
        )
        self._local_bus.unsubscribe(
            PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
            self._forward_pending_atom_cancelled,
        )

    async def _forward_pending_atom_settled(self, *, settlement) -> None:
        if self._global_bus is None:
            return
        await self._global_bus.publish(
            GlobalEvents.PENDING_ATOM_SETTLED,
            settlement=settlement,
        )

    async def _forward_pending_atom_failed(self, *, pending_alias: str) -> None:
        if self._global_bus is None:
            return
        await self._global_bus.publish(
            GlobalEvents.PENDING_ATOM_FAILED,
            pending_alias=pending_alias,
        )

    async def _forward_pending_atom_cancelled(self, *, pending_alias: str) -> None:
        if self._global_bus is None:
            return
        await self._global_bus.publish(
            GlobalEvents.PENDING_ATOM_CANCELLED,
            pending_alias=pending_alias,
        )


__all__ = ["PatchouliBridge", "PatchouliPublicApi"]
