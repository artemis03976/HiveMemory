"""Patchouli cross-system bus bridge."""

from __future__ import annotations

from typing import Any

from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.runtime import PatchouliRuntime
from hivememory.patchouli.service import PatchouliService
from hivememory.system.contracts.events import GlobalEvents
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class PatchouliBridge:
    """Bridge Patchouli local capabilities and events to system-level buses."""

    _PUBLIC_ROUTES = (
        PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE,
        PatchouliRoutes.SUBMIT_INTERACTION,
        PatchouliRoutes.MEMORY_CREATE,
        PatchouliRoutes.MEMORY_LIST,
        PatchouliRoutes.MEMORY_GET,
        PatchouliRoutes.MEMORY_UPDATE,
        PatchouliRoutes.MEMORY_DELETE,
        PatchouliRoutes.MEMORY_RECORD_FEEDBACK,
        PatchouliRoutes.MEMORY_TASK_LIST,
        PatchouliRoutes.MEMORY_TASK_GET,
        PatchouliRoutes.MEMORY_TASK_CANCEL,
        PatchouliRoutes.AGENT_PROFILE_CREATE,
        PatchouliRoutes.AGENT_PROFILE_LIST,
        PatchouliRoutes.TOPIC_LIST_ACTIVE,
        PatchouliRoutes.MEMORY_RETRIEVE,
        PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES,
        PatchouliRoutes.GET_AGENT_PROFILE,
        PatchouliRoutes.PREPARE_AGENT_RUN,
        PatchouliRoutes.FINALIZE_AGENT_RUN,
        PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN,
        PatchouliRoutes.MANUAL_ARCHIVE_TOPIC,
        PatchouliRoutes.EVICT_TOPIC,
        PatchouliRoutes.RECORD_MEMORY_CITATION,
        PatchouliRoutes.WARMUP_MODELS,
        PatchouliRoutes.MODELS_READY,
    )

    def __init__(
        self,
        *,
        runtime: PatchouliRuntime,
        service: PatchouliService,
        memory_management_service: Any,
        memory_task_management_service: Any,
        agent_profile_management_service: Any,
        topic_management_service: Any,
        model_readiness_service: Any,
        global_bus: GlobalSystemBus | None = None,
    ) -> None:
        self._runtime = runtime
        self._service = service
        self._memory_management_service = memory_management_service
        self._memory_task_management_service = memory_task_management_service
        self._agent_profile_management_service = agent_profile_management_service
        self._topic_management_service = topic_management_service
        self._model_readiness_service = model_readiness_service
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
                self._service.analyze_and_retrieve,
            ),
            (
                PatchouliRoutes.SUBMIT_INTERACTION,
                self._runtime.librarian_core.submit_interaction,
            ),
            (
                PatchouliRoutes.MEMORY_CREATE,
                self._memory_management_service.create_memory,
            ),
            (
                PatchouliRoutes.MEMORY_LIST,
                self._memory_management_service.list_memories,
            ),
            (
                PatchouliRoutes.MEMORY_GET,
                self._memory_management_service.get_memory,
            ),
            (
                PatchouliRoutes.MEMORY_UPDATE,
                self._memory_management_service.update_memory,
            ),
            (
                PatchouliRoutes.MEMORY_DELETE,
                self._memory_management_service.delete_memory,
            ),
            (
                PatchouliRoutes.MEMORY_RECORD_FEEDBACK,
                self._memory_management_service.record_feedback,
            ),
            (
                PatchouliRoutes.MEMORY_TASK_LIST,
                self._memory_task_management_service.list_memory_tasks,
            ),
            (
                PatchouliRoutes.MEMORY_TASK_GET,
                self._memory_task_management_service.get_memory_task,
            ),
            (
                PatchouliRoutes.MEMORY_TASK_CANCEL,
                self._memory_task_management_service.cancel_memory_task,
            ),
            (
                PatchouliRoutes.AGENT_PROFILE_CREATE,
                self._agent_profile_management_service.create_agent_profile,
            ),
            (
                PatchouliRoutes.AGENT_PROFILE_LIST,
                self._agent_profile_management_service.list_agent_profiles,
            ),
            (
                PatchouliRoutes.TOPIC_LIST_ACTIVE,
                self._topic_management_service.list_active_topics,
            ),
            (
                PatchouliRoutes.MEMORY_RETRIEVE,
                self._runtime.retrieval_familiar.retrieve_async,
            ),
            (
                PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES,
                self._runtime.retrieval_familiar.retrieve_by_aliases_async,
            ),
            (
                PatchouliRoutes.GET_AGENT_PROFILE,
                self._runtime._get_agent_profile,
            ),
            (
                PatchouliRoutes.PREPARE_AGENT_RUN,
                self._service.prepare_agent_run,
            ),
            (
                PatchouliRoutes.FINALIZE_AGENT_RUN,
                self._service.finalize_agent_run,
            ),
            (
                PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN,
                self._service.cleanup_prepared_agent_run,
            ),
            (
                PatchouliRoutes.MANUAL_ARCHIVE_TOPIC,
                self._topic_management_service.archive_topic,
            ),
            (
                PatchouliRoutes.EVICT_TOPIC,
                self._topic_management_service.evict_topic,
            ),
            (
                PatchouliRoutes.RECORD_MEMORY_CITATION,
                self._service.record_memory_citation,
            ),
            (
                PatchouliRoutes.WARMUP_MODELS,
                self._model_readiness_service.warmup_models,
            ),
            (
                PatchouliRoutes.MODELS_READY,
                self._model_readiness_service.is_models_ready,
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
        for route in self._PUBLIC_ROUTES:
            self._global_bus.unregister(route)

    def _register_local_event_bridges(self) -> None:
        self._runtime.local_bus.subscribe(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            self._forward_pending_atom_settled,
        )
        self._runtime.local_bus.subscribe(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            self._forward_pending_atom_failed,
        )
        self._runtime.local_bus.subscribe(
            PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
            self._forward_pending_atom_cancelled,
        )

    def _unregister_local_event_bridges(self) -> None:
        self._runtime.local_bus.unsubscribe(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            self._forward_pending_atom_settled,
        )
        self._runtime.local_bus.unsubscribe(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            self._forward_pending_atom_failed,
        )
        self._runtime.local_bus.unsubscribe(
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


__all__ = ["PatchouliBridge"]
