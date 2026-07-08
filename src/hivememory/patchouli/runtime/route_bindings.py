"""Declarative local route bindings for PatchouliRuntime."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.core import PatchouliRuntime
    from hivememory.patchouli.service import PatchouliService

RouteBinding = tuple[str, Callable[..., Any]]


def build_patchouli_route_bindings(
    runtime: "PatchouliRuntime",
    service: "PatchouliService",
) -> tuple[RouteBinding, ...]:
    """Build Patchouli local route bindings from the runtime composition root."""
    return (
        (
            PatchouliLocalRoutes.INGESTION_SUBMIT_INTERACTION,
            runtime.perception_familiar.submit_interaction,
        ),
        (
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            runtime.memory_generation_coordinator.submit_settlement,
        ),
        (
            PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
            runtime.memory_generation_coordinator.submit_active,
        ),
        (
            PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC,
            runtime.memory_generation_familiar.execute,
        ),
        (
            PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION,
            runtime._task_controller.submit_generation,
        ),
        (
            PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
            runtime._task_controller.submit_generation_many,
        ),
        (PatchouliLocalRoutes.MEMORY_TASK_LIST, runtime._task_controller.list_tasks),
        (PatchouliLocalRoutes.MEMORY_TASK_GET, runtime._task_controller.get_task),
        (
            PatchouliLocalRoutes.MEMORY_TASK_CANCEL,
            runtime._task_controller.cancel_task,
        ),
        (PatchouliLocalRoutes.MEMORY_TASK_WAIT, runtime._task_controller.wait_task),
        (
            PatchouliLocalRoutes.MEMORY_TASK_WAIT_MANY,
            runtime._task_controller.wait_many,
        ),
        (PatchouliLocalRoutes.MEMORY_TASK_WAIT_ALL, runtime._task_controller.wait_all),
        (
            PatchouliLocalRoutes.MEMORY_CREATE,
            runtime.memory_generation_familiar.create_external_memory,
        ),
        (PatchouliLocalRoutes.MEMORY_LIST, runtime.retrieval_familiar.list_memories),
        (
            PatchouliLocalRoutes.MEMORY_RETRIEVE,
            runtime.retrieval_familiar.retrieve_async,
        ),
        (
            PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
            runtime.retrieval_familiar.retrieve_by_aliases_async,
        ),
        (PatchouliLocalRoutes.MEMORY_GET, runtime.retrieval_familiar.get_memory),
        (
            PatchouliLocalRoutes.MEMORY_UPDATE,
            runtime.memory_generation_familiar.update_external_memory,
        ),
        (PatchouliLocalRoutes.MEMORY_DELETE, runtime.memory_library.mid_term.delete),
        (
            PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY,
            runtime.lifecycle_familiar.refresh_memory_vitality,
        ),
        (
            PatchouliLocalRoutes.LIFECYCLE_RUN_GARDENING_ONCE,
            runtime.lifecycle_familiar.run_gardening_once,
        ),
        (PatchouliLocalRoutes.RUNTIME_MODELS_WARMUP, runtime.warmup_models),
        (PatchouliLocalRoutes.RUNTIME_MODELS_READY, runtime.is_models_ready),
        (PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH, runtime.check_storage_health),
        (PatchouliLocalRoutes.MEMORY_RECORD_HIT, runtime.lifecycle_familiar.record_hit),
        (
            PatchouliLocalRoutes.MEMORY_RECORD_CITATION,
            runtime.lifecycle_familiar.record_citation,
        ),
        (
            PatchouliLocalRoutes.MEMORY_RECORD_FEEDBACK,
            runtime.lifecycle_familiar.record_feedback,
        ),
        (PatchouliLocalRoutes.MEMORY_REVIVE, runtime.lifecycle_familiar.revive_memory),
        (
            PatchouliLocalRoutes.GET_AGENT_PROFILE,
            runtime.retrieval_familiar.get_agent_profile,
        ),
        (PatchouliLocalRoutes.TOPIC_PREPARE, runtime.perception_familiar.prepare_topic),
        (PatchouliLocalRoutes.TOPIC_GET, runtime.retrieval_familiar.get_topic),
        (
            PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
            runtime.retrieval_familiar.list_active_topics,
        ),
        (PatchouliLocalRoutes.TOPIC_EVICT, runtime.perception_familiar.evict_topic),
        (
            PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY,
            runtime.perception_familiar.discard_if_empty,
        ),
        (
            PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE,
            runtime.perception_familiar.manual_settle_topic,
        ),
    )


__all__ = ["RouteBinding", "build_patchouli_route_bindings"]
