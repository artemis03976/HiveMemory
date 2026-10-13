"""PatchouliRuntime 的声明式局部路由绑定。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.components.bus.binding import RouteBinding
from hivememory.core.models import TopicData, TopicSnapshot, WorkspaceIdentity
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.core import PatchouliRuntime
    from hivememory.patchouli.service import PatchouliService


def build_patchouli_route_bindings(
    runtime: PatchouliRuntime,
    service: PatchouliService,
) -> tuple[RouteBinding, ...]:
    """从运行时组合根构建 Patchouli 局部路由绑定。"""

    # 同步查询保留原有直接调用接口；只在 RPC 边界适配为异步 handler。
    async def models_ready() -> bool:
        """读取运行时模型就绪状态。"""
        return runtime.is_models_ready()

    async def get_topic(topic_id: str, *, belong_to: WorkspaceIdentity) -> TopicData | None:
        """读取指定归属的话题。"""
        return runtime.retrieval_familiar.get_topic(topic_id, belong_to=belong_to)

    async def list_active_topics(
        *,
        belong_to: WorkspaceIdentity,
        include_empty: bool = False,
        sort_by_recency: bool = True,
    ) -> tuple[TopicSnapshot, ...]:
        """列出指定归属的活跃话题快照。"""
        return runtime.retrieval_familiar.list_active_topics(
            belong_to=belong_to,
            include_empty=include_empty,
            sort_by_recency=sort_by_recency,
        )

    async def discard_if_empty(belong_to: WorkspaceIdentity, topic_id: str) -> bool:
        """只清理指定归属下仍为空的话题。"""
        return runtime.perception_familiar.discard_if_empty(belong_to, topic_id)

    return (
        RouteBinding.of(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            runtime.memory_generation_coordinator.submit_settlement,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
            runtime.memory_generation_coordinator.submit_active,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC,
            runtime.memory_generation_familiar.execute,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION,
            runtime._task_controller.submit_generation,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
            runtime._task_controller.submit_generation_many,
        ),
        RouteBinding.of(PatchouliLocalRoutes.MEMORY_TASK_LIST, runtime._task_controller.list_tasks),
        RouteBinding.of(PatchouliLocalRoutes.MEMORY_TASK_GET, runtime._task_controller.get_task),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_TASK_CANCEL,
            runtime._task_controller.cancel_task,
        ),
        RouteBinding.of(PatchouliLocalRoutes.MEMORY_TASK_WAIT, runtime._task_controller.wait_task),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_TASK_WAIT_MANY,
            runtime._task_controller.wait_many,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_TASK_WAIT_ALL, runtime._task_controller.wait_all
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_CREATE,
            runtime.memory_generation_familiar.create_external_memory,
        ),
        RouteBinding.of(PatchouliLocalRoutes.MEMORY_LIST, runtime.retrieval_familiar.list_memories),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_RETRIEVE,
            runtime.retrieval_familiar.retrieve_async,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES,
            runtime.retrieval_familiar.retrieve_by_aliases_async,
        ),
        RouteBinding.of(PatchouliLocalRoutes.MEMORY_GET, runtime.retrieval_familiar.get_memory),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_UPDATE,
            runtime.memory_generation_familiar.update_external_memory,
        ),
        RouteBinding.of(PatchouliLocalRoutes.MEMORY_DELETE, runtime.memory_library.delete),
        RouteBinding.of(
            PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY,
            runtime.lifecycle_familiar.refresh_memory_vitality,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.LIFECYCLE_RUN_GARDENING_ONCE,
            runtime.lifecycle_familiar.run_gardening_once,
        ),
        RouteBinding.of(PatchouliLocalRoutes.RUNTIME_MODELS_WARMUP, runtime.warmup_models),
        RouteBinding.of(PatchouliLocalRoutes.RUNTIME_MODELS_READY, models_ready),
        RouteBinding.of(PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH, runtime.check_storage_health),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_RECORD_HIT, runtime.lifecycle_familiar.record_hit
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_RECORD_CITATION,
            runtime.lifecycle_familiar.record_citation,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_RECORD_FEEDBACK,
            runtime.lifecycle_familiar.record_feedback,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.MEMORY_REVIVE, runtime.lifecycle_familiar.revive_memory
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.GET_AGENT_PROFILE,
            runtime.retrieval_familiar.get_agent_profile,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.TOPIC_PREPARE, runtime.perception_familiar.prepare_topic
        ),
        RouteBinding.of(PatchouliLocalRoutes.TOPIC_GET, get_topic),
        RouteBinding.of(
            PatchouliLocalRoutes.TOPIC_LIST_ACTIVE,
            list_active_topics,
        ),
        RouteBinding.of(PatchouliLocalRoutes.TOPIC_EVICT, runtime.perception_familiar.evict_topic),
        RouteBinding.of(
            PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY,
            discard_if_empty,
        ),
        RouteBinding.of(
            PatchouliLocalRoutes.TOPIC_MANUAL_SETTLE,
            runtime.perception_familiar.manual_settle_topic,
        ),
    )


__all__ = ["RouteBinding", "build_patchouli_route_bindings"]
