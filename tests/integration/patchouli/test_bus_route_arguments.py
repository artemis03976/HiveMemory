"""Patchouli 真实 handler 挂到总线后的参数检查集成测试。

真实协作边界：GlobalSystemBus + PatchouliBridge + 真实 application 服务（公开路由），
以及 PatchouliBus + 真实 MemoryGenerationCoordinator（本地路由）。chat 与模型就绪
入口不在本次边界内，以 MagicMock 占位。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hivememory.components.bus import GlobalSystemBus, RouteArgumentError
from hivememory.patchouli.application import (
    AgentProfileManagementService,
    InteractionSubmissionService,
    MemoryIntentSubmissionService,
    MemoryManagementService,
    MemoryTaskManagementService,
    TopicManagementService,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.patchouli.control.memory_generation.coordinator import (
    MemoryGenerationCoordinator,
)
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
from hivememory.patchouli.runtime.bus import PatchouliBus
from tests.helpers.workspace import make_identity_scope


async def _never_applied(*_args, **_kwargs):
    raise AssertionError("本测试不应触达交互 apply")


def _mount_public_routes() -> GlobalSystemBus:
    local_bus = PatchouliBus()
    global_bus = GlobalSystemBus()
    public_api = PatchouliPublicApi(
        chat=MagicMock(),
        memory=MemoryManagementService(bus=local_bus),
        memory_tasks=MemoryTaskManagementService(bus=local_bus),
        agent_profiles=AgentProfileManagementService(bus=local_bus),
        interactions=InteractionSubmissionService(
            interaction_queue=InteractionSubmissionQueue(_never_applied)
        ),
        memory_intents=MemoryIntentSubmissionService(bus=local_bus),
        topics=TopicManagementService(bus=local_bus),
        readiness=MagicMock(),
    )
    PatchouliBridge(local_bus=local_bus, global_bus=global_bus, public_api=public_api).mount()
    return global_bus


def test_public_route_handlers_have_resolvable_annotations():
    """Patchouli 公开路由的真实 handler 标注都能在运行时解析，参数类型检查全部生效。"""
    global_bus = _mount_public_routes()

    assert global_bus.list_unresolved_routes() == []


@pytest.mark.asyncio
async def test_identity_scope_passed_as_ownership_to_real_local_handler_is_rejected():
    """未迁移的调用方把 IdentityScope 当作归属传给真实本地 handler 时，请求被拒绝。

    协调器在没有物化任务时直接返回空列表、不读取归属；没有总线参数检查时，
    这样的调用会静默成功。
    """
    local_bus = PatchouliBus()
    coordinator = MemoryGenerationCoordinator(bus=local_bus)
    local_bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, coordinator.submit_active)
    scope = make_identity_scope(user_id="u1", agent_id="a1")

    with pytest.raises(RouteArgumentError, match="'belong_to' expects WorkspaceIdentity"):
        await local_bus.request(
            PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
            [],
            "topic-1",
            belong_to=scope,
        )
