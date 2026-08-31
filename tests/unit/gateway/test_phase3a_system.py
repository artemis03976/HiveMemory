"""Gateway Phase 3A 子系统骨架测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.protocol.gateway import GatewayIngressMode, IntentType
from hivememory.gateway import GatewaySystem
from hivememory.gateway.contracts import GatewayLocalRoutes, GatewayPublicRoutes
from hivememory.gateway.service import GatewayService
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from tests.helpers.workspace import make_identity_scope


@pytest.mark.asyncio
async def test_gateway_system_mount_and_unmount_are_idempotent() -> None:
    global_bus = GlobalSystemBus()
    system = GatewaySystem(HiveMemoryConfig(), global_bus)

    assert isinstance(system, SubsystemProtocol)
    await system.start()
    await system.start()

    assert system.public_routes_registered is True
    assert system.runtime.local_routes_registered is True
    assert GatewayPublicRoutes.PROCESS in global_bus.list_routes()
    assert GatewayLocalRoutes.PROCESS in system.runtime.local_bus.list_routes()

    await system.stop()
    await system.stop()

    assert system.public_routes_registered is False
    assert system.runtime.local_routes_registered is False
    assert GatewayPublicRoutes.PROCESS not in global_bus.list_routes()


@pytest.mark.asyncio
async def test_gateway_service_runs_fallback_workflow() -> None:
    global_bus = GlobalSystemBus()
    system = GatewaySystem(HiveMemoryConfig(), global_bus)
    await system.start()

    result = await global_bus.request(
        GatewayPublicRoutes.PROCESS,
        message="hello",
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert result.decision.intent_type == IntentType.CHAT
    await system.stop()


@pytest.mark.asyncio
async def test_gateway_service_rejects_missing_workspace_scope() -> None:
    """防止 Gateway 中层在缺 scope 时从默认用户隐式补出 Workspace。"""
    workflow = SimpleNamespace(run=AsyncMock())
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            workflow=SimpleNamespace(default_request_timeout_ms=1000)
        ),
        workflow=workflow,
    )

    with pytest.raises(ScopeRequiredError):
        await GatewayService(runtime).process(
            "hello",
            identity_scope=None,
            ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
        )

    workflow.run.assert_not_awaited()
