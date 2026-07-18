"""Gateway Phase 3A 子系统骨架测试。"""

from unittest.mock import AsyncMock

import pytest

from hivememory.core.models import Identity
from hivememory.gateway import GatewayService, GatewaySystem
from hivememory.gateway.contracts import GatewayLocalRoutes, GatewayPublicRoutes
from hivememory.gateway.runtime import GatewayRuntime
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


@pytest.mark.asyncio
async def test_gateway_service_delegates_to_workflow() -> None:
    workflow = AsyncMock()
    workflow.run.return_value = "ok"
    runtime = GatewayRuntime(
        config=HiveMemoryConfig().gateway,
        global_bus=GlobalSystemBus(),
        workflow=workflow,
    )
    service = GatewayService(runtime)
    identity = Identity(user_id="u1", agent_id="a1")

    result = await service.process("hello", identity=identity)

    assert result == "ok"
    workflow.run.assert_awaited_once_with("hello", identity=identity)


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
async def test_gateway_service_runs_minimal_empty_workflow() -> None:
    global_bus = GlobalSystemBus()
    system = GatewaySystem(HiveMemoryConfig(), global_bus)
    await system.start()

    result = await global_bus.request(
        GatewayPublicRoutes.PROCESS,
        message="hello",
        identity=Identity(user_id="u1", agent_id="a1"),
    )

    assert result is None
    await system.stop()
