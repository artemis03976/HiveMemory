import pytest

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.system import AliceSystem
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


@pytest.mark.asyncio
async def test_start_registers_public_routes_and_stop_unregisters():
    bus = GlobalSystemBus()
    system = AliceSystem(config=HiveMemoryConfig(), global_bus=bus)

    await system.start()

    assert AliceRoutes.RUN_AGENT in bus.list_routes()
    assert AliceRoutes.RUN_AGENT_STREAM in bus.list_routes()

    await system.stop()

    assert AliceRoutes.RUN_AGENT not in bus.list_routes()
    assert AliceRoutes.RUN_AGENT_STREAM not in bus.list_routes()


@pytest.mark.asyncio
async def test_start_is_idempotent_for_public_routes():
    bus = GlobalSystemBus()
    system = AliceSystem(config=HiveMemoryConfig(), global_bus=bus)

    await system.start()
    await system.start()

    assert bus.list_routes().count(AliceRoutes.RUN_AGENT) == 1
    await system.stop()


@pytest.mark.asyncio
async def test_health_reports_runtime_health():
    system = AliceSystem(config=HiveMemoryConfig())

    health = await system.health()

    assert health["status"] == "ok"
    assert "runtime" in health
