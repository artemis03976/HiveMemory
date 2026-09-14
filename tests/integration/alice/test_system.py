"""
AliceSystem 集成测试 — 真实 System + GlobalSystemBus 协作

驱动真实 AliceSystem（内部真实装配 AliceRuntime/AliceBridge/AgentRunService）
+ 真实 GlobalSystemBus，零 mock；验证系统 facade start()/stop() 对全局总线的
路由挂载与卸载副作用。
"""

from uuid import uuid4

import pytest

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.system import AliceSystem
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_workspace_identity


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
async def test_stop_clears_runtime_derived_caches():
    """AliceSystem.stop 在 bridge 卸载后清空执行路径派生 cache（ADR-0005）。"""
    system = AliceSystem(config=HiveMemoryConfig())
    atom = MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test"),
        index=IndexLayer(
            title="Stop Memory",
            summary="Stop summary",
            memory_type=MemoryType.FACT,
            alias="fact_stop",
        ),
        payload=PayloadLayer(content="stop"),
    )
    system.runtime.atom_cache.ingest_atom(
        atom,
        workspace_identity=make_workspace_identity(),
    )

    await system.start()
    await system.stop()

    assert system.runtime.atom_cache.get_atom_by_alias(
        "fact_stop",
        workspace_identity=make_workspace_identity(),
    ) is None
    assert system.runtime.atom_cache.get_atom_by_uuid(str(atom.id)) is None
