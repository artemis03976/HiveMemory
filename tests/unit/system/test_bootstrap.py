"""SystemBootstrap 组装闭环测试"""

from unittest.mock import MagicMock, patch

from hivememory.system.bootstrap import SystemBootstrap
from hivememory.system.patchouli_subsystem import PatchouliSubsystemAdapter
from hivememory.system.runtime.bus import GlobalSystemBus


class _FakePatchouliSystem:
    def __init__(self, config, bus):
        self.config = config
        self.bus = bus
        self.kernel = MagicMock()
        self.kernel.is_models_ready.return_value = True
        self.storage = MagicMock()

    def start_scheduler(self):
        pass

    async def shutdown_drain(self):
        return {"success": True}


def test_build_registers_patchouli_and_shares_global_bus():
    config = MagicMock()

    with patch("hivememory.patchouli.system.PatchouliSystem", _FakePatchouliSystem):
        system = SystemBootstrap.build(config=config)

    assert isinstance(system._runtime.bus, GlobalSystemBus)
    assert system.patchouli.bus is system._runtime.bus

    registered = system._runtime.registry.get("patchouli")
    assert isinstance(registered, PatchouliSubsystemAdapter)
    assert registered._patchouli is system.patchouli

    health = system._runtime.registry._subsystems["patchouli"]
    assert health.name == "patchouli"
