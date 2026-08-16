"""SubsystemProtocol 协议检查测试"""

import pytest
from unittest.mock import MagicMock

from hivememory.patchouli.system import PatchouliSystem
from hivememory.system.contracts.subsystem import SubsystemProtocol
from hivememory.system.contracts.events import SystemEvent, SystemEventType


class _ValidSubsystem:
    @property
    def name(self) -> str:
        return "test"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health(self) -> dict:
        return {"status": "ok"}


class _InvalidSubsystem:
    pass


class TestSubsystemProtocol:
    def test_valid_subsystem_passes_isinstance(self):
        sub = _ValidSubsystem()
        assert isinstance(sub, SubsystemProtocol)

    def test_patchouli_system_passes_isinstance(self):
        sub = PatchouliSystem.__new__(PatchouliSystem)
        sub.runtime = MagicMock()
        sub.runtime.is_models_ready.return_value = True
        assert isinstance(sub, SubsystemProtocol)

    def test_invalid_subsystem_fails_isinstance(self):
        sub = _InvalidSubsystem()
        assert not isinstance(sub, SubsystemProtocol)


class TestSystemEvent:
    def test_event_is_frozen(self):
        from dataclasses import FrozenInstanceError

        event = SystemEvent(event_type=SystemEventType.SYSTEM_READY)
        with pytest.raises(FrozenInstanceError):
            event.event_type = SystemEventType.SYSTEM_SHUTTING_DOWN
