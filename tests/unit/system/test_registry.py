"""SubsystemRegistry 测试"""

import pytest
import pytest_asyncio

from hivememory.system.runtime.registry import SubsystemRegistry


class FakeSubsystem:
    def __init__(self, name: str):
        self._name = name
        self.started = False
        self.stopped = False

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def health(self) -> dict:
        return {"status": "ok", "name": self._name}


@pytest.fixture
def registry():
    return SubsystemRegistry()


class TestSubsystemRegistry:
    def test_register_and_get(self, registry: SubsystemRegistry):
        sub = FakeSubsystem("alpha")
        registry.register(sub)
        assert registry.get("alpha") is sub

    def test_get_nonexistent_returns_none(self, registry: SubsystemRegistry):
        assert registry.get("nonexistent") is None

    def test_all_returns_registered(self, registry: SubsystemRegistry):
        a = FakeSubsystem("a")
        b = FakeSubsystem("b")
        registry.register(a)
        registry.register(b)
        assert registry.all() == [a, b]

    @pytest.mark.asyncio
    async def test_start_all(self, registry: SubsystemRegistry):
        a = FakeSubsystem("a")
        b = FakeSubsystem("b")
        registry.register(a)
        registry.register(b)
        await registry.start_all()
        assert a.started and b.started

    @pytest.mark.asyncio
    async def test_stop_all_reverse_order(self, registry: SubsystemRegistry):
        order = []

        class TrackedSubsystem(FakeSubsystem):
            async def stop(self):
                order.append(self._name)

        a = TrackedSubsystem("a")
        b = TrackedSubsystem("b")
        registry.register(a)
        registry.register(b)
        await registry.stop_all()
        assert order == ["b", "a"]

    @pytest.mark.asyncio
    async def test_health_all(self, registry: SubsystemRegistry):
        a = FakeSubsystem("a")
        b = FakeSubsystem("b")
        registry.register(a)
        registry.register(b)
        health = await registry.health_all()
        assert health == {
            "a": {"status": "ok", "name": "a"},
            "b": {"status": "ok", "name": "b"},
        }
