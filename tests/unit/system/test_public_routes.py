"""公开路由注册/卸载测试 — 验证 System 门面在生命周期中正确管理全局总线路由。"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.runtime.bus import AliceBus
from hivememory.alice.system import AliceSystem
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


# ========== Alice ==========


class TestAlicePublicRoutes:

    def setup_method(self):
        self.global_bus = GlobalSystemBus()
        self.config = MagicMock()
        self.config.koakuma = MagicMock()
        self.config.koakuma.enabled = False
        self.config.llm = MagicMock()
        self.config.llm.worker = MagicMock()

    @pytest.mark.asyncio
    async def test_start_registers_public_routes_on_global_bus(self):
        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        await system.start()

        routes = self.global_bus.list_routes()
        assert AliceRoutes.RUN_AGENT in routes
        assert AliceRoutes.RUN_AGENT_STREAM in routes
        assert AliceRoutes.REGISTER_PRERETRIEVAL_ALIASES in routes
        assert AliceRoutes.GET_INTERACTION_STATE in routes

    @pytest.mark.asyncio
    async def test_stop_removes_public_routes_from_global_bus(self):
        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        await system.start()
        await system.stop()

        routes = self.global_bus.list_routes()
        assert AliceRoutes.RUN_AGENT not in routes
        assert AliceRoutes.RUN_AGENT_STREAM not in routes

    @pytest.mark.asyncio
    async def test_request_through_global_bus_reaches_handler(self):
        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        system._service.run_agent = AsyncMock(return_value="agent_result")
        await system.start()

        result = await self.global_bus.request(
            AliceRoutes.RUN_AGENT,
            messages=[],
            identity="id",
        )

        assert result == "agent_result"
        system._service.run_agent.assert_awaited_once_with(messages=[], identity="id")

    @pytest.mark.asyncio
    async def test_stream_route_returns_async_generator(self):
        system = AliceSystem(config=self.config, global_bus=self.global_bus)

        async def _stream(**kwargs):
            yield {"event": "token"}
            yield {"event": "done"}

        system._service.run_agent_stream = MagicMock(side_effect=_stream)
        await system.start()

        stream = await self.global_bus.request(
            AliceRoutes.RUN_AGENT_STREAM,
            messages=[],
            identity="id",
        )

        events = []
        async for event in stream:
            events.append(event)

        assert [e["event"] for e in events] == ["token", "done"]

    @pytest.mark.asyncio
    async def test_no_global_bus_skips_public_routes(self):
        system = AliceSystem(config=self.config, global_bus=None)
        await system.start()
        await system.stop()


# ========== Patchouli (lightweight — full integration tested in test_bootstrap) ==========


class TestPatchouliPublicRoutes:

    def setup_method(self):
        self.global_bus = GlobalSystemBus()

    @pytest.mark.asyncio
    async def test_public_route_constants_are_consistent(self):
        assert PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE == "patchouli.public.passive.analyze_and_retrieve"
        assert PatchouliRoutes.SUBMIT_INTERACTION == "patchouli.public.submit_interaction"
        assert PatchouliRoutes.MEMORY_RETRIEVE == "patchouli.public.memory.retrieve"
        assert PatchouliRoutes.MEMORY_GET_BY_ALIAS == "patchouli.public.memory.get_memory_by_alias"
        assert PatchouliRoutes.PREPARE_AGENT_RUN == "patchouli.public.prepare_agent_run"
        assert PatchouliRoutes.FINALIZE_AGENT_RUN == "patchouli.public.finalize_agent_run"
        assert PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN == "patchouli.public.cleanup_prepared_agent_run"
        assert AliceRoutes.RUN_AGENT == "alice.public.run_agent"
        assert AliceRoutes.RUN_AGENT_STREAM == "alice.public.run_agent_stream"
        assert AliceRoutes.REGISTER_PRERETRIEVAL_ALIASES == "alice.public.register_preretrieval_aliases"
        assert AliceRoutes.GET_INTERACTION_STATE == "alice.public.get_interaction_state"
