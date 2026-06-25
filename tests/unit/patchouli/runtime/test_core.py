from unittest.mock import AsyncMock, Mock, patch

import pytest

from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.patchouli.services.perception import ShutdownFlushResult


def _create_runtime():
    with (
        patch.object(PatchouliRuntime, "_init_infrastructure"),
        patch.object(
            PatchouliRuntime,
            "_build_engines",
            return_value={
                "perception": Mock(),
                "generation": Mock(),
                "lifecycle": Mock(),
                "retrieval": Mock(),
                "artifact": Mock(),
            },
        ),
        patch.object(PatchouliRuntime, "_register_services"),
    ):
        runtime = PatchouliRuntime(patchouli_config=Mock(), shared_config=Mock())
        runtime._services = {
            "perception": Mock(),
            "retrieval": Mock(),
            "generation": Mock(),
            "generation_coordinator": Mock(),
            "lifecycle": Mock(),
        }
        runtime._task_controller = Mock()
        runtime.storage = Mock()
        runtime.memory_library = Mock()
        runtime.memory_library.mid_term.upsert = Mock()
        runtime.memory_library.mid_term.delete = Mock()
        return runtime


class TestRuntimeShutdownDrain:
    @pytest.mark.asyncio
    async def test_shutdown_drain_flushes_perception_once(self):
        runtime = _create_runtime()
        runtime.perception_familiar.flush_all_for_shutdown = AsyncMock(
            return_value=ShutdownFlushResult(
                success=True,
                trigger_reason="shutdown",
                flushed_topics=["t1"],
                skipped_topics=[],
                archived_blocks=1,
            )
        )

        result = await runtime.shutdown_drain()

        runtime.perception_familiar.flush_all_for_shutdown.assert_awaited_once()
        assert result["reentrant"] is False
        assert result["perception"].trigger_reason == "shutdown"

    @pytest.mark.asyncio
    async def test_shutdown_drain_is_reentrant(self):
        runtime = _create_runtime()
        runtime.perception_familiar.flush_all_for_shutdown = AsyncMock(
            return_value=ShutdownFlushResult(
                success=True,
                trigger_reason="shutdown",
                flushed_topics=[],
                skipped_topics=[],
                archived_blocks=0,
            )
        )

        first = await runtime.shutdown_drain()
        second = await runtime.shutdown_drain()

        assert first["reentrant"] is False
        assert second["reentrant"] is True
        runtime.perception_familiar.flush_all_for_shutdown.assert_awaited_once()


class TestRuntimeLocalRoutes:
    def test_local_routes_exclude_public_workflow_mirrors(self):
        routes = set(PatchouliLocalRoutes.ALL)
        assert not any(route.startswith("patchouli.public.") for route in routes)
        assert not any("prepare_agent_run" in route for route in routes)
        assert not any("finalize_agent_run" in route for route in routes)
        assert not any("cleanup_prepared_agent_run" in route for route in routes)

    def test_mount_local_routes_registers_all_declared_routes(self):
        runtime = _create_runtime()
        service = Mock()

        runtime.mount_local_routes(service)

        assert set(PatchouliLocalRoutes.ALL).issubset(set(runtime.list_local_routes()))

    def test_unmount_local_routes_removes_declared_routes(self):
        runtime = _create_runtime()
        runtime.mount_local_routes(Mock())

        runtime.unmount_local_routes()

        assert set(runtime.list_local_routes()).isdisjoint(set(PatchouliLocalRoutes.ALL))

    @pytest.mark.asyncio
    async def test_missing_local_route_raises_explicit_key_error(self):
        bus = PatchouliBus()

        with pytest.raises(KeyError, match="missing.route"):
            await bus.request("missing.route")


class TestRuntimeStorageHealth:
    @pytest.mark.asyncio
    async def test_check_storage_health_awaits_qdrant_client(self):
        runtime = _create_runtime()
        runtime.storage.client.get_collections = AsyncMock(return_value=Mock())

        assert await runtime.check_storage_health() is True
        runtime.storage.client.get_collections.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_storage_health_returns_false_on_failure(self):
        runtime = _create_runtime()
        runtime.storage.client.get_collections = AsyncMock(side_effect=OSError("down"))

        assert await runtime.check_storage_health() is False
        runtime.storage.client.get_collections.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ensure_storage_ready_delegates_to_store(self):
        runtime = _create_runtime()
        runtime.storage.ensure_ready = AsyncMock()

        await runtime.ensure_storage_ready()

        runtime.storage.ensure_ready.assert_awaited_once()
