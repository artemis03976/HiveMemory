from unittest.mock import AsyncMock, Mock, patch

import pytest

from hivememory.patchouli.runtime.core import PatchouliRuntime


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
            },
        ),
        patch.object(PatchouliRuntime, "_register_services"),
    ):
        runtime = PatchouliRuntime(patchouli_config=Mock(), shared_config=Mock())
        runtime._services = {
            "retrieval": Mock(),
            "librarian": Mock(),
        }
        runtime.storage = Mock()
        return runtime


class TestRuntimeShutdownDrain:
    @pytest.mark.asyncio
    async def test_shutdown_drain_flushes_perception_once(self):
        runtime = _create_runtime()
        runtime._services["librarian"].perception_layer = Mock()
        runtime._services["librarian"].perception_layer.flush_all_for_shutdown = AsyncMock(
            return_value={
                "success": True,
                "trigger_reason": "shutdown",
                "flushed_topics": ["t1"],
                "skipped_topics": [],
                "archived_blocks": 1,
            }
        )

        result = await runtime.shutdown_drain()

        runtime._services[
            "librarian"
        ].perception_layer.flush_all_for_shutdown.assert_awaited_once()
        assert result["reentrant"] is False
        assert result["perception"]["trigger_reason"] == "shutdown"

    @pytest.mark.asyncio
    async def test_shutdown_drain_is_reentrant(self):
        runtime = _create_runtime()
        runtime._services["librarian"].perception_layer = Mock()
        runtime._services["librarian"].perception_layer.flush_all_for_shutdown = AsyncMock(
            return_value={
                "success": True,
                "trigger_reason": "shutdown",
                "flushed_topics": [],
                "skipped_topics": [],
                "archived_blocks": 0,
            }
        )

        first = await runtime.shutdown_drain()
        second = await runtime.shutdown_drain()

        assert first["reentrant"] is False
        assert second["reentrant"] is True
        runtime._services[
            "librarian"
        ].perception_layer.flush_all_for_shutdown.assert_awaited_once()


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
