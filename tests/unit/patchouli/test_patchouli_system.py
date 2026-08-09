from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.patchouli.system import PatchouliSystem


@pytest.mark.asyncio
async def test_start_ensures_storage_ready_before_mounting_routes():
    system = PatchouliSystem.__new__(PatchouliSystem)
    system.runtime = MagicMock()
    system.runtime.ensure_storage_ready = AsyncMock()
    system.runtime.start_memory_generation_queue = AsyncMock()
    system.runtime.local_routes_registered = False
    system.runtime.mount_local_routes = MagicMock()
    system._service = MagicMock()
    system._bridge = MagicMock()
    system._interaction_submission_queue = MagicMock()
    system._interaction_submission_queue.start = AsyncMock()
    system._scheduler = None
    system._maintenance_registered = False

    await system.start()

    system.runtime.ensure_storage_ready.assert_awaited_once()
    system.runtime.mount_local_routes.assert_called_once_with(system.service)
    system._bridge.mount.assert_called_once()
    system._interaction_submission_queue.start.assert_awaited_once()
    system.runtime.start_memory_generation_queue.assert_awaited_once()
