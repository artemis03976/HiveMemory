from unittest.mock import AsyncMock, MagicMock, call

import pytest

from hivememory.patchouli.system import PatchouliSystem


def _system_with_mocked_runtime() -> PatchouliSystem:
    system = PatchouliSystem.__new__(PatchouliSystem)
    system.runtime = MagicMock()
    system.runtime.ensure_storage_ready = AsyncMock()
    system.runtime.restore_topic_pool = AsyncMock()
    system.runtime.start_memory_generation_queue = AsyncMock()
    system.runtime.restore_saved_generation = AsyncMock()
    system.runtime.local_routes_registered = False
    system.runtime.mount_local_routes = MagicMock()
    system._service = MagicMock()
    system._bridge = MagicMock()
    system._interaction_submission_queue = MagicMock()
    system._interaction_submission_queue.start = AsyncMock()
    system._scheduler = None
    system._maintenance_registered = False
    return system


@pytest.mark.asyncio
async def test_start_ensures_storage_ready_before_mounting_routes():
    system = _system_with_mocked_runtime()

    await system.start()

    system.runtime.ensure_storage_ready.assert_awaited_once()
    system.runtime.mount_local_routes.assert_called_once_with(system.service)
    system._bridge.mount.assert_called_once()
    system._interaction_submission_queue.start.assert_awaited_once()
    system.runtime.start_memory_generation_queue.assert_awaited_once()
    system.runtime.restore_saved_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_restores_topic_pool_before_interactions_and_saved_generation():
    """话题池先于公开路由、交互提交与保存的生成工作读回：读回的结算执行时它的话题已在存储里，
    交互与 Gateway 看到的是关闭前的话题池。"""
    system = _system_with_mocked_runtime()
    steps = MagicMock()
    steps.attach_mock(system.runtime.restore_topic_pool, "restore_topic_pool")
    steps.attach_mock(system._bridge.mount, "mount_public_routes")
    steps.attach_mock(system._interaction_submission_queue.start, "start_interactions")
    steps.attach_mock(system.runtime.restore_task_history, "restore_task_history")
    steps.attach_mock(system.runtime.restore_saved_generation, "restore_saved_generation")

    await system.start()

    assert steps.mock_calls == [
        call.restore_topic_pool(),
        call.mount_public_routes(),
        call.start_interactions(),
        call.restore_task_history(),
        call.restore_saved_generation(),
    ]
