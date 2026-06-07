"""
AsyncMaintenanceScheduler 单元测试

测试覆盖:
- 任务注册与注销
- 调度循环执行与非重入保护
- 启停生命周期
- 异常隔离
- 内省状态
"""

import asyncio
import pytest
from unittest.mock import AsyncMock

from hivememory.system.runtime.scheduler.async_scheduler import AsyncMaintenanceScheduler
from hivememory.system.runtime.scheduler.models import MaintenanceTaskSpec

TEST_OWNER = "test_owner"


def make_spec(name: str, **kwargs) -> MaintenanceTaskSpec:
    return MaintenanceTaskSpec(
        owner=TEST_OWNER,
        name=name,
        interval_seconds=kwargs.pop("interval_seconds", 10.0),
        **kwargs,
    )


class TestSchedulerRegistration:

    def test_register_task(self):
        scheduler = AsyncMaintenanceScheduler()
        spec = make_spec(name="test")
        scheduler.register(spec, AsyncMock())
        status = scheduler.get_status()
        assert spec.task_key in status
        assert status[spec.task_key]["owner"] == TEST_OWNER
        assert status[spec.task_key]["name"] == "test"
        assert status[spec.task_key]["enabled"] is True
        assert status[spec.task_key]["interval_seconds"] == 10.0

    def test_register_overwrites_existing(self):
        scheduler = AsyncMaintenanceScheduler()
        scheduler.register(make_spec(name="test", interval_seconds=10.0), AsyncMock())
        scheduler.register(make_spec(name="test", interval_seconds=20.0), AsyncMock())
        status = scheduler.get_status()
        assert status[f"{TEST_OWNER}.test"]["interval_seconds"] == 20.0

    def test_unregister_task(self):
        scheduler = AsyncMaintenanceScheduler()
        spec = make_spec(name="test")
        scheduler.register(spec, AsyncMock())
        assert scheduler.unregister(spec.task_key) is True
        assert spec.task_key not in scheduler.get_status()

    def test_unregister_nonexistent(self):
        scheduler = AsyncMaintenanceScheduler()
        assert scheduler.unregister("nope") is False

    def test_set_enabled(self):
        scheduler = AsyncMaintenanceScheduler()
        spec = make_spec(name="test", enabled=True)
        scheduler.register(spec, AsyncMock())
        scheduler.set_enabled(spec.task_key, False)
        assert scheduler.get_status()[spec.task_key]["enabled"] is False

    def test_set_enabled_nonexistent(self):
        scheduler = AsyncMaintenanceScheduler()
        assert scheduler.set_enabled("nope", True) is False


class TestSchedulerExecution:

    @pytest.mark.asyncio
    async def test_task_is_called(self):
        callback = AsyncMock()
        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.05)
        scheduler.register(make_spec(name="test", interval_seconds=0.1), callback)
        # Force immediate first run
        for state in scheduler._tasks.values():
            state.next_run_at = 0.0

        scheduler.start()
        await asyncio.sleep(0.2)
        await scheduler.stop()

        assert callback.call_count >= 1

    @pytest.mark.asyncio
    async def test_disabled_task_not_called(self):
        callback = AsyncMock()
        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.05)
        scheduler.register(
            make_spec(name="test", interval_seconds=0.1, enabled=False),
            callback,
        )
        for state in scheduler._tasks.values():
            state.next_run_at = 0.0

        scheduler.start()
        await asyncio.sleep(0.2)
        await scheduler.stop()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_does_not_crash_scheduler(self):
        failing = AsyncMock(side_effect=RuntimeError("boom"))
        healthy = AsyncMock()
        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.05)
        scheduler.register(make_spec(name="failing", interval_seconds=0.1), failing)
        scheduler.register(make_spec(name="healthy", interval_seconds=0.1), healthy)
        for state in scheduler._tasks.values():
            state.next_run_at = 0.0

        scheduler.start()
        await asyncio.sleep(0.25)
        await scheduler.stop()

        assert failing.call_count >= 1
        assert healthy.call_count >= 1
        status = scheduler.get_status()
        assert status[f"{TEST_OWNER}.failing"]["failure_count"] >= 1
        assert status[f"{TEST_OWNER}.failing"]["last_error"] == "boom"

    @pytest.mark.asyncio
    async def test_non_reentrant_skip(self):
        """长时间运行的任务不会重入"""
        call_count = 0

        async def slow_task():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.3)

        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.05)
        scheduler.register(
            make_spec(
                name="slow",
                interval_seconds=0.05,
                non_reentrant=True,
                skip_if_running=True,
            ),
            slow_task,
        )
        for state in scheduler._tasks.values():
            state.next_run_at = 0.0

        scheduler.start()
        await asyncio.sleep(0.4)
        await scheduler.stop()

        assert call_count <= 2
        status = scheduler.get_status()
        assert status[f"{TEST_OWNER}.slow"]["skip_count"] >= 1


class TestSchedulerLifecycle:

    def test_start_without_running_loop_raises_clear_error(self):
        scheduler = AsyncMaintenanceScheduler()
        with pytest.raises(
            RuntimeError,
            match="must be called within a running asyncio event loop",
        ):
            scheduler.start()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        scheduler = AsyncMaintenanceScheduler()
        assert scheduler.is_running is False
        scheduler.start()
        assert scheduler.is_running is True
        await scheduler.stop()
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self):
        scheduler = AsyncMaintenanceScheduler()
        scheduler.start()
        scheduler.start()
        assert scheduler.is_running is True
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self):
        scheduler = AsyncMaintenanceScheduler()
        await scheduler.stop()
