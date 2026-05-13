"""
SystemAsyncScheduler 单元测试

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

from hivememory.patchouli.kernel.runtime.maintenance_scheduler import (
    SystemAsyncScheduler,
    MaintenanceTaskSpec,
)


class TestSchedulerRegistration:

    def test_register_task(self):
        scheduler = SystemAsyncScheduler()
        spec = MaintenanceTaskSpec(name="test", interval_seconds=10.0)
        scheduler.register(spec, AsyncMock())
        status = scheduler.get_status()
        assert "test" in status
        assert status["test"]["enabled"] is True
        assert status["test"]["interval_seconds"] == 10.0

    def test_register_overwrites_existing(self):
        scheduler = SystemAsyncScheduler()
        scheduler.register(
            MaintenanceTaskSpec(name="test", interval_seconds=10.0), AsyncMock()
        )
        scheduler.register(
            MaintenanceTaskSpec(name="test", interval_seconds=20.0), AsyncMock()
        )
        status = scheduler.get_status()
        assert status["test"]["interval_seconds"] == 20.0

    def test_unregister_task(self):
        scheduler = SystemAsyncScheduler()
        scheduler.register(
            MaintenanceTaskSpec(name="test", interval_seconds=10.0), AsyncMock()
        )
        assert scheduler.unregister("test") is True
        assert "test" not in scheduler.get_status()

    def test_unregister_nonexistent(self):
        scheduler = SystemAsyncScheduler()
        assert scheduler.unregister("nope") is False

    def test_set_enabled(self):
        scheduler = SystemAsyncScheduler()
        scheduler.register(
            MaintenanceTaskSpec(name="test", interval_seconds=10.0, enabled=True),
            AsyncMock(),
        )
        scheduler.set_enabled("test", False)
        assert scheduler.get_status()["test"]["enabled"] is False

    def test_set_enabled_nonexistent(self):
        scheduler = SystemAsyncScheduler()
        assert scheduler.set_enabled("nope", True) is False


class TestSchedulerExecution:

    @pytest.mark.asyncio
    async def test_task_is_called(self):
        callback = AsyncMock()
        scheduler = SystemAsyncScheduler(tick_seconds=0.05)
        scheduler.register(
            MaintenanceTaskSpec(name="test", interval_seconds=0.1),
            callback,
        )
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
        scheduler = SystemAsyncScheduler(tick_seconds=0.05)
        scheduler.register(
            MaintenanceTaskSpec(name="test", interval_seconds=0.1, enabled=False),
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
        scheduler = SystemAsyncScheduler(tick_seconds=0.05)
        scheduler.register(
            MaintenanceTaskSpec(name="failing", interval_seconds=0.1),
            failing,
        )
        scheduler.register(
            MaintenanceTaskSpec(name="healthy", interval_seconds=0.1),
            healthy,
        )
        for state in scheduler._tasks.values():
            state.next_run_at = 0.0

        scheduler.start()
        await asyncio.sleep(0.25)
        await scheduler.stop()

        assert failing.call_count >= 1
        assert healthy.call_count >= 1
        status = scheduler.get_status()
        assert status["failing"]["failure_count"] >= 1
        assert status["failing"]["last_error"] == "boom"

    @pytest.mark.asyncio
    async def test_non_reentrant_skip(self):
        """长时间运行的任务不会重入"""
        call_count = 0

        async def slow_task():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.3)

        scheduler = SystemAsyncScheduler(tick_seconds=0.05)
        scheduler.register(
            MaintenanceTaskSpec(
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
        assert status["slow"]["skip_count"] >= 1


class TestSchedulerLifecycle:

    def test_start_without_running_loop_raises_clear_error(self):
        scheduler = SystemAsyncScheduler()
        with pytest.raises(
            RuntimeError,
            match="必须在运行中的 asyncio 事件循环内调用",
        ):
            scheduler.start()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        scheduler = SystemAsyncScheduler()
        assert scheduler.is_running is False
        scheduler.start()
        assert scheduler.is_running is True
        await scheduler.stop()
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self):
        scheduler = SystemAsyncScheduler()
        scheduler.start()
        scheduler.start()
        assert scheduler.is_running is True
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self):
        scheduler = SystemAsyncScheduler()
        await scheduler.stop()
