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
from typing import Any
import pytest
from unittest.mock import AsyncMock

from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink
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


async def _wait_for_call_count(callback: AsyncMock, count: int, timeout: float = 1.0) -> None:
    """有界轮询等待 mock 被调用 count 次，避免固定 sleep 竞态。"""
    async with asyncio.timeout(timeout):
        while callback.call_count < count:
            await asyncio.sleep(0.01)


async def _wait_for_event(
    recorder: RecordingRuntimeEventSink,
    event_type: RuntimeEventType,
    timeout: float = 1.0,
) -> Any:
    """有界轮询等待指定观测事件出现。"""
    async with asyncio.timeout(timeout):
        while True:
            for event in recorder.events:
                if event.event_type == event_type:
                    return event
            await asyncio.sleep(0.01)


async def _wait_for_skip_count(
    scheduler: AsyncMaintenanceScheduler,
    task_key: str,
    count: int,
    timeout: float = 1.0,
) -> None:
    """有界轮询等待任务 skip_count 达到阈值。"""
    async with asyncio.timeout(timeout):
        while True:
            status = scheduler.get_status()
            if status[task_key]["skip_count"] >= count:
                return
            await asyncio.sleep(0.01)


class TestSchedulerRegistration:

    def test_register_task(self):
        scheduler = AsyncMaintenanceScheduler()
        spec = make_spec(name="test")
        scheduler.register(spec, AsyncMock())
        status = scheduler.get_status()
        assert spec.task_key in status

    def test_register_overwrites_existing(self):
        scheduler = AsyncMaintenanceScheduler()
        scheduler.register(make_spec(name="test", interval_seconds=10.0), AsyncMock())
        scheduler.register(make_spec(name="test", interval_seconds=20.0), AsyncMock())
        status = scheduler.get_status()
        # 覆盖注册后只保留一个任务，且采用最新 spec 的间隔
        assert len(status) == 1
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

        scheduler.start()
        try:
            await _wait_for_call_count(callback, 1)
        finally:
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

        scheduler.start()
        # 覆盖两个 interval 周期后仍未调用（disabled 任务永不触发，不 flaky）
        await asyncio.sleep(0.25)
        await scheduler.stop()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_does_not_crash_scheduler(self):
        failing = AsyncMock(side_effect=RuntimeError("boom"))
        healthy = AsyncMock()
        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.05)
        scheduler.register(make_spec(name="failing", interval_seconds=0.1), failing)
        scheduler.register(make_spec(name="healthy", interval_seconds=0.1), healthy)

        scheduler.start()
        try:
            await _wait_for_call_count(failing, 1)
            await _wait_for_call_count(healthy, 1)
        finally:
            await scheduler.stop()

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

        scheduler.start()
        try:
            await _wait_for_skip_count(scheduler, f"{TEST_OWNER}.slow", 1)
            await scheduler.stop()
        except asyncio.TimeoutError:
            await scheduler.stop()
            raise

        assert call_count <= 2
        status = scheduler.get_status()
        assert status[f"{TEST_OWNER}.slow"]["skip_count"] >= 1

    @pytest.mark.asyncio
    async def test_success_task_emits_runtime_events(self):
        recorder = RecordingRuntimeEventSink()

        async def callback():
            return {"flushed": 2}

        scheduler = AsyncMaintenanceScheduler(
            tick_seconds=0.05,
            runtime_events=recorder,
        )
        scheduler.register(make_spec(name="success", interval_seconds=0.05), callback)

        scheduler.start()
        try:
            completed = await _wait_for_event(
                recorder, RuntimeEventType.MAINTENANCE_TASK_COMPLETED
            )
        finally:
            await scheduler.stop()

        assert completed.status == "completed"
        assert completed.data["task_key"] == f"{TEST_OWNER}.success"
        assert completed.data["result"] == {"flushed": 2}

    @pytest.mark.asyncio
    async def test_failed_task_emits_runtime_event_and_keeps_scheduler_alive(self):
        recorder = RecordingRuntimeEventSink()

        async def callback():
            raise RuntimeError("boom")

        scheduler = AsyncMaintenanceScheduler(
            tick_seconds=0.05,
            runtime_events=recorder,
        )
        scheduler.register(make_spec(name="failing", interval_seconds=0.05), callback)

        scheduler.start()
        try:
            failed = await _wait_for_event(
                recorder, RuntimeEventType.MAINTENANCE_TASK_FAILED
            )
        finally:
            await scheduler.stop()

        assert failed.status == "failed"
        assert failed.severity == "error"
        assert failed.reason == "boom"
        assert failed.data["error"] == "boom"
        assert failed.data["failure_count"] == 1
        assert scheduler.get_status()[f"{TEST_OWNER}.failing"]["last_error"] == "boom"

    @pytest.mark.asyncio
    async def test_disabled_task_emits_no_runtime_events(self):
        recorder = RecordingRuntimeEventSink()
        callback = AsyncMock()
        scheduler = AsyncMaintenanceScheduler(
            tick_seconds=0.05,
            runtime_events=recorder,
        )
        scheduler.register(
            make_spec(name="disabled", interval_seconds=0.1, enabled=False),
            callback,
        )

        scheduler.start()
        # 覆盖两个 interval 周期后仍无事件（disabled 任务永不触发）
        await asyncio.sleep(0.25)
        await scheduler.stop()

        assert recorder.events == []

    @pytest.mark.asyncio
    async def test_non_reentrant_skip_emits_no_extra_runtime_events(self):
        recorder = RecordingRuntimeEventSink()

        async def slow_task():
            await asyncio.sleep(0.3)

        scheduler = AsyncMaintenanceScheduler(
            tick_seconds=0.05,
            runtime_events=recorder,
        )
        scheduler.register(
            make_spec(
                name="slow_events",
                interval_seconds=0.05,
                non_reentrant=True,
                skip_if_running=True,
            ),
            slow_task,
        )

        scheduler.start()
        try:
            await _wait_for_skip_count(scheduler, f"{TEST_OWNER}.slow_events", 1)
            await scheduler.stop()
        except asyncio.TimeoutError:
            await scheduler.stop()
            raise

        event_types = [event.event_type for event in recorder.events]
        # skip 不产生额外 started 事件：每次运行都是成对的 STARTED/COMPLETED
        assert event_types.count(RuntimeEventType.MAINTENANCE_TASK_STARTED) == event_types.count(
            RuntimeEventType.MAINTENANCE_TASK_COMPLETED
        )
        assert event_types.count(RuntimeEventType.MAINTENANCE_TASK_STARTED) >= 1

    @pytest.mark.asyncio
    async def test_default_null_runtime_event_sink_does_not_affect_task(self):
        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.05)
        callback = AsyncMock(return_value=1)
        scheduler.register(make_spec(name="default_sink", interval_seconds=0.05), callback)

        scheduler.start()
        try:
            await _wait_for_call_count(callback, 1)
        finally:
            await scheduler.stop()

        assert scheduler.get_status()[f"{TEST_OWNER}.default_sink"]["run_count"] == 1


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
