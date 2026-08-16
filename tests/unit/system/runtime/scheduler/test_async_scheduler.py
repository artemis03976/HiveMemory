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


class _FakeMonotonic:
    """可控单调时钟：读取返回当前值，测试显式推进。

    AsyncMaintenanceScheduler 以 `from time import monotonic` 模块级导入，
    通过 monkeypatch `async_scheduler.monotonic` 即可让「interval 到期判定」
    由假时钟控制：注册后 next_run_at = 0 + interval，推进时钟即可触发 dispatch，
    不再依赖真实时间等待（消除固定 sleep 与机器负载导致的 flaky）。
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture
def fake_clock(monkeypatch) -> _FakeMonotonic:
    clock = _FakeMonotonic()
    monkeypatch.setattr(
        "hivememory.system.runtime.scheduler.async_scheduler.monotonic",
        clock,
    )
    return clock


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
    async def test_task_is_called(self, fake_clock):
        callback = AsyncMock()
        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.01)
        scheduler.register(make_spec(name="test", interval_seconds=1.0), callback)
        fake_clock.advance(1.0)  # 推进时钟使任务到期，无需真实等待

        scheduler.start()
        try:
            await _wait_for_call_count(callback, 1)
        finally:
            await scheduler.stop()

        assert callback.call_count >= 1

    @pytest.mark.asyncio
    async def test_disabled_task_not_called(self, fake_clock):
        callback = AsyncMock()
        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.01)
        scheduler.register(
            make_spec(name="test", interval_seconds=1.0, enabled=False),
            callback,
        )
        # 假时钟覆盖 3 个 interval 周期：disabled 任务永不触发
        fake_clock.advance(3.0)

        scheduler.start()
        # 真实等待仅需覆盖几个 tick（10ms 级），保证 tick 循环已跨过多周期
        await asyncio.sleep(0.05)
        await scheduler.stop()

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_does_not_crash_scheduler(self, fake_clock):
        failing = AsyncMock(side_effect=RuntimeError("boom"))
        healthy = AsyncMock()
        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.01)
        scheduler.register(make_spec(name="failing", interval_seconds=1.0), failing)
        scheduler.register(make_spec(name="healthy", interval_seconds=1.0), healthy)
        fake_clock.advance(1.0)

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
    async def test_non_reentrant_skip(self, fake_clock):
        """长时间运行的任务不会重入"""
        call_count = 0

        async def slow_task():
            nonlocal call_count
            call_count += 1
            # 模拟耗时任务：必须真实占用事件循环，时钟无法替代
            await asyncio.sleep(0.1)

        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.01)
        scheduler.register(
            make_spec(
                name="slow",
                interval_seconds=0.5,
                non_reentrant=True,
                skip_if_running=True,
            ),
            slow_task,
        )
        fake_clock.advance(0.5)  # 第一次到期 → dispatch slow_task

        scheduler.start()
        try:
            # 等 slow_task 被派发（call_count >= 1）
            async with asyncio.timeout(1.0):
                while call_count < 1:
                    await asyncio.sleep(0.01)
            # 任务运行中再次到期 → 应 skip 而非重入
            fake_clock.advance(0.5)
            await _wait_for_skip_count(scheduler, f"{TEST_OWNER}.slow", 1)
            await scheduler.stop()
        except asyncio.TimeoutError:
            await scheduler.stop()
            raise

        assert call_count <= 2
        status = scheduler.get_status()
        assert status[f"{TEST_OWNER}.slow"]["skip_count"] >= 1

    @pytest.mark.asyncio
    async def test_success_task_emits_runtime_events(self, fake_clock):
        recorder = RecordingRuntimeEventSink()

        async def callback():
            return {"flushed": 2}

        scheduler = AsyncMaintenanceScheduler(
            tick_seconds=0.01,
            runtime_events=recorder,
        )
        scheduler.register(make_spec(name="success", interval_seconds=1.0), callback)
        fake_clock.advance(1.0)

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
    async def test_failed_task_emits_runtime_event_and_keeps_scheduler_alive(self, fake_clock):
        recorder = RecordingRuntimeEventSink()

        async def callback():
            raise RuntimeError("boom")

        scheduler = AsyncMaintenanceScheduler(
            tick_seconds=0.01,
            runtime_events=recorder,
        )
        scheduler.register(make_spec(name="failing", interval_seconds=1.0), callback)
        fake_clock.advance(1.0)

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
    async def test_disabled_task_emits_no_runtime_events(self, fake_clock):
        recorder = RecordingRuntimeEventSink()
        callback = AsyncMock()
        scheduler = AsyncMaintenanceScheduler(
            tick_seconds=0.01,
            runtime_events=recorder,
        )
        scheduler.register(
            make_spec(name="disabled", interval_seconds=1.0, enabled=False),
            callback,
        )
        # 假时钟覆盖 3 个 interval 周期：disabled 任务永不触发
        fake_clock.advance(3.0)

        scheduler.start()
        # 真实等待仅需覆盖几个 tick，保证 tick 循环已跨过多周期
        await asyncio.sleep(0.05)
        await scheduler.stop()

        assert recorder.events == []

    @pytest.mark.asyncio
    async def test_non_reentrant_skip_emits_no_extra_runtime_events(self, fake_clock):
        recorder = RecordingRuntimeEventSink()

        async def slow_task():
            # 模拟耗时任务：必须真实占用事件循环，时钟无法替代
            await asyncio.sleep(0.1)

        scheduler = AsyncMaintenanceScheduler(
            tick_seconds=0.01,
            runtime_events=recorder,
        )
        scheduler.register(
            make_spec(
                name="slow_events",
                interval_seconds=0.5,
                non_reentrant=True,
                skip_if_running=True,
            ),
            slow_task,
        )
        fake_clock.advance(0.5)  # 第一次到期 → dispatch slow_task

        scheduler.start()
        try:
            # 等任务已派发（产生 STARTED 事件）后，任务运行中再次到期 → skip
            await _wait_for_event(
                recorder, RuntimeEventType.MAINTENANCE_TASK_STARTED
            )
            fake_clock.advance(0.5)
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
    async def test_default_null_runtime_event_sink_does_not_affect_task(self, fake_clock):
        scheduler = AsyncMaintenanceScheduler(tick_seconds=0.01)
        callback = AsyncMock(return_value=1)
        scheduler.register(make_spec(name="default_sink", interval_seconds=1.0), callback)
        fake_clock.advance(1.0)

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
