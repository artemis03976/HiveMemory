"""SingletonModelService：模型的按需加载、空闲卸载、加载失败重试与就绪判定。"""

import asyncio
import itertools
import logging
import threading
from collections.abc import Callable

import pytest
import pytest_asyncio

from hivememory.config.shared import EmbeddingConfig
from hivememory.infrastructure.model_service import ModelLoadState, SingletonModelService

# 等待线程侧事件的上限；正常路径远小于此值，只在实现出错时才会耗尽。
_WAIT_SECONDS = 5.0
# 断言“某件事没有发生”时的观察窗口。
_ABSENCE_WINDOW_SECONDS = 0.2
_IDLE_TIMEOUT_SECONDS = 60.0

_model_names = itertools.count()


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeModel:
    """加载器每次加载产生一个新对象，测试据此区分是否重新加载。"""


class _FakeModelService(SingletonModelService[_FakeModel]):
    """以可替换的加载函数代替真实模型文件。"""

    def _load_model(self) -> _FakeModel:
        self.load_threads.append(threading.get_ident())
        return self.loader()

    async def infer[T](self, func: Callable[[_FakeModel], T]) -> T:
        return await self._run_with_model(func)


@pytest_asyncio.fixture
async def make_service():
    """以独立模型名创建实例；结束时释放推理线程并移出 multiton 实例表。"""
    services: list[_FakeModelService] = []

    def _make(
        *,
        residency: str = "idle_unload",
        loader: Callable[[], _FakeModel] = _FakeModel,
    ) -> tuple[_FakeModelService, _FakeClock]:
        clock = _FakeClock()
        config = EmbeddingConfig(
            model_name=f"test/model-{next(_model_names)}",
            residency=residency,
            idle_timeout_seconds=_IDLE_TIMEOUT_SECONDS,
        )
        service = _FakeModelService(config, clock=clock)
        service.loader = loader
        service.load_threads = []
        services.append(service)
        return service, clock

    yield _make
    for service in services:
        await service.close()
        SingletonModelService._instances.pop((_FakeModelService, service.model_name), None)


@pytest.mark.asyncio
async def test_load_and_inference_run_off_the_event_loop_thread(make_service):
    """未加载时的加载与推理都在推理线程中执行，不占用事件循环线程。"""
    service, _ = make_service()

    inference_thread = await service.infer(lambda _model: threading.get_ident())

    assert service.load_threads == [inference_thread]
    assert inference_thread != threading.get_ident()


@pytest.mark.asyncio
async def test_idle_unload_unloads_only_after_idle_timeout(make_service):
    """空闲时间按注入时钟从最近一次使用结束起算，超过时限才卸载。"""
    service, clock = make_service()
    await service.load()
    clock.advance(_IDLE_TIMEOUT_SECONDS - 1)
    await service.infer(lambda _model: None)

    clock.advance(_IDLE_TIMEOUT_SECONDS - 1)
    await service.unload_if_idle()
    assert service.state is ModelLoadState.LOADED

    clock.advance(2)
    await service.unload_if_idle()
    assert service.state is ModelLoadState.UNLOADED


@pytest.mark.asyncio
async def test_resident_model_is_never_unloaded(make_service):
    service, clock = make_service(residency="resident")
    await service.load()

    clock.advance(_IDLE_TIMEOUT_SECONDS * 100)
    await service.unload_if_idle()

    assert service.state is ModelLoadState.LOADED


@pytest.mark.asyncio
async def test_inference_after_unload_reloads_model(make_service):
    service, clock = make_service()
    first = await service.infer(lambda model: model)
    clock.advance(_IDLE_TIMEOUT_SECONDS)
    await service.unload_if_idle()

    second = await service.infer(lambda model: model)

    assert second is not first
    assert len(service.load_threads) == 2
    assert service.state is ModelLoadState.LOADED


@pytest.mark.asyncio
async def test_model_with_inference_in_flight_is_not_unloaded(make_service):
    """推理进行中即使已超过空闲时限也不卸载，推理结束后重新计时。"""
    service, clock = make_service()
    await service.load()
    started = threading.Event()
    release = threading.Event()

    def blocking_inference(_model: _FakeModel) -> None:
        started.set()
        release.wait(_WAIT_SECONDS)

    inference = asyncio.create_task(service.infer(blocking_inference))
    assert await asyncio.to_thread(started.wait, _WAIT_SECONDS)
    clock.advance(_IDLE_TIMEOUT_SECONDS)

    await service.unload_if_idle()
    release.set()
    await inference

    assert service.state is ModelLoadState.LOADED
    assert len(service.load_threads) == 1


@pytest.mark.asyncio
async def test_unload_waits_for_inference_whose_caller_was_cancelled(make_service):
    """调用方取消后推理仍在线程中跑完；卸载排在它之后，不在推理进行中释放模型。"""
    service, clock = make_service()
    await service.load()
    started = threading.Event()
    release = threading.Event()
    observed_states: list[ModelLoadState] = []

    def blocking_inference(_model: _FakeModel) -> None:
        started.set()
        release.wait(_WAIT_SECONDS)
        observed_states.append(service.state)

    inference = asyncio.create_task(service.infer(blocking_inference))
    assert await asyncio.to_thread(started.wait, _WAIT_SECONDS)
    inference.cancel()
    with pytest.raises(asyncio.CancelledError):
        await inference
    clock.advance(_IDLE_TIMEOUT_SECONDS)

    unload = asyncio.create_task(service.unload_if_idle())
    done, _ = await asyncio.wait({unload}, timeout=_ABSENCE_WINDOW_SECONDS)
    assert not done
    release.set()
    await unload

    assert observed_states == [ModelLoadState.LOADED]
    assert service.state is ModelLoadState.UNLOADED


@pytest.mark.asyncio
async def test_failed_load_is_retried_by_next_inference(make_service):
    """加载失败时调用方收到加载错误并进入失败状态，下一次推理重新加载。"""
    attempts = itertools.count()

    def flaky_loader() -> _FakeModel:
        if next(attempts) == 0:
            raise OSError("model file missing")
        return _FakeModel()

    service, _ = make_service(loader=flaky_loader)

    with pytest.raises(OSError, match="model file missing"):
        await service.infer(lambda model: model)
    assert service.state is ModelLoadState.FAILED

    assert isinstance(await service.infer(lambda model: model), _FakeModel)
    assert service.state is ModelLoadState.LOADED


@pytest.mark.asyncio
async def test_ready_after_first_load_and_stays_ready_after_idle_unload(make_service):
    service, clock = make_service()
    assert service.is_ready() is False

    await service.load()
    assert service.is_ready() is True

    clock.advance(_IDLE_TIMEOUT_SECONDS)
    await service.unload_if_idle()
    assert service.state is ModelLoadState.UNLOADED
    assert service.is_ready() is True


@pytest.mark.asyncio
async def test_failed_reload_makes_service_not_ready_until_next_success(make_service):
    """曾经加载成功的模型重新加载失败时变为未就绪，下一次成功加载后恢复。"""
    fail_next = False

    def loader() -> _FakeModel:
        if fail_next:
            raise OSError("model file missing")
        return _FakeModel()

    service, clock = make_service(loader=loader)
    await service.load()
    clock.advance(_IDLE_TIMEOUT_SECONDS)
    await service.unload_if_idle()

    fail_next = True
    with pytest.raises(OSError):
        await service.load()
    assert service.is_ready() is False

    fail_next = False
    await service.load()
    assert service.is_ready() is True


@pytest.mark.asyncio
async def test_prefetch_starts_loading_in_background_and_loads_once(make_service):
    """预取不等待加载即返回，加载在后台进行；重复预取与随后的推理不重复加载。"""
    started = threading.Event()
    release = threading.Event()

    def blocking_loader() -> _FakeModel:
        started.set()
        release.wait(_WAIT_SECONDS)
        return _FakeModel()

    service, _ = make_service(loader=blocking_loader)

    service.prefetch()
    assert await asyncio.to_thread(started.wait, _WAIT_SECONDS)
    service.prefetch()
    release.set()
    await service.infer(lambda model: model)

    assert len(service.load_threads) == 1
    assert service.state is ModelLoadState.LOADED


@pytest.mark.asyncio
async def test_close_cancels_prefetch_that_has_not_started(make_service):
    """关闭时尚未开始的预取被取消，关闭之后不再加载模型。"""
    started = threading.Event()

    def loader() -> _FakeModel:
        started.set()
        return _FakeModel()

    service, _ = make_service(loader=loader)

    service.prefetch()
    await service.close()

    assert not await asyncio.to_thread(started.wait, _ABSENCE_WINDOW_SECONDS)
    assert service.state is ModelLoadState.UNLOADED


@pytest.mark.asyncio
async def test_failed_prefetch_is_logged_and_retried_by_next_load(make_service, caplog):
    """后台加载失败只记录日志并进入失败状态，下一次加载重试。"""
    attempts = itertools.count()
    first_attempt = threading.Event()

    def flaky_loader() -> _FakeModel:
        if next(attempts) == 0:
            first_attempt.set()
            raise OSError("model file missing")
        return _FakeModel()

    service, _ = make_service(loader=flaky_loader)

    with caplog.at_level(logging.WARNING):
        service.prefetch()
        assert await asyncio.to_thread(first_attempt.wait, _WAIT_SECONDS)
        # 排在失败的预取之后：返回时预取任务已经结束。
        await service.load()

    assert "推理模型预取失败" in caplog.text
    assert len(service.load_threads) == 2
    assert service.is_ready() is True
