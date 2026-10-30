"""
进程级推理模型服务：实例复用、按需加载与空闲卸载

嵌入与重排模型各占数百 MB 到 1 GB 以上的常驻内存，而使用并不连续。模型服务在推理时
按需加载模型；``idle_unload`` 模式下空闲超过时限即卸载，并把释放的堆内存归还操作系统，
``resident`` 模式加载后一直驻留到进程结束。

加载、推理与卸载都在服务自己的推理线程中逐个执行：推理进行中不会卸载，加载与卸载互不
重叠，都由同一个线程保证，不另加锁。
"""

import ctypes
import functools
import gc
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Concatenate

from hivememory.infrastructure.inference_worker import InferenceWorker

if TYPE_CHECKING:
    from hivememory.config.patchouli import RerankerConfig
    from hivememory.config.shared import EmbeddingConfig

logger = logging.getLogger(__name__)


class ModelLoadState(StrEnum):
    """模型的驻留状态。"""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


class SingletonModelService[M](ABC):
    """
    推理模型服务基类（Multiton）

    按（服务类，配置的 model_name）复用实例：模型服务是进程级的，之后启动的系统继续使用
    同一实例与已加载的模型。子类实现 ``_load_model``，推理经 ``_run_with_model`` 在推理
    线程中取得模型执行。

    状态分两组，各只有一个写入方：模型、驻留状态与是否加载过只在推理线程中写入；在途
    推理数与最近一次使用时间只在事件循环线程中读写。
    """

    _instances: ClassVar[dict[tuple[type, str], "SingletonModelService[Any]"]] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, config, *args, **kwargs):
        """按服务类与配置的 model_name 获取或创建实例。"""
        key = (cls, config.model_name)
        if key not in cls._instances:
            with cls._instances_lock:
                if key not in cls._instances:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instances[key] = instance
        return cls._instances[key]

    def __init__(
        self,
        config: "EmbeddingConfig | RerankerConfig",
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """
        初始化服务配置与驻留状态；同一实例再次构造时沿用首次的配置。

        Args:
            config: 嵌入或重排模型配置，提供模型名、缓存目录、驻留方式与空闲时限
            clock: 判定空闲时长的时钟；默认单调时钟，不受系统墙钟回拨影响
        """
        if getattr(self, "_initialized", False):
            return

        self.config = config
        self.model_name = config.model_name
        self.cache_dir = config.cache_dir
        self._clock = clock or time.monotonic
        # 加载、推理与卸载都在这个线程中逐个执行，不占用事件循环。
        self._inference = InferenceWorker(self.model_name)

        # 只在推理线程中写入，事件循环线程只读取。
        self._model: M | None = None
        self._state = ModelLoadState.UNLOADED
        self._loaded_once = False

        # 只在事件循环线程中读写。在途推理数包含已提交、尚未开始执行的调用。
        self._in_flight = 0
        self._last_used = self._clock()

        self._initialized = True

        logger.info(
            f"{self.__class__.__name__} 配置: model={self.model_name}, "
            f"residency={config.residency}"
        )

    @abstractmethod
    def _load_model(self) -> M:
        """在推理线程中加载并返回模型，由子类实现。"""

    @property
    def state(self) -> ModelLoadState:
        """当前驻留状态。"""
        return self._state

    def is_ready(self) -> bool:
        """自创建以来至少成功加载过一次，且当前不处于加载失败状态。

        空闲卸载不改变就绪：卸载后的下一次推理会重新加载。
        """
        return self._loaded_once and self._state is not ModelLoadState.FAILED

    async def load(self) -> None:
        """加载模型并等待完成；已加载时只记为一次使用。加载失败时抛出加载错误。"""
        await self._run_with_model(lambda _model: None)

    async def unload_if_idle(self) -> None:
        """``idle_unload`` 模式下，模型空闲超过时限时卸载并归还内存。

        模型不处于已加载状态，或有在途推理时不卸载。判定之后、卸载执行之前新到的推理
        排在卸载之后，执行时重新加载。
        """
        if self.config.residency != "idle_unload" or self._state is not ModelLoadState.LOADED:
            return
        if self._in_flight > 0:
            return
        if self._clock() - self._last_used < self.config.idle_timeout_seconds:
            return
        await self._inference.run(self._unload)

    async def close(self) -> None:
        """等待在途调用结束并释放推理线程；模型保持驻留，之后的调用重新创建线程。"""
        await self._inference.close()

    async def _run_with_model[**P, T](
        self,
        func: Callable[Concatenate[M, P], T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """在推理线程中以已加载的模型执行 ``func(model, *args, **kwargs)``。

        模型未加载或上次加载失败时，先在同一线程中加载，调用方等待。调用期间计入在途
        推理数；结束时（含失败与取消）记录最近一次使用时间。
        """

        def call() -> T:
            return func(self._loaded_model(), *args, **kwargs)

        self._in_flight += 1
        try:
            return await self._inference.run(call)
        finally:
            self._in_flight -= 1
            self._last_used = self._clock()

    def _loaded_model(self) -> M:
        """在推理线程中取得模型；未加载或上次加载失败时先加载。"""
        if self._model is not None:
            return self._model

        self._state = ModelLoadState.LOADING
        started = time.perf_counter()
        try:
            model = self._load_model()
        except Exception:
            self._state = ModelLoadState.FAILED
            raise
        # 加载完成后才对推理可见。
        self._model = model
        self._state = ModelLoadState.LOADED
        self._loaded_once = True
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(f"推理模型加载完成: model={self.model_name}, 耗时={elapsed_ms:.0f}ms")
        return model

    def _unload(self) -> None:
        """在推理线程中释放模型，回收循环引用，并把空闲堆内存归还操作系统。

        glibc 不会主动把释放的大块堆内存还给操作系统，只删除模型对象时进程常驻内存
        基本不降，需要再调用 ``malloc_trim``。
        """
        rss_before = _resident_memory_mb()
        started = time.perf_counter()
        self._model = None
        self._state = ModelLoadState.UNLOADED
        gc.collect()
        malloc_trim = _find_malloc_trim()
        if malloc_trim is not None:
            malloc_trim(0)
        elapsed_ms = (time.perf_counter() - started) * 1000
        rss_after = _resident_memory_mb()

        message = f"推理模型已卸载: model={self.model_name}, 耗时={elapsed_ms:.0f}ms"
        if rss_before is not None and rss_after is not None:
            message += f", 常驻内存 {rss_before:.0f}MB → {rss_after:.0f}MB"
        logger.info(message)


@functools.cache
def _find_malloc_trim() -> Callable[[int], int] | None:
    """取得 glibc 的 ``malloc_trim``；运行库不提供时记录一次 debug 日志并返回 None。"""
    try:
        return ctypes.CDLL("libc.so.6").malloc_trim
    except (OSError, AttributeError):
        logger.debug("运行库不提供 malloc_trim，卸载模型后释放的内存可能不会归还操作系统")
        return None


def _resident_memory_mb() -> float | None:
    """读取进程当前的常驻内存（MB）；平台不提供 ``/proc/self/status`` 时返回 None。"""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


__all__ = [
    "ModelLoadState",
    "SingletonModelService",
]
