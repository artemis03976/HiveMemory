"""Patchouli Memory Generation 的工作队列适配。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from hivememory.core.models import LogicalBlock, MemoryAtom
from hivememory.engines.generation.models import GenerationRequest
from hivememory.infrastructure.work_queue import InMemoryWorkStore
from hivememory.patchouli.runtime.memory_tasks import (
    InteractionArtifactInput,
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTaskSpec,
)
from hivememory.system.runtime.events import RuntimeEventSink
from hivememory.system.runtime.work_queue import (
    FailureAction,
    FailureDecision,
    QueuePolicy,
    WorkExecutionContext,
    WorkHandlerPort,
    WorkItem,
    WorkPayloadCodecRegistry,
    WorkQueueRuntime,
    WorkQueueShutdownSummary,
    WorkReceipt,
    WorkRecord,
)

ExecuteGeneration = Callable[
    [MemoryGenerationTaskSpec],
    Awaitable[list[MemoryGenerationResult]],
]


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


class TransientMemoryGenerationError(RuntimeError):
    """由生成数据面显式标记、允许安全重试的瞬态失败。"""


class MemoryGenerationTaskSpecCodec:
    """MemoryGenerationTaskSpec 的 v1 canonical JSON codec。"""

    kind = "patchouli.memory_generation"
    schema_version = 1

    def encode(self, spec: MemoryGenerationTaskSpec) -> object:
        if not isinstance(spec, MemoryGenerationTaskSpec):
            raise TypeError("memory generation payload has an unexpected type")
        _require_text(spec.topic_id, field_name="topic_id")
        _require_text(spec.label, field_name="label")
        return {
            "topic_id": spec.topic_id,
            "label": spec.label,
            "source": spec.source.value,
            "request": spec.request.model_dump(mode="json"),
            "interaction_input": self._encode_interaction_input(
                spec.interaction_input
            ),
            "intent_id": spec.intent_id,
            "pending_alias": spec.pending_alias,
        }

    def decode(self, payload: object) -> MemoryGenerationTaskSpec:
        if not isinstance(payload, dict):
            raise TypeError("memory generation payload must be an object")

        raw_request = payload.get("request")
        if not isinstance(raw_request, dict):
            raise TypeError("memory generation payload.request must be an object")
        request_data = dict(raw_request)
        existing_memory = request_data.get("existing_memory")
        if existing_memory is not None:
            if not isinstance(existing_memory, dict):
                raise TypeError("request.existing_memory must be an object")
            # GenerationRequest 的 existing_memory 是 Any，需要在 codec 边界
            # 显式恢复领域类型，避免数据面收到普通 dict。
            request_data["existing_memory"] = MemoryAtom.model_validate(
                existing_memory
            )

        return MemoryGenerationTaskSpec(
            topic_id=_require_text(payload.get("topic_id"), field_name="topic_id"),
            label=_require_text(payload.get("label"), field_name="label"),
            source=MemoryGenerationSource(payload["source"]),
            request=GenerationRequest.model_validate(request_data),
            interaction_input=self._decode_interaction_input(
                payload.get("interaction_input")
            ),
            intent_id=payload.get("intent_id"),
            pending_alias=payload.get("pending_alias"),
        )

    @staticmethod
    def _encode_interaction_input(
        interaction_input: InteractionArtifactInput | None,
    ) -> object:
        if interaction_input is None:
            return None
        return {
            "topic_id": interaction_input.topic_id,
            "topic_title": interaction_input.topic_title,
            "topic_summary": interaction_input.topic_summary,
            "blocks": [
                block.model_dump(mode="json")
                for block in interaction_input.blocks
            ],
        }

    @staticmethod
    def _decode_interaction_input(
        payload: object,
    ) -> InteractionArtifactInput | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise TypeError("interaction_input must be an object")
        raw_blocks = payload.get("blocks", [])
        if not isinstance(raw_blocks, list):
            raise TypeError("interaction_input.blocks must be an array")
        return InteractionArtifactInput(
            topic_id=_require_text(
                payload.get("topic_id"),
                field_name="interaction_input.topic_id",
            ),
            topic_title=payload.get("topic_title", ""),
            topic_summary=payload.get("topic_summary", ""),
            blocks=tuple(LogicalBlock.model_validate(block) for block in raw_blocks),
        )


@dataclass(frozen=True)
class MemoryGenerationExecutionResult:
    """handler 成功后的轻量结果引用。"""

    work_id: str
    result_count: int

    @property
    def result_ref(self) -> str:
        return self.work_id


class MemoryGenerationHandler(
    WorkHandlerPort[MemoryGenerationTaskSpec, MemoryGenerationExecutionResult]
):
    """把 memory generation work attempt 适配到现有生成数据面。"""

    def __init__(
        self,
        execute_generation: ExecuteGeneration,
        *,
        retry_after_seconds: float = 0.05,
    ) -> None:
        self._execute_generation = execute_generation
        self._retry_after_seconds = retry_after_seconds
        self._results: dict[str, tuple[MemoryGenerationResult, ...]] = {}
        self._errors: dict[str, str] = {}

    async def execute(
        self,
        payload: MemoryGenerationTaskSpec,
        context: WorkExecutionContext,
    ) -> MemoryGenerationExecutionResult:
        # handler 已完成但通用状态确认发生模糊失败时，后续 attempt 直接复用
        # 内存结果，不重复进入具有写入副作用的生成数据面。
        cached = self._results.get(context.work_id)
        if cached is not None:
            return MemoryGenerationExecutionResult(
                work_id=context.work_id,
                result_count=len(cached),
            )

        results = await self._execute_generation(payload)
        if not isinstance(results, list) or not all(
            isinstance(result, MemoryGenerationResult) for result in results
        ):
            raise TypeError(
                "memory generation handler must return MemoryGenerationResult list"
            )

        self._results[context.work_id] = tuple(results)
        self._errors.pop(context.work_id, None)
        return MemoryGenerationExecutionResult(
            work_id=context.work_id,
            result_count=len(results),
        )

    def classify_failure(
        self,
        error: Exception,
        context: WorkExecutionContext,
    ) -> FailureDecision:
        self._errors[context.work_id] = str(error) or type(error).__name__
        if isinstance(
            error,
            (TimeoutError, ConnectionError, TransientMemoryGenerationError),
        ):
            return FailureDecision(
                action=FailureAction.RETRY,
                retry_after_seconds=self._retry_after_seconds,
                reason="memory_generation_transient_failure",
            )
        return FailureDecision(
            action=FailureAction.FAIL,
            reason="memory_generation_failed",
        )

    def take_results(
        self,
        work_id: str,
    ) -> tuple[MemoryGenerationResult, ...] | None:
        """取出已确认成功 work 的领域结果。"""
        return self._results.pop(work_id, None)

    def take_error(self, work_id: str) -> str | None:
        """取出终止 work 的原始错误文本，仅用于领域任务投影。"""
        return self._errors.pop(work_id, None)


class MemoryGenerationQueue:
    """Patchouli memory generation lane 的轻量业务队列。"""

    LANE = "patchouli.memory_generation"

    def __init__(
        self,
        execute_generation: ExecuteGeneration,
        *,
        store: InMemoryWorkStore | None = None,
        runtime_events: RuntimeEventSink | None = None,
        policy: QueuePolicy | None = None,
        retry_after_seconds: float = 0.05,
    ) -> None:
        self._codecs = WorkPayloadCodecRegistry()
        self._codecs.register(MemoryGenerationTaskSpecCodec())

        self._handler = MemoryGenerationHandler(
            execute_generation,
            retry_after_seconds=retry_after_seconds,
        )
        self._runtime = WorkQueueRuntime(
            store=store or InMemoryWorkStore(),
            payload_codecs=self._codecs,
            runtime_events=runtime_events,
            worker_poll_interval_seconds=0.02,
            shutdown_wait_seconds=2.0,
        )
        self._runtime.register_lane(
            self.LANE,
            handler=self._handler,
            policy=policy
            or QueuePolicy(
                capacity=128,
                max_concurrency=2,
                timeout_seconds=300.0,
                max_attempts=1,
                terminal_retention=100,
            ),
        )
        self._start_lock = asyncio.Lock()

    @property
    def runtime(self) -> WorkQueueRuntime:
        return self._runtime

    @property
    def started(self) -> bool:
        return self._runtime.started

    async def start(self) -> None:
        async with self._start_lock:
            if not self._runtime.started:
                await self._runtime.start()

    async def stop(self) -> WorkQueueShutdownSummary:
        return await self._runtime.stop()

    async def submit(
        self,
        task_id: str,
        spec: MemoryGenerationTaskSpec,
    ) -> WorkReceipt:
        payload_bytes = self._codecs.encode(
            MemoryGenerationTaskSpecCodec.kind,
            MemoryGenerationTaskSpecCodec.schema_version,
            spec,
        )
        item = WorkItem(
            work_id=self.work_id_for(task_id),
            lane=self.LANE,
            kind=MemoryGenerationTaskSpecCodec.kind,
            schema_version=MemoryGenerationTaskSpecCodec.schema_version,
            payload=payload_bytes,
            ordering_key=spec.topic_id,
            correlation_id=spec.intent_id or spec.pending_alias or task_id,
            idempotency_key=task_id,
        )
        return await self._runtime.enqueue(item)

    async def get(self, work_id: str) -> WorkRecord | None:
        return await self._runtime.get(work_id)

    async def wait(
        self,
        work_id: str,
        timeout: float | None = None,
    ) -> WorkRecord | None:
        return await self._runtime.wait(work_id, timeout=timeout)

    async def cancel(
        self,
        work_id: str,
        *,
        reason: str = "user_requested",
    ) -> bool:
        return await self._runtime.cancel(work_id, reason=reason)

    def take_results(
        self,
        work_id: str,
    ) -> tuple[MemoryGenerationResult, ...] | None:
        return self._handler.take_results(work_id)

    def take_error(self, work_id: str) -> str | None:
        return self._handler.take_error(work_id)

    @staticmethod
    def work_id_for(task_id: str) -> str:
        return f"memory_generation:{task_id}"


__all__ = [
    "MemoryGenerationExecutionResult",
    "MemoryGenerationHandler",
    "MemoryGenerationQueue",
    "MemoryGenerationTaskSpecCodec",
    "TransientMemoryGenerationError",
]
