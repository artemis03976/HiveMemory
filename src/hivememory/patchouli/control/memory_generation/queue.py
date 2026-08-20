"""Patchouli 记忆生成的工作队列适配层。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from hivememory.core.models import IdentityScope, LogicalBlock, MemoryAtom
from hivememory.engines.generation.models import GenerationRequest
from hivememory.infrastructure.work_queue import InMemoryWorkStore
from hivememory.patchouli.control.memory_generation.models import (
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
    QueueTaskIdentity,
    TaskHandle,
    TaskOutcome,
    WorkExecutionContext,
    WorkHandlerPort,
    WorkPayloadCodecRegistry,
    WorkQueueRuntime,
    WorkQueueShutdownSummary,
    adapt_queue_task,
)

MemoryGenerationResults = tuple[MemoryGenerationResult, ...]
ExecuteGeneration = Callable[
    [MemoryGenerationTaskSpec],
    Awaitable[list[MemoryGenerationResult]],
]


@dataclass(frozen=True)
class _MemoryGenerationWork:
    """控制层送入队列的私有不可变信封。

    ``MemoryGenerationTaskSpec`` 是跨控制面与数据面的业务输入；``task_id``
    只服务队列身份和任务观测，因此二者只在队列边界内组合，不再作为一份
    公开的 Memory Generation 模型。
    """

    task_id: str
    spec: MemoryGenerationTaskSpec

    @property
    def topic_id(self) -> str:
        return self.spec.topic_id

    @property
    def intent_id(self) -> str | None:
        return self.spec.intent_id

    @property
    def pending_alias(self) -> str | None:
        return self.spec.pending_alias


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


class _MemoryGenerationWorkAdapter:
    """Memory Generation 私有工作信封 v1 的队列适配器。

    适配器集中定义工作标识、同 topic 顺序键和幂等键，并负责在每次执行尝试前
    重建独立的领域任务对象。
    """

    kind = "patchouli.memory_generation"
    schema_version = 1

    @staticmethod
    def identity(work: _MemoryGenerationWork) -> QueueTaskIdentity:
        """从领域任务中派生稳定的队列标识和调度元数据。"""

        return QueueTaskIdentity(
            work_id=MemoryGenerationQueue.work_id_for(work.task_id),
            ordering_key=work.topic_id,
            correlation_id=work.intent_id or work.pending_alias or work.task_id,
            idempotency_key=work.task_id,
        )

    def encode(self, work: _MemoryGenerationWork) -> object:
        """将领域任务编码为可进入通用 payload codec 的 JSON 值。"""

        if not isinstance(work, _MemoryGenerationWork):
            raise TypeError("memory generation payload has an unexpected type")

        spec = work.spec
        _require_text(work.task_id, field_name="task_id")
        _require_text(spec.topic_id, field_name="topic_id")
        _require_text(spec.label, field_name="label")
        return {
            "task_id": work.task_id,
            "spec": {
                "identity_scope": spec.identity_scope.model_dump(mode="json"),
                "topic_id": spec.topic_id,
                "label": spec.label,
                "source": spec.source.value,
                "request": spec.request.model_dump(mode="json"),
                "interaction_input": self._encode_interaction_input(
                    spec.interaction_input
                ),
                "intent_id": spec.intent_id,
                "pending_alias": spec.pending_alias,
            },
        }

    def decode(self, payload: object) -> _MemoryGenerationWork:
        """从 JSON 值重建一次执行尝试专用的领域任务。"""

        if not isinstance(payload, dict):
            raise TypeError("memory generation payload must be an object")
        raw_spec = payload.get("spec")
        if not isinstance(raw_spec, dict):
            raise TypeError("memory generation payload.spec must be an object")
        raw_request = raw_spec.get("request")
        raw_scope = raw_spec.get("identity_scope")
        if not isinstance(raw_request, dict):
            raise TypeError("memory generation payload.spec.request must be an object")
        if not isinstance(raw_scope, dict):
            raise TypeError("memory generation payload.spec.identity_scope must be an object")

        request_data = dict(raw_request)
        existing_memory = request_data.get("existing_memory")
        if existing_memory is not None:
            if not isinstance(existing_memory, dict):
                raise TypeError("request.existing_memory must be an object")
            # GenerationRequest 将 existing_memory 声明为 Any，需要在 codec
            # 边界显式恢复领域类型，避免数据面收到普通 dict。
            request_data["existing_memory"] = MemoryAtom.model_validate(existing_memory)

        return _MemoryGenerationWork(
            task_id=_require_text(payload.get("task_id"), field_name="task_id"),
            spec=MemoryGenerationTaskSpec(
                identity_scope=IdentityScope.model_validate(raw_scope),
                topic_id=_require_text(
                    raw_spec.get("topic_id"),
                    field_name="topic_id",
                ),
                label=_require_text(raw_spec.get("label"), field_name="label"),
                source=MemoryGenerationSource(raw_spec["source"]),
                request=GenerationRequest.model_validate(request_data),
                interaction_input=self._decode_interaction_input(
                    raw_spec.get("interaction_input")
                ),
                intent_id=raw_spec.get("intent_id"),
                pending_alias=raw_spec.get("pending_alias"),
            ),
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
            "blocks": [block.model_dump(mode="json") for block in interaction_input.blocks],
        }

    @staticmethod
    def _decode_interaction_input(payload: object) -> InteractionArtifactInput | None:
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


class MemoryGenerationHandle(TaskHandle[MemoryGenerationResults]):
    """记忆生成工作被接纳后返回的类型化控制句柄。

    payload bytes 只用于同一 work identity 的进程内冲突检测，不作为任务状态
    或对外工作定义暴露。
    """

    def __init__(
        self,
        *,
        payload_bytes: bytes,
        work_id: str,
        queue: WorkQueueRuntime,
    ) -> None:
        super().__init__(work_id=work_id, queue=queue)
        self._payload_bytes = payload_bytes

    def matches_payload(self, payload_bytes: bytes) -> bool:
        """判断重复 work identity 是否仍指向同一不可变 payload。"""

        return self._payload_bytes == payload_bytes


class MemoryGenerationHandler(WorkHandlerPort[_MemoryGenerationWork, str]):
    """将单次队列执行尝试适配到现有的记忆生成数据面。

    handler 只执行生成并分类失败；任务状态仍由通用 runtime 维护，领域结果通过
    对应的类型化 handle 暂存在进程内，供控制器完成终态结算。
    """

    def __init__(
        self,
        execute_generation: ExecuteGeneration,
        handles: dict[str, MemoryGenerationHandle],
    ) -> None:
        self._execute_generation = execute_generation
        self._handles = handles

    async def execute(
        self,
        work: _MemoryGenerationWork,
        context: WorkExecutionContext,
    ) -> str:
        """执行一次记忆生成尝试，并保存领域结果供 finalize 使用。"""

        handle = self._handles[context.work_id]
        handle._record_execution_started()
        results = await self._execute_generation(work.spec)
        if not isinstance(results, list) or not all(
            isinstance(result, MemoryGenerationResult) for result in results
        ):
            raise TypeError(
                "memory generation handler must return MemoryGenerationResult list"
            )
        typed_results = tuple(results)
        handle._record_execution_result(typed_results)
        return context.work_id

    def classify_failure(
        self,
        error: Exception,
        context: WorkExecutionContext,
    ) -> FailureDecision:
        """记录失败，并禁止自动重放整条记忆生成数据面。"""

        handle = self._handles.get(context.work_id)
        if handle is not None:
            handle._record_execution_error(error)
        return FailureDecision(
            action=FailureAction.FAIL,
            reason="memory_generation_failed",
        )


class MemoryGenerationQueue:
    """Patchouli 的结构化记忆生成队列边界。

    对外由 Controller 交付任务规范和身份，在本模块内包装为私有工作信封后
    转换成 ``WorkItem``；类型化 handle 与执行结果仅服务当前进程，运行时状态
    仍统一来自 ``WorkRecord``。
    """

    LANE = "patchouli.memory_generation"

    def __init__(
        self,
        execute_generation: ExecuteGeneration,
        *,
        store: InMemoryWorkStore | None = None,
        runtime_events: RuntimeEventSink | None = None,
        policy: QueuePolicy | None = None,
    ) -> None:
        if policy is not None and policy.max_attempts != 1:
            raise ValueError("memory generation queue requires max_attempts=1")

        self._adapter = _MemoryGenerationWorkAdapter()
        self._codecs = WorkPayloadCodecRegistry()
        self._codecs.register(self._adapter)
        self._handles: dict[str, MemoryGenerationHandle] = {}
        self._handler = MemoryGenerationHandler(
            execute_generation,
            self._handles,
        )
        self._runtime = WorkQueueRuntime(
            store=store or InMemoryWorkStore(),
            payload_codecs=self._codecs,
            runtime_events=runtime_events,
            worker_poll_interval_seconds=0.02,
            shutdown_wait_seconds=2.0,
        )
        self._lane = self._runtime.register_lane(
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
    def terminal_retention(self) -> int:
        return self._lane.policy.terminal_retention

    @property
    def started(self) -> bool:
        return self._runtime.started

    async def start(self) -> None:
        """幂等启动底层工作队列运行时。"""

        async with self._start_lock:
            if not self._runtime.started:
                await self._runtime.start()

    async def stop(self) -> WorkQueueShutdownSummary:
        """停止底层运行时并返回尚未完成的工作摘要。"""

        return await self._runtime.stop()

    async def submit(
        self,
        task_id: str,
        spec: MemoryGenerationTaskSpec,
    ) -> MemoryGenerationHandle:
        """接纳一个领域工作定义并返回类型化控制句柄。"""

        work = _MemoryGenerationWork(task_id=task_id, spec=spec)
        item = adapt_queue_task(
            work,
            lane=self.LANE,
            adapter=self._adapter,
            codecs=self._codecs,
        )
        existing = self._handles.get(item.work_id)
        if existing is not None:
            if not existing.matches_payload(item.payload):
                raise ValueError(
                    f"memory generation work already exists with different payload: {item.work_id}"
                )
            return existing
        handle = MemoryGenerationHandle(
            payload_bytes=item.payload,
            work_id=item.work_id,
            queue=self._runtime,
        )
        # worker 可能在 enqueue 返回前取得工作，必须先注册 handle，确保 handler
        # 总能找到用于记录进程内结果的目标。
        self._handles[item.work_id] = handle
        try:
            await self._runtime.enqueue(item)
        except BaseException:
            self._handles.pop(item.work_id, None)
            raise
        return handle

    def release(self, handle: MemoryGenerationHandle) -> None:
        """保留任务被淘汰后，释放其进程内类型化结果数据。"""

        self._handles.pop(handle.work_id, None)

    async def get(self, work_id: str):
        """读取底层 ``WorkRecord`` 快照。"""

        return await self._runtime.get(work_id)

    async def wait(self, work_id: str, timeout: float | None = None):
        """等待底层工作进入终态。"""

        return await self._runtime.wait(work_id, timeout=timeout)

    async def cancel(
        self,
        work_id: str,
        *,
        reason: str = "user_requested",
    ) -> bool:
        """向底层运行时请求取消指定工作。"""

        return await self._runtime.cancel(work_id, reason=reason)

    @staticmethod
    def work_id_for(task_id: str) -> str:
        """生成记忆任务在通用运行时中的稳定工作标识。"""

        return f"memory_generation:{task_id}"


__all__ = [
    "MemoryGenerationHandle",
    "MemoryGenerationHandler",
    "MemoryGenerationQueue",
    "MemoryGenerationResults",
    "TaskOutcome",
]
