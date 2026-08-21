"""Memory Generation Queue 的 Q3 契约测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    IndexLayer,
    LogicalBlock,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TurnRecord,
)
from hivememory.engines.generation.models import GenerationContext, GenerationRequest
from hivememory.patchouli.control.memory_generation.controller import (
    MemoryGenerationTaskController,
)
from hivememory.patchouli.control.memory_generation.models import (
    InteractionArtifactInput,
    MemoryGenerationSource,
    MemoryGenerationTaskSpec,
    MemoryGenerationTaskStatus,
)
from hivememory.patchouli.control.memory_generation.queue import (
    _MemoryGenerationWork,
    _MemoryGenerationWorkAdapter,
)
from hivememory.system.runtime.work_queue import (
    QueuePolicy,
    WorkPayloadCodecRegistry,
    WorkPayloadDecodeError,
    WorkState,
    encode_canonical_json,
)
from tests.helpers.memory import make_memory_creation_context, make_memory_metadata


def _memory_atom() -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="agent-1", user_id="u1"),
        index=IndexLayer(
            title="memory title",
            summary="summary text",
            tags=["tag"],
            memory_type=MemoryType.FACT,
            alias="memory_alias",
        ),
        payload=PayloadLayer(content="original content"),
    )


def _spec(
    *,
    label: str = "memory-1",
    topic_id: str = "topic-1",
    request: GenerationRequest | None = None,
    intent_id: str | None = None,
    pending_alias: str | None = None,
) -> MemoryGenerationTaskSpec:
    return MemoryGenerationTaskSpec(
        identity_scope=make_memory_creation_context(),
        topic_id=topic_id,
        label=label,
        source=MemoryGenerationSource.WRITE,
        request=request or GenerationRequest(
            context=GenerationContext(),
        ),
        intent_id=intent_id,
        pending_alias=pending_alias,
    )


def _controller(
    bus,
    *,
    max_concurrency: int = 1,
    timeout_seconds: float | None = None,
) -> MemoryGenerationTaskController:
    return MemoryGenerationTaskController(
        bus=bus,
        queue_policy=QueuePolicy(
            capacity=16,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            max_attempts=1,
            terminal_retention=16,
        ),
    )


def test_spec_codec_creates_canonical_deep_snapshot_and_restores_domain_types() -> None:
    atom = _memory_atom()
    block = LogicalBlock(
        turn=TurnRecord(user_query="question", assistant_final_text="answer")
    )
    spec = MemoryGenerationTaskSpec(
        identity_scope=make_memory_creation_context(),
        topic_id="topic-codec",
        label="codec",
        source=MemoryGenerationSource.UPDATE,
        request=GenerationRequest(
            context=GenerationContext(state_summary="original summary"),
            existing_memory=atom,
        ),
        interaction_input=InteractionArtifactInput(
            topic_id="topic-codec",
            topic_title="title",
            topic_summary="summary",
            blocks=(block,),
        ),
        intent_id="intent-codec",
        pending_alias="draft-codec",
    )
    codecs = WorkPayloadCodecRegistry()
    codecs.register(_MemoryGenerationWorkAdapter())
    work = _MemoryGenerationWork(task_id="task-codec", spec=spec)

    payload_bytes = codecs.encode(
        _MemoryGenerationWorkAdapter.kind,
        _MemoryGenerationWorkAdapter.schema_version,
        work,
    )

    spec.request.context.state_summary = "external mutation"
    atom.payload.content = "external mutation"
    first = codecs.decode(
        _MemoryGenerationWorkAdapter.kind,
        _MemoryGenerationWorkAdapter.schema_version,
        payload_bytes,
    )
    second = codecs.decode(
        _MemoryGenerationWorkAdapter.kind,
        _MemoryGenerationWorkAdapter.schema_version,
        payload_bytes,
    )

    assert first.task_id == "task-codec"
    assert first.spec.identity_scope == spec.identity_scope
    assert first.spec.identity_scope is not spec.identity_scope
    assert first.spec.identity_scope is not second.spec.identity_scope
    assert first.spec.request.context.state_summary == "original summary"
    assert isinstance(first.spec.request.existing_memory, MemoryAtom)
    assert first.spec.request.existing_memory.payload.content == "original content"
    assert first.spec.interaction_input is not None
    assert isinstance(first.spec.interaction_input.blocks[0], LogicalBlock)
    first.spec.request.context.state_summary = "attempt-local mutation"
    assert second.spec.request.context.state_summary == "original summary"


def test_memory_generation_codec_rejects_flattened_owner_projection() -> None:
    """捕获 codec 接受 TaskSpec 内第二份 owner_user_id 身份事实的缺陷。"""
    adapter = _MemoryGenerationWorkAdapter()
    encoded = adapter.encode(
        _MemoryGenerationWork(task_id="tampered-task", spec=_spec())
    )
    encoded["spec"]["owner_user_id"] = "attacker"
    codecs = WorkPayloadCodecRegistry()
    codecs.register(adapter)

    with pytest.raises(WorkPayloadDecodeError):
        codecs.decode(
            adapter.kind,
            adapter.schema_version,
            encode_canonical_json(encoded),
        )


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_memory_generation_codec_rejects_nested_noncanonical_request(
    mutation: str,
) -> None:
    """捕获 GenerationRequest 嵌套字段被静默忽略或补默认值的缺陷。"""
    adapter = _MemoryGenerationWorkAdapter()
    encoded = adapter.encode(
        _MemoryGenerationWork(task_id="nested-tamper", spec=_spec())
    )
    if mutation == "extra":
        encoded["spec"]["request"]["workspace_id"] = "isolation_workspace"
    else:
        del encoded["spec"]["request"]["write_focus"]
    codecs = WorkPayloadCodecRegistry()
    codecs.register(adapter)

    with pytest.raises(WorkPayloadDecodeError):
        codecs.decode(
            adapter.kind,
            adapter.schema_version,
            encode_canonical_json(encoded),
        )


@pytest.mark.asyncio
async def test_concurrency_limit_keeps_later_task_queued_and_pending() -> None:
    first_started = asyncio.Event()
    release = asyncio.Event()

    async def execute(route, spec):
        if spec.label == "first":
            first_started.set()
            await release.wait()
        return []

    bus = Mock(request=AsyncMock(side_effect=execute), publish=AsyncMock())
    controller = _controller(bus, max_concurrency=1)
    await controller.start()

    try:
        first = await controller.submit_generation(_spec(label="first"))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = await controller.submit_generation(_spec(label="second"))
        await asyncio.sleep(0.03)

        second_work = await controller.queue.get(
            controller.queue.work_id_for(second.task_id)
        )
        assert second_work is not None
        assert second_work.state == WorkState.QUEUED
        assert second.status == MemoryGenerationTaskStatus.PENDING
        assert await controller.get_task(second.task_id) == second

        release.set()
        completed = await controller.wait_all(timeout=2)
        assert len(completed) == 2
        assert all(
            task.status == MemoryGenerationTaskStatus.COMPLETED
            for task in completed
        )
        assert (await controller.get_task(first.task_id)).status == MemoryGenerationTaskStatus.COMPLETED
        assert (await controller.get_task(second.task_id)).status == MemoryGenerationTaskStatus.COMPLETED
    finally:
        release.set()
        await controller.stop()


@pytest.mark.asyncio
async def test_queued_cancel_never_calls_generation_handler() -> None:
    first_started = asyncio.Event()
    release = asyncio.Event()
    called_labels: list[str] = []

    async def execute(route, spec):
        called_labels.append(spec.label)
        if spec.label == "first":
            first_started.set()
            await release.wait()
        return []

    bus = Mock(request=AsyncMock(side_effect=execute), publish=AsyncMock())
    controller = _controller(bus, max_concurrency=1)
    await controller.start()

    try:
        first = await controller.submit_generation(_spec(label="first"))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        queued = await controller.submit_generation(_spec(label="queued"))

        assert await controller.cancel_task(queued.task_id) is True
        cancelled = await controller.wait_task(queued.task_id, timeout=1)

        assert cancelled.status == MemoryGenerationTaskStatus.CANCELLED
        assert called_labels == ["first"]
        release.set()
        await controller.wait_task(first.task_id, timeout=1)
    finally:
        release.set()
        await controller.stop()


@pytest.mark.asyncio
async def test_running_cancel_interrupts_handler_and_projects_cancelled() -> None:
    started = asyncio.Event()
    released = asyncio.Event()

    async def execute(route, spec):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    bus = Mock(request=AsyncMock(side_effect=execute), publish=AsyncMock())
    controller = _controller(bus)
    await controller.start()

    try:
        memory_task = await controller.submit_generation(_spec())
        await asyncio.wait_for(started.wait(), timeout=1)

        assert await controller.cancel_task(memory_task.task_id) is True
        result = await controller.wait_task(memory_task.task_id, timeout=1)

        await asyncio.wait_for(released.wait(), timeout=1)
        assert result.status == MemoryGenerationTaskStatus.CANCELLED
        assert (await controller.get_task(memory_task.task_id)).cancelled is True
    finally:
        await controller.stop()


def test_memory_generation_rejects_multiple_attempt_policy() -> None:
    bus = Mock(request=AsyncMock(), publish=AsyncMock())

    with pytest.raises(ValueError, match="requires max_attempts=1"):
        MemoryGenerationTaskController(
            bus=bus,
            queue_policy=QueuePolicy(
                capacity=16,
                max_concurrency=1,
                max_attempts=2,
            ),
        )


@pytest.mark.asyncio
async def test_memory_generation_submit_requires_explicit_start() -> None:
    controller = MemoryGenerationTaskController(
        bus=Mock(request=AsyncMock(), publish=AsyncMock())
    )

    with pytest.raises(RuntimeError, match="must be started"):
        await controller.submit_generation(_spec())


@pytest.mark.asyncio
async def test_default_policy_does_not_retry_generation_side_effects() -> None:
    bus = Mock(
        request=AsyncMock(
            side_effect=ConnectionError(
                "generation may already have written partial results"
            )
        ),
        publish=AsyncMock(),
    )
    controller = MemoryGenerationTaskController(bus=bus)
    await controller.start()

    try:
        memory_task = await controller.submit_generation(_spec())
        result = await controller.wait_task(memory_task.task_id, timeout=1)
    finally:
        await controller.stop()

    assert result.status == MemoryGenerationTaskStatus.FAILED
    assert bus.request.await_count == 1
    assert (await controller.get_task(memory_task.task_id)).status == MemoryGenerationTaskStatus.FAILED


@pytest.mark.asyncio
async def test_non_retryable_failure_fails_once() -> None:
    bus = Mock(
        request=AsyncMock(side_effect=ValueError("invalid generation spec")),
        publish=AsyncMock(),
    )
    controller = _controller(bus)
    await controller.start()

    try:
        memory_task = await controller.submit_generation(_spec())
        result = await controller.wait_task(memory_task.task_id, timeout=1)
    finally:
        await controller.stop()

    assert result.status == MemoryGenerationTaskStatus.FAILED
    assert result.error == "invalid generation spec"
    assert bus.request.await_count == 1


@pytest.mark.asyncio
async def test_timeout_fails_without_retry() -> None:
    async def execute(route, spec):
        await asyncio.Event().wait()

    bus = Mock(
        request=AsyncMock(side_effect=execute),
        publish=AsyncMock(),
    )
    controller = _controller(
        bus,
        timeout_seconds=0.01,
    )
    await controller.start()

    try:
        memory_task = await controller.submit_generation(_spec())
        result = await controller.wait_task(memory_task.task_id, timeout=1)
        record = await controller.queue.get(
            controller.queue.work_id_for(memory_task.task_id)
        )
    finally:
        await controller.stop()

    assert result.status == MemoryGenerationTaskStatus.FAILED
    assert result.error == "TimeoutError"
    assert record is not None
    assert record.state == WorkState.FAILED
    assert bus.request.await_count == 1
