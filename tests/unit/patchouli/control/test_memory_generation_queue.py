"""Memory Generation Queue 的 Q3 契约测试。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.components.work_queue import (
    QueuePolicy,
    UnknownWorkPayloadCodecError,
    WorkExecutionContext,
    WorkPayloadCodecRegistry,
    WorkPayloadDecodeError,
    WorkQueueStoppedError,
    WorkState,
    encode_canonical_json,
)
from hivememory.core.errors import (
    MemoryAliasConflictError,
    MemoryMergeAbortedError,
    ModelContextWindowExceededError,
    ModelProviderError,
)
from hivememory.core.models import (
    IndexLayer,
    LogicalBlock,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TopicAssetBinding,
    TopicFold,
    TurnEvent,
    UpdateFocus,
    WorkspaceAssetRef,
)
from hivememory.engines.generation.models import GenerationContext, GenerationRequest, MergeFocus
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
    MemoryGenerationHandler,
    MemoryGenerationQueue,
    _MemoryGenerationWork,
    _MemoryGenerationWorkAdapter,
    memory_generation_failure_reason,
)
from hivememory.prompts.transcript import GenerationTranscriptBuilder
from tests.helpers.generation_bus import make_generation_bus
from tests.helpers.memory import make_memory_identity_scope, make_memory_metadata
from tests.helpers.sessions import make_logical_block


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
        belong_to=make_memory_identity_scope().workspace_identity,
        from_actor=make_memory_identity_scope().actor_identity,
        topic_id=topic_id,
        label=label,
        source=MemoryGenerationSource.WRITE,
        request=request
        or GenerationRequest(
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


@pytest.mark.parametrize("merge", [False, True], ids=["revision", "merge"])
def test_spec_codec_creates_canonical_deep_snapshot_and_restores_domain_types(merge) -> None:
    atom = _memory_atom()
    block = make_logical_block("question", "answer")
    spec = MemoryGenerationTaskSpec(
        belong_to=make_memory_identity_scope().workspace_identity,
        from_actor=make_memory_identity_scope().actor_identity,
        topic_id="topic-codec",
        label="codec",
        source=MemoryGenerationSource.UPDATE,
        request=GenerationRequest(
            context=GenerationContext(fold_summary="original summary"),
            existing_memory=atom,
        ),
        interaction_input=InteractionArtifactInput(
            topic_id="topic-codec",
            topic_title="title",
            topic_summary="summary",
            interactions=(block,),
        ),
        intent_id="intent-codec",
        pending_alias="draft-codec",
    )
    if merge:
        spec = replace(
            spec,
            source=MemoryGenerationSource.MERGE,
            intent_id=None,
            pending_alias=None,
            interaction_input=None,
            request=GenerationRequest(
                merge_focus=MergeFocus(
                    retained_memory_id="retained",
                    absorbed_memory_id="absorbed",
                    pending_alias="draft-codec",
                    intent_id="intent-codec",
                )
            ),
        )
    codecs = WorkPayloadCodecRegistry()
    codecs.register(_MemoryGenerationWorkAdapter())
    work = _MemoryGenerationWork(task_id="task-codec", spec=spec)

    payload_bytes = codecs.encode(
        _MemoryGenerationWorkAdapter.kind,
        _MemoryGenerationWorkAdapter.schema_version,
        work,
    )

    spec.request.context.fold_summary = "external mutation"
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
    assert first.spec.belong_to == spec.belong_to
    assert first.spec.belong_to is not spec.belong_to
    assert first.spec.belong_to is not second.spec.belong_to
    if merge:
        assert first.spec.source == MemoryGenerationSource.MERGE
        assert first.spec.request.merge_focus == MergeFocus(
            retained_memory_id="retained",
            absorbed_memory_id="absorbed",
            pending_alias="draft-codec",
            intent_id="intent-codec",
        )
        first.spec.request.merge_focus.retained_memory_id = "external change"
        assert second.spec.request.merge_focus.retained_memory_id == "retained"
        return
    assert first.spec.request.context.fold_summary == "original summary"
    assert isinstance(first.spec.request.existing_memory, MemoryAtom)
    assert first.spec.request.existing_memory.payload.content == "original content"
    assert first.spec.interaction_input is not None
    assert isinstance(first.spec.interaction_input.interactions[0], LogicalBlock)
    first.spec.request.context.fold_summary = "attempt-local mutation"
    assert second.spec.request.context.fold_summary == "original summary"


@pytest.mark.parametrize("schema_version", [1, 2])
def test_registry_rejects_previous_memory_generation_versions(schema_version: int) -> None:
    """旧 scope 信封与此前的整数编号不得被误解码为新任务。"""
    registry = WorkPayloadCodecRegistry()
    adapter = _MemoryGenerationWorkAdapter()
    registry.register(adapter)
    payload = encode_canonical_json(
        {
            "task_id": "legacy",
            "spec": {"identity_scope": make_memory_identity_scope().model_dump(mode="json")},
        }
    )

    with pytest.raises(
        UnknownWorkPayloadCodecError,
        match=f"schema version {schema_version} is not registered",
    ):
        registry.decode(adapter.kind, schema_version, payload)


def _with_topic_write_memory_ids(item: dict) -> dict:
    """把当前版本保存的工作改回 1.6 之前的写法：那时本话题生成的记忆叫 topic_write_memory_ids。"""
    request = item["payload"]["spec"]["request"]
    request["topic_write_memory_ids"] = request.pop("topic_generated_memory_ids")
    return item


def _with_state_summary(item: dict) -> dict:
    """把当前版本保存的工作改回 1.5 之前的写法：那时生成材料的折叠摘要叫 state_summary。"""
    context = _with_topic_write_memory_ids(item)["payload"]["spec"]["request"]["context"]
    context["state_summary"] = context.pop("fold_summary")
    return item


def test_settlement_saved_as_1_5_reads_back_with_its_generated_memories() -> None:
    """升级前以 1.5 保存的结算里本话题生成的记忆叫 topic_write_memory_ids：读回时改名，
    记忆不变，照常执行。"""
    queue = MemoryGenerationQueue(AsyncMock())
    spec = replace(
        _spec(
            topic_id="topic-old",
            request=GenerationRequest(topic_generated_memory_ids=frozenset({"m1", "m2"})),
        ),
        source=MemoryGenerationSource.SETTLE,
        retires_topic=True,
    )
    item = _with_topic_write_memory_ids(queue.encode_for_restart("settle:topic-old", spec))

    restored = queue.decode_for_restart({**item, "schema_version": "1.5"})

    assert restored == spec


def test_work_saved_as_1_4_reads_back_with_its_fold_summary() -> None:
    """升级前以 1.4 保存的工作里折叠摘要叫 state_summary：读回时改名，摘要不变，照常执行。"""
    queue = MemoryGenerationQueue(AsyncMock())
    spec = replace(
        _spec(
            topic_id="topic-old",
            request=GenerationRequest(context=GenerationContext(fold_summary="较早内容的摘要")),
        ),
        source=MemoryGenerationSource.SETTLE,
    )
    item = _with_state_summary(queue.encode_for_restart("settle:topic-old", spec))

    restored = queue.decode_for_restart({**item, "schema_version": "1.4"})

    assert restored == spec


def test_revision_saved_as_1_1_reads_back_with_the_base_version_it_was_built_on() -> None:
    """升级前以 1.1 保存的修订没有 base_version：读回时取任务构建时读到的基础记忆版本，
    其余内容不变，下次启动后照常执行。"""
    queue = MemoryGenerationQueue(AsyncMock())
    atom = _memory_atom()
    atom.meta.version = 3
    spec = MemoryGenerationTaskSpec(
        belong_to=make_memory_identity_scope().workspace_identity,
        from_actor=make_memory_identity_scope().actor_identity,
        topic_id="topic-old",
        label="rev_memory_alias_0001",
        source=MemoryGenerationSource.UPDATE,
        request=GenerationRequest(
            context=GenerationContext(),
            update_focus=UpdateFocus(
                instruction="revise",
                base_uuid=str(atom.id),
                base_alias="memory_alias",
                base_version=3,
            ),
            existing_memory=atom,
        ),
        intent_id="intent-old",
        pending_alias="rev_memory_alias_0001",
    )
    item = _with_state_summary(queue.encode_for_restart("active:intent-old", spec))
    del item["payload"]["spec"]["retires_topic"]
    del item["payload"]["spec"]["request"]["update_focus"]["base_version"]

    restored = queue.decode_for_restart({**item, "schema_version": "1.1"})

    assert restored == spec


def test_settlement_saved_as_1_2_reads_back_without_segments_and_cannot_cite() -> None:
    """当前版本保存的结算原样读回；升级前以 1.2 保存的结算没有轮次标识、段与响应序号，读回时
    补成空值后照常执行，材料不印段标签，也就无从引用。"""
    queue = MemoryGenerationQueue(AsyncMock())
    block = make_logical_block(
        turn_events=[
            TurnEvent(kind="user_message", sequence=0, content="问题"),
            TurnEvent(kind="assistant_message", sequence=1, content="回答", response_index=1),
        ]
    )
    context = GenerationTranscriptBuilder().build_context([block])
    spec = replace(
        _spec(topic_id="topic-old", request=GenerationRequest(context=context)),
        source=MemoryGenerationSource.SETTLE,
        interaction_input=InteractionArtifactInput(topic_id="topic-old", interactions=(block,)),
    )
    item = queue.encode_for_restart("settle:topic-old", spec)
    assert queue.decode_for_restart(item) == spec

    _with_state_summary(item)
    del item["payload"]["spec"]["retires_topic"]
    for turn in item["payload"]["spec"]["request"]["context"]["turns"]:
        del turn["turn_id"], turn["segments"]
    for saved_block in item["payload"]["spec"]["interaction_input"]["blocks"]:
        for event in saved_block["turn"]["turn_events"]:
            del event["response_index"]
    restored = queue.decode_for_restart({**item, "schema_version": "1.2"})

    [turn] = restored.request.context.turns
    assert (turn.turn_id, turn.segments) == ("", [])
    assert turn.material_entries == context.turns[0].material_entries
    assert restored.request.context.cited_segments(["1.1"]) == []
    [restored_block] = restored.interaction_input.interactions
    assert [event.response_index for event in restored_block.turn_events] == [None, None]


def test_work_saved_as_1_3_reads_back_with_every_settlement_retiring() -> None:
    """当前版本保存的、话题停下后的沉淀原样读回、不退役；升级前以 1.3 保存的工作不带 retires_topic：
    那时每次结算都结束话题，结算读回为退役，其余来源读回为不退役。"""
    queue = MemoryGenerationQueue(AsyncMock())
    settlement = replace(_spec(topic_id="topic-old"), source=MemoryGenerationSource.SETTLE)
    write = _spec(topic_id="topic-old", intent_id="intent-old", pending_alias="draft-old")
    settlement_item = queue.encode_for_restart("settle:topic-old", settlement)
    write_item = queue.encode_for_restart("active:intent-old", write)
    assert queue.decode_for_restart(settlement_item) == settlement

    restored = []
    for item in (settlement_item, write_item):
        _with_state_summary(item)
        del item["payload"]["spec"]["retires_topic"]
        restored.append(queue.decode_for_restart({**item, "schema_version": "1.3"}))

    assert restored == [replace(settlement, retires_topic=True), write]


def test_codec_roundtrips_settlement_records_without_loss() -> None:
    """settle 冻结的交互、折叠记录与 binding refs 必须原样通过 codec/retry 保留。"""
    binding = TopicAssetBinding(
        asset_ref=WorkspaceAssetRef(token="token-1", asset_id="asset-1"),
        first_bound_interaction_id="i1",
        bound_at=datetime.now(UTC),
    )
    folded, retained = make_logical_block("q1", "a1"), make_logical_block("q2", "a2")
    fold = TopicFold(folded_block_ids=(folded.block_id,), summary="折叠摘要", folded_at=1.0)
    spec = _spec(topic_id="topic-bind")
    spec = replace(
        spec,
        interaction_input=InteractionArtifactInput(
            topic_id="topic-bind",
            interactions=(folded, retained),
            folds=(fold,),
            asset_bindings=(binding,),
        ),
    )
    adapter = _MemoryGenerationWorkAdapter()
    work = _MemoryGenerationWork(task_id="task-bind", spec=spec)

    encoded = adapter.encode(work)
    decoded = adapter.decode(encoded)

    assert decoded.spec.interaction_input is not None
    assert decoded.spec.interaction_input.interactions == (folded, retained)
    assert decoded.spec.interaction_input.folds == (fold,)
    assert decoded.spec.interaction_input.asset_bindings == (binding,)
    assert decoded.spec.interaction_input.asset_bindings[0].asset_ref.token == "token-1"


@pytest.mark.asyncio
async def test_queue_identity_does_not_partition_by_payload_scope() -> None:
    """捕获 IdentityScope 被用于 memory work 调度与幂等 key 的缺陷。"""
    queue = MemoryGenerationQueue(AsyncMock(return_value=[]))
    main_spec = _spec(intent_id="intent-shared", pending_alias="draft-shared")
    isolated_spec = replace(
        main_spec,
        belong_to=make_memory_identity_scope(workspace_id="isolation_workspace").workspace_identity,
    )

    try:
        handle = await queue.submit("task-shared", main_spec)
        with pytest.raises(ValueError, match="different payload"):
            await queue.submit("task-shared", isolated_spec)
    finally:
        await queue.stop()

    assert handle.work_id == "memory_generation:task-shared"


def test_memory_generation_codec_rejects_flattened_owner_projection() -> None:
    """捕获 codec 接受 TaskSpec 内第二份 owner_user_id 身份事实的缺陷。"""
    adapter = _MemoryGenerationWorkAdapter()
    encoded = adapter.encode(_MemoryGenerationWork(task_id="tampered-task", spec=_spec()))
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
    encoded = adapter.encode(_MemoryGenerationWork(task_id="nested-tamper", spec=_spec()))
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

    async def execute(route, spec, on_commit_started):
        if spec.label == "first":
            first_started.set()
            await release.wait()
        return []

    bus = make_generation_bus(execute)
    controller = _controller(bus, max_concurrency=1)
    await controller.start()

    try:
        first = await controller.submit_generation(_spec(label="first"))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = await controller.submit_generation(_spec(label="second"))
        await asyncio.sleep(0.03)

        second_work = await controller.queue.get(controller.queue.work_id_for(second.task_id))
        assert second_work is not None
        assert second_work.state == WorkState.QUEUED
        assert second.status == MemoryGenerationTaskStatus.PENDING
        assert await controller.get_task(second.task_id) == second

        release.set()
        completed = await controller.wait_all(timeout=2)
        assert len(completed) == 2
        assert all(task.status == MemoryGenerationTaskStatus.COMPLETED for task in completed)
        assert (
            await controller.get_task(first.task_id)
        ).status == MemoryGenerationTaskStatus.COMPLETED
        assert (
            await controller.get_task(second.task_id)
        ).status == MemoryGenerationTaskStatus.COMPLETED
    finally:
        release.set()
        await controller.stop()


@pytest.mark.asyncio
async def test_settlements_of_one_topic_run_in_order_while_its_writes_do_not_wait() -> None:
    """同一话题的结算依次执行：前一次还在执行时，后一次排在队列中，执行时才读得到前一次
    生成的记忆；同一话题的主动写入不受影响，照常并行。"""
    write_started = asyncio.Event()
    release = asyncio.Event()

    async def execute(route, spec, on_commit_started):
        if spec.source == MemoryGenerationSource.WRITE:
            write_started.set()
        else:
            await release.wait()
        return []

    bus = make_generation_bus(execute)
    controller = _controller(bus, max_concurrency=3)
    await controller.start()
    settle = replace(
        _spec(),
        source=MemoryGenerationSource.SETTLE,
        interaction_input=InteractionArtifactInput(topic_id="topic-1"),
    )

    try:
        await controller.submit_generation(replace(settle, label="idle"))
        retire = await controller.submit_generation(
            replace(settle, label="retire", retires_topic=True)
        )
        await controller.submit_generation(
            _spec(label="write", intent_id="intent-w", pending_alias="draft-w")
        )
        # 写入排在第二次结算之后：它开始执行时，调度已经越过了第二次结算。
        await asyncio.wait_for(write_started.wait(), timeout=1)

        retire_work = await controller.queue.get(controller.queue.work_id_for(retire.task_id))
        assert retire_work.state == WorkState.QUEUED

        release.set()
        completed = await controller.wait_all(timeout=2)
        assert [task.status for task in completed] == [MemoryGenerationTaskStatus.COMPLETED] * 3
    finally:
        release.set()
        await controller.stop()


@pytest.mark.asyncio
async def test_queued_cancel_never_calls_generation_handler() -> None:
    first_started = asyncio.Event()
    release = asyncio.Event()
    called_labels: list[str] = []

    async def execute(route, spec, on_commit_started):
        called_labels.append(spec.label)
        if spec.label == "first":
            first_started.set()
            await release.wait()
        return []

    bus = make_generation_bus(execute)
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

    async def execute(route, spec, on_commit_started):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    bus = make_generation_bus(execute)
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
    controller = MemoryGenerationTaskController(bus=Mock(request=AsyncMock(), publish=AsyncMock()))

    with pytest.raises(WorkQueueStoppedError, match="must be started"):
        await controller.submit_generation(_spec())


@pytest.mark.asyncio
async def test_default_policy_does_not_retry_generation_side_effects() -> None:
    execute = AsyncMock(
        side_effect=ConnectionError("generation may already have written partial results")
    )
    controller = MemoryGenerationTaskController(bus=make_generation_bus(execute))
    await controller.start()

    try:
        memory_task = await controller.submit_generation(_spec())
        result = await controller.wait_task(memory_task.task_id, timeout=1)
    finally:
        await controller.stop()

    assert result.status == MemoryGenerationTaskStatus.FAILED
    assert execute.await_count == 1
    assert (
        await controller.get_task(memory_task.task_id)
    ).status == MemoryGenerationTaskStatus.FAILED


@pytest.mark.asyncio
async def test_non_retryable_failure_fails_once() -> None:
    execute = AsyncMock(side_effect=ValueError("invalid generation spec"))
    controller = _controller(make_generation_bus(execute))
    await controller.start()

    try:
        memory_task = await controller.submit_generation(_spec())
        result = await controller.wait_task(memory_task.task_id, timeout=1)
    finally:
        await controller.stop()

    assert result.status == MemoryGenerationTaskStatus.FAILED
    assert result.error == "记忆生成失败，请检查服务状态"
    assert execute.await_count == 1


@pytest.mark.asyncio
async def test_timeout_fails_without_retry() -> None:
    async def never_returns(route, spec, on_commit_started):
        await asyncio.Event().wait()

    execute = AsyncMock(side_effect=never_returns)
    controller = _controller(
        make_generation_bus(execute),
        timeout_seconds=0.01,
    )
    await controller.start()

    try:
        memory_task = await controller.submit_generation(_spec())
        result = await controller.wait_task(memory_task.task_id, timeout=1)
        record = await controller.queue.get(controller.queue.work_id_for(memory_task.task_id))
    finally:
        await controller.stop()

    assert result.status == MemoryGenerationTaskStatus.FAILED
    assert result.error == "记忆生成失败，请检查服务状态"
    assert record is not None
    assert record.state == WorkState.FAILED
    assert record.last_error.message == "memory_generation_timed_out"
    assert execute.await_count == 1


def _merge_aborted() -> MemoryMergeAbortedError:
    return MemoryMergeAbortedError(details={"reason": "version_changed", "memory_id": "m1"})


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (_merge_aborted(), "merge_aborted.version_changed"),
        (ModelContextWindowExceededError("window"), "model_context_window_exceeded"),
        (ModelProviderError("rate limited"), "model_provider_failed"),
        (TimeoutError(), "memory_generation_timed_out"),
        (MemoryAliasConflictError(), "workspace.memory.alias_conflict"),
        (ValueError("bug"), "memory_generation_failed"),
    ],
    ids=["merge-aborted", "context-window", "provider", "timeout", "domain-error", "unexpected"],
)
def test_failure_reason_maps_each_error_kind(error, reason) -> None:
    """合并中止带拒绝原因，超窗先于提供商失败判断，其他领域错误取自己的 code。"""
    assert memory_generation_failure_reason(error) == reason


@pytest.mark.parametrize(
    ("error", "with_stack", "logged_detail"),
    [
        (ValueError("bug"), True, "details=None"),
        (_merge_aborted(), False, "'memory_id': 'm1'"),
    ],
    ids=["unexpected", "domain-error"],
)
def test_classifier_logs_each_failure_once_with_reason(
    caplog, error, with_stack, logged_detail
) -> None:
    """分类处记一次带原因码的 error 日志：未预期的异常带堆栈，领域错误带 details。"""
    handler = MemoryGenerationHandler(AsyncMock(), handles={})
    context = WorkExecutionContext(
        work_id="memory_generation:t1",
        lane=MemoryGenerationQueue.LANE,
        kind="patchouli.memory_generation",
        schema_version="1.1",
        attempt_count=1,
    )

    with caplog.at_level(logging.ERROR, logger="hivememory.patchouli.control.memory_generation"):
        decision = handler.classify_failure(error, context)

    (record,) = caplog.records
    message = record.getMessage()
    assert record.levelno == logging.ERROR
    assert f"reason={decision.reason}" in message
    assert "work_id=memory_generation:t1" in message
    assert logged_detail in message
    assert (record.exc_info is not None) is with_stack
