"""CPU 端口的执行准入、终态返回、事件推送与线程操作绑定。"""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from hivememory.alice.interface.cpu import AliceCPU, AliceUnavailableError
from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.models.reference import ReferenceResolution
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    OperationRequest,
    ResolveReferencesRequest,
)
from tests.helpers.chat_handoff import make_input_manifest, make_mtp_turn_events
from tests.helpers.sessions import make_compaction_checkpoint
from tests.helpers.workspace import make_access_composition, make_actor_access_record


class _ReferenceEntry:
    """线程 context 绑定测试的外部入口替身，按线程 context 返回对应的引用结果。"""

    def __init__(self) -> None:
        self.labels: dict[WorkspaceAccessContext, str] = {}

    async def execute[R](
        self, request: OperationRequest[R], *, access: WorkspaceAccessContext
    ) -> R:
        if not isinstance(request, ResolveReferencesRequest):
            raise TypeError(f"Unsupported request: {type(request).__name__}")
        return cast(
            R,
            [
                ReferenceResolution(
                    kind="not_found", requested_alias=f"{self.labels[access]}:{alias}"
                )
                for alias in request.aliases
            ],
        )


def _cpu(run_agent, entry=None):
    cpu = AliceCPU(SimpleNamespace(run_agent=run_agent), entry or _ReferenceEntry())
    cpu.start()
    return cpu


async def _access():
    return await make_access_composition([make_actor_access_record()]).authenticate_thread()


async def _discard(_event: dict[str, Any]) -> None:
    return None


@pytest.mark.asyncio
async def test_cpu_returns_service_result_and_pushes_run_output_to_sink():
    """终态是运行服务返回的同一结果对象；运行输出端口发出的事件推送到进程给出的 sink。"""
    expected = CPUExecutionResult(
        final_text="完成",
        model_used="glm-4",
        turn_events=make_mtp_turn_events(),
        compaction_checkpoint=make_compaction_checkpoint(reduced_through="old"),
    )

    async def run_agent(manifest, *, run_output, generation_options, **kwargs):
        assert manifest.process_id == "process-1"
        assert generation_options == {"temperature": 0.5}
        run_output.bind_run("run-1")
        await run_output.context_compaction(status="started", data={})
        return expected

    received: list[dict[str, Any]] = []

    async def sink(event: dict[str, Any]) -> None:
        received.append(event)

    result = await _cpu(run_agent).execute(
        make_input_manifest(process_id="process-1"),
        generation_options={"temperature": 0.5},
        access=await _access(),
        events=sink,
    )
    assert result is expected
    assert received == [
        {
            "event": "context_compaction",
            "data": {"agent_run_id": "run-1", "status": "started", "stream_sequence": 0},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [False, True])
async def test_unavailable_cpu_fails_when_awaited_without_entering_service(stopped):
    """未启动与已停止的 CPU 都在 await 执行时以稳定错误拒绝，不进入运行服务。"""

    async def run_agent(*args, **kwargs):
        raise AssertionError("不可用的 CPU 不应进入运行服务")

    cpu = AliceCPU(SimpleNamespace(run_agent=run_agent), _ReferenceEntry())
    if stopped:
        cpu.start()
        cpu.stop()
    execution = cpu.execute(
        make_input_manifest(), generation_options=None, access=await _access(), events=_discard
    )
    with pytest.raises(AliceUnavailableError, match="Alice CPU is not running") as caught:
        await execution
    assert caught.value.code == "alice.unavailable"
    assert caught.value.details == {"reason": "not_running"}


@pytest.mark.asyncio
async def test_cpu_binds_each_thread_context_to_its_operation_submitter() -> None:
    """每次执行的提交函数使用自己的线程 context，不能被下一次执行的线程 context 覆盖。"""
    entry = _ReferenceEntry()
    first_access = await make_access_composition([make_actor_access_record()]).authenticate_thread()
    second_access = await make_access_composition(
        [make_actor_access_record()]
    ).authenticate_thread()
    entry.labels[first_access] = "first"
    entry.labels[second_access] = "second"
    submitters = []

    async def route(_manifest, *, submit_operation, **_kwargs):
        submitters.append(submit_operation)
        references = await submit_operation(ResolveReferencesRequest(("fact",)))
        return CPUExecutionResult(final_text=references[0].requested_alias)

    cpu = _cpu(route, entry)
    results = [
        await cpu.execute(
            make_input_manifest(), access=access, generation_options=None, events=_discard
        )
        for access in (first_access, second_access)
    ]

    assert [result.final_text for result in results] == ["first:fact", "second:fact"]
    first_references = await submitters[0](ResolveReferencesRequest(("again",)))
    assert first_references[0].requested_alias == "first:again"
