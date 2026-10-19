"""CPU 端口的执行准入、类型化结果交付与线程操作绑定。"""

from types import SimpleNamespace
from typing import cast

import pytest

from hivememory.alice.interface.cpu import AliceCPU, AliceUnavailableError
from hivememory.alice.interface.stream import AgentRunStreamAdapter
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
    cpu = AliceCPU(
        SimpleNamespace(run_agent=run_agent),
        entry or _ReferenceEntry(),
        stream_adapter=AgentRunStreamAdapter(),
    )
    cpu.start()
    return cpu


async def _access():
    return await make_access_composition([make_actor_access_record()]).authenticate_thread()


@pytest.mark.asyncio
async def test_cpu_delivers_typed_result_and_checkpoint_without_serialization():
    """终态交回运行服务的同一结果对象，检查点与轮次事件无需序列化还原。"""
    expected = CPUExecutionResult(
        final_text="完成",
        model_used="glm-4",
        turn_events=make_mtp_turn_events(),
        compaction_checkpoint=make_compaction_checkpoint(reduced_through="old"),
    )

    async def run_agent(manifest, *, run_output, result_slot, generation_options, **kwargs):
        assert manifest.process_id == "process-1"
        assert generation_options == {"temperature": 0.5}
        run_output.bind_run("run-1")
        result_slot.result = expected
        return expected

    handle = _cpu(run_agent).execute(
        make_input_manifest(process_id="process-1"),
        generation_options={"temperature": 0.5},
        access=await _access(),
    )
    items = [item async for item in handle.output]
    assert items == [expected]
    assert items[0] is expected
    assert handle.result_slot.result is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [False, True])
async def test_unavailable_cpu_fails_at_first_pull_without_entering_service(stopped):
    """未启动与已停止的 CPU 都先返回句柄，拉取时以稳定错误拒绝执行。"""

    async def run_agent(*args, **kwargs):
        raise AssertionError("不可用的 CPU 不应进入运行服务")

    cpu = AliceCPU(
        SimpleNamespace(run_agent=run_agent),
        _ReferenceEntry(),
        stream_adapter=AgentRunStreamAdapter(),
    )
    if stopped:
        cpu.start()
        cpu.stop()
    handle = cpu.execute(make_input_manifest(), generation_options=None, access=await _access())
    with pytest.raises(AliceUnavailableError, match="Alice CPU is not running") as caught:
        await anext(handle.output)
    assert caught.value.code == "alice.unavailable"
    assert caught.value.details == {"reason": "not_running"}
    assert handle.result_slot.result is None


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
    results = []
    for access in (first_access, second_access):
        results.extend(
            [
                result
                async for result in cpu.execute(
                    make_input_manifest(),
                    access=access,
                    generation_options=None,
                ).output
            ]
        )

    assert [result.final_text for result in results] == ["first:fact", "second:fact"]
    first_references = await submitters[0](ResolveReferencesRequest(("again",)))
    assert first_references[0].requested_alias == "first:again"
