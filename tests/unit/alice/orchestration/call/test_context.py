"""CALL 上下文的编译、操作边界错误映射与派生后失败时的子线程结束。"""

import asyncio
from datetime import UTC, datetime

import pytest

from hivememory.agent_runtime.frame import ExecutionFrame, FrameExecutionPolicy
from hivememory.alice.orchestration.call import CallContextProvider
from hivememory.core.errors import (
    OperationDeniedError,
    ResourceNotFoundError,
    StorageOfflineError,
    ThreadEndedError,
)
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.models.reference import ReferenceResolution
from hivememory.core.mtp import MTPCallRequest
from hivememory.workspace.contracts import DeriveCallThreadRequest, ResolveReferencesRequest
from tests.helpers.call import StubOperationSubmitter, make_callee
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_runtime_scope


def _frame(
    *, submit_operation, profile: AgentProfile = OMNI_DOLL_PROFILE, language: str = "zh"
) -> ExecutionFrame:
    return ExecutionFrame(
        execution_policy=FrameExecutionPolicy.from_verbs(
            {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
        ),
        submit_operation=submit_operation,
        runtime_scope=make_runtime_scope(run_id="run-1", frame_id="frame-1"),
        agent_profile=profile,
        working_history=[],
        topic_id="topic-1",
        language=language,
    )


def _atom(title: str, content: str) -> MemoryAtom:
    return MemoryAtom(
        index=IndexLayer(
            title=title,
            summary=f"{title} summary",
            memory_type=MemoryType.FACT,
            tags=["context"],
        ),
        payload=PayloadLayer(content=content),
        meta=make_memory_metadata(
            source_agent_id="caller",
            user_id="user-1",
            updated_at=datetime.now(UTC),
            confidence_score=0.9,
        ),
    )


def _provider(*, profile=OMNI_DOLL_PROFILE, alias_results=(), profile_error=None) -> tuple:
    results = iter(alias_results)

    async def submit_operation(request):
        """只替代 workspace 操作边界，CALL 的请求构造与上下文编译保持真实。"""
        if not isinstance(request, ResolveReferencesRequest):
            raise TypeError(type(request))
        result = next(results)
        if isinstance(result, Exception):
            raise result
        assert request.aliases == (result.requested_alias,)
        return [result]

    async def derive(request):
        if not isinstance(request, DeriveCallThreadRequest):
            raise TypeError(type(request))
        if profile_error is not None:
            raise profile_error
        return make_callee(profile)

    return CallContextProvider(), StubOperationSubmitter(operation=submit_operation, derive=derive)


@pytest.mark.asyncio
async def test_provide_compiles_atom_context_ref_for_callee():
    resolved = ReferenceResolution(
        kind="atom",
        requested_alias="fact_a",
        atom=_atom("Fact A", "context payload"),
    )
    provider, submit_operation = _provider(alias_results=[resolved])
    # 共享上下文按调用方帧的语言编译，Profile 的语言设置不参与。
    caller = _frame(submit_operation=submit_operation, language="en")

    context = await provider.provide(
        caller,
        MTPCallRequest(
            target_alias="helper",
            task="summarize",
            context_refs=["fact_a"],
        ),
    )

    assert context.shared_context.startswith("[Shared Context from Parent Agent]")
    assert "Use READ" in context.shared_context
    assert '<memory alias="' in context.shared_context
    assert "Fact A" in context.shared_context
    assert "context payload" in context.shared_context
    # 子线程交给子帧继续运行，准备成功时不结束。
    assert context.callee is submit_operation.children[0]
    assert context.callee.end_calls == 0


@pytest.mark.asyncio
async def test_provide_compiles_redirected_context_ref_as_canonical_atom():
    resolved = ReferenceResolution(
        kind="redirect",
        requested_alias="draft_ctx_1234",
        canonical_alias="fact_canonical",
        atom=_atom("Canonical Fact", "canonical context"),
    )
    provider, submit_operation = _provider(alias_results=[resolved])

    context = await provider.provide(
        _frame(submit_operation=submit_operation),
        MTPCallRequest(
            target_alias="helper",
            task="summarize",
            context_refs=["draft_ctx_1234"],
        ),
    )

    assert "Canonical Fact" in context.shared_context
    assert "canonical context" in context.shared_context
    assert "<memory alias=" in context.shared_context


@pytest.mark.asyncio
async def test_provide_keeps_resolvable_refs_when_one_resolution_fails():
    resolved = ReferenceResolution(
        kind="atom",
        requested_alias="fact_b",
        atom=_atom("Fact B", "usable context"),
    )
    provider, submit_operation = _provider(
        alias_results=[StorageOfflineError("storage unavailable"), resolved]
    )

    context = await provider.provide(
        _frame(submit_operation=submit_operation),
        MTPCallRequest(
            target_alias="helper",
            task="summarize",
            context_refs=["fact_a", "fact_b"],
        ),
    )

    assert "Fact B" in context.shared_context
    assert "usable context" in context.shared_context


@pytest.mark.asyncio
async def test_provide_returns_empty_context_when_no_ref_can_be_rendered():
    provider, submit_operation = _provider(
        alias_results=[ReferenceResolution(kind="not_found", requested_alias="missing")]
    )

    context = await provider.provide(
        _frame(submit_operation=submit_operation),
        MTPCallRequest(
            target_alias="helper",
            task="summarize",
            context_refs=["missing"],
        ),
    )

    assert context.shared_context == ""


@pytest.mark.asyncio
async def test_profile_failure_propagates_before_resolving_context_refs():
    """派生子线程失败保留原始原因，不继续共享引用读取。"""
    error = ResourceNotFoundError("missing profile")
    provider, submit_operation = _provider(profile_error=error)
    with pytest.raises(ResourceNotFoundError) as caught:
        await provider.provide(
            _frame(submit_operation=submit_operation),
            MTPCallRequest(target_alias="helper", task="summarize", context_refs=["fact_a"]),
        )
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        ThreadEndedError("thread ended"),
        OperationDeniedError("read denied"),
        TypeError("resolver defect"),
    ],
)
async def test_nonrecoverable_error_stops_shared_context_resolution(failure):
    """生命周期、授权与程序错误不能被当作单条引用缺失而跳过；已派生的子线程随之结束。"""
    provider, submit_operation = _provider(alias_results=[failure])
    with pytest.raises(type(failure)) as caught:
        await provider.provide(
            _frame(submit_operation=submit_operation),
            MTPCallRequest(
                target_alias="helper", task="summarize", context_refs=["first", "second"]
            ),
        )
    assert caught.value is failure
    (child,) = submit_operation.children
    assert child.end_calls == 1


@pytest.mark.asyncio
async def test_cancelled_shared_context_read_ends_child_thread():
    """共享上下文读取被取消时子帧不会运行：同步结束已派生的子线程，取消照常传播。"""
    entered = asyncio.Event()

    async def hanging_read(_request):
        entered.set()
        await asyncio.Event().wait()

    submit_operation = StubOperationSubmitter(
        operation=hanging_read, derive=lambda _request: _derived(make_callee())
    )
    task = asyncio.create_task(
        CallContextProvider().provide(
            _frame(submit_operation=submit_operation),
            MTPCallRequest(target_alias="helper", task="summarize", context_refs=["fact_a"]),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (child,) = submit_operation.children
    assert child.end_calls == 1


async def _derived(child):
    """派生替身：立即交回已准备好的子线程句柄。"""
    return child
