"""CALL 上下文的编译与操作边界错误映射。"""

from datetime import UTC, datetime

import pytest

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.alice.orchestration.sub_agent import CallContextProvider
from hivememory.core.errors import OperationDeniedError, ResourceUnavailableError
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
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    BusRouteUnavailableError,
    PermissionDeniedError,
    SystemFault,
)
from hivememory.workspace.contracts import GetAgentProfileRequest, ResolveReferencesRequest
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_runtime_scope


def _frame(*, profile: AgentProfile = OMNI_DOLL_PROFILE, submit_operation=None) -> ExecutionFrame:
    return ExecutionFrame(
        submit_operation=submit_operation,
        runtime_scope=make_runtime_scope(run_id="run-1", frame_id="frame-1"),
        agent_profile=profile,
        working_history=[],
        topic_id="topic-1",
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
        if isinstance(request, GetAgentProfileRequest):
            if profile_error is not None:
                raise profile_error
            return profile
        if not isinstance(request, ResolveReferencesRequest):
            raise TypeError(type(request))
        result = next(results)
        if isinstance(result, Exception):
            raise result
        assert request.aliases == (result.requested_alias,)
        return [result]

    return CallContextProvider(), submit_operation


@pytest.mark.asyncio
async def test_provide_compiles_atom_context_ref_for_callee():
    resolved = ReferenceResolution(
        kind="atom",
        requested_alias="fact_a",
        atom=_atom("Fact A", "context payload"),
    )
    provider, submit_operation = _provider(alias_results=[resolved])
    caller = _frame(
        submit_operation=submit_operation,
        profile=OMNI_DOLL_PROFILE.model_copy(update={"language": "en"}),
    )

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
        alias_results=[RuntimeError("storage unavailable"), resolved]
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
@pytest.mark.parametrize(
    ("error", "expected_type", "expected_code", "expected_key"),
    [
        (
            OperationDeniedError("profile.read 未获授权"),
            PermissionDeniedError,
            "mtp.permission.denied",
            "mtp.permission.verb_denied",
        ),
        (
            ResourceUnavailableError("Profile 存储路由不可达"),
            BusRouteUnavailableError,
            "mtp.system.service_unavailable",
            "mtp.system.service_unavailable",
        ),
        (
            AliasNotFoundError(message_key="mtp.call.profile_not_found"),
            AliasNotFoundError,
            "mtp.alias.not_found",
            "mtp.call.profile_not_found",
        ),
        (
            RuntimeError("profile unavailable"),
            SystemFault,
            "mtp.system.fault",
            "mtp.call.profile_load_failed",
        ),
    ],
)
async def test_profile_failure_maps_to_mtp_error_before_resolving_context_refs(
    error, expected_type, expected_code, expected_key
):
    """Profile 准备失败保留具体错误语义，不继续读取 context_refs。"""
    provider, submit_operation = _provider(profile_error=error)

    with pytest.raises(expected_type) as exc_info:
        await provider.provide(
            _frame(submit_operation=submit_operation),
            MTPCallRequest(
                target_alias="helper",
                task="summarize",
                context_refs=["fact_a"],
            ),
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.message_key == expected_key


@pytest.mark.asyncio
async def test_call_without_submitter_fails_explicitly():
    """没有操作提交函数时不得以默认 Profile 旁路执行 CALL。"""
    with pytest.raises(RuntimeError, match="CALL 缺少操作提交函数"):
        await CallContextProvider().provide(
            _frame(), MTPCallRequest(target_alias="omni_doll", task="summarize")
        )
