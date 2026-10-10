from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.alice.orchestration.sub_agent import CallContextProvider
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
from hivememory.workspace.contracts import ResolveReferencesRequest
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


def _provider(*, profile=OMNI_DOLL_PROFILE, alias_results=()) -> tuple:
    profile_resolver = MagicMock()
    profile_resolver.resolve = AsyncMock(return_value=profile)
    results = iter(alias_results)

    async def alias_resolver(request: ResolveReferencesRequest):
        """引用读取以请求中的 alias 选择既有测试结果，模拟外部读取边界。"""
        result = next(results)
        if isinstance(result, Exception):
            raise result
        assert request.aliases == (result.requested_alias,)
        return [result]

    return (
        CallContextProvider(profile_resolver),
        profile_resolver,
        alias_resolver,
    )


@pytest.mark.asyncio
async def test_provide_resolves_profile_with_caller_identity_and_skips_empty_refs():
    provider, profile_resolver, alias_resolver = _provider()
    caller = _frame(submit_operation=alias_resolver)

    context = await provider.provide(
        caller,
        MTPCallRequest(target_alias="helper", task="summarize"),
    )

    assert context.shared_context == ""
    profile_resolver.resolve.assert_awaited_once_with(
        "helper",
        identity_scope=caller.identity_scope,
    )


@pytest.mark.asyncio
async def test_provide_compiles_atom_context_ref_for_callee():
    resolved = ReferenceResolution(
        kind="atom",
        requested_alias="fact_a",
        atom=_atom("Fact A", "context payload"),
    )
    provider, _, alias_resolver = _provider(alias_results=[resolved])
    caller = _frame(
        submit_operation=alias_resolver,
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
    provider, _, alias_resolver = _provider(alias_results=[resolved])

    context = await provider.provide(
        _frame(submit_operation=alias_resolver),
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
    provider, _, alias_resolver = _provider(
        alias_results=[RuntimeError("storage unavailable"), resolved]
    )

    context = await provider.provide(
        _frame(submit_operation=alias_resolver),
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
    provider, _, alias_resolver = _provider(
        alias_results=[ReferenceResolution(kind="not_found", requested_alias="missing")]
    )

    context = await provider.provide(
        _frame(submit_operation=alias_resolver),
        MTPCallRequest(
            target_alias="helper",
            task="summarize",
            context_refs=["missing"],
        ),
    )

    assert context.shared_context == ""


@pytest.mark.asyncio
async def test_provide_propagates_profile_resolution_failure_before_resolving_refs():
    error = RuntimeError("profile unavailable")
    provider, profile_resolver, alias_resolver = _provider()
    profile_resolver.resolve.side_effect = error

    with pytest.raises(RuntimeError, match="profile unavailable"):
        await provider.provide(
            _frame(submit_operation=alias_resolver),
            MTPCallRequest(
                target_alias="helper",
                task="summarize",
                context_refs=["fact_a"],
            ),
        )
