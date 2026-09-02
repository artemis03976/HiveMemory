from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.aliases import ResolveResult
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
from hivememory.core.mtp import MTPCallRequest
from tests.helpers.workspace import make_runtime_scope
from tests.helpers.memory import make_memory_metadata


def _frame(*, profile: AgentProfile = OMNI_DOLL_PROFILE) -> ExecutionFrame:
    return ExecutionFrame(
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
            updated_at=datetime.now(),
            confidence_score=0.9,
        ),
    )


def _provider(*, profile=OMNI_DOLL_PROFILE, alias_results=()) -> tuple:
    profile_resolver = MagicMock()
    profile_resolver.resolve = AsyncMock(return_value=profile)
    alias_resolver = MagicMock()
    alias_resolver.resolve = AsyncMock(side_effect=list(alias_results))
    return (
        CallContextProvider(profile_resolver, alias_resolver),
        profile_resolver,
        alias_resolver,
    )


@pytest.mark.asyncio
async def test_provide_resolves_profile_with_caller_identity_and_skips_empty_refs():
    provider, profile_resolver, alias_resolver = _provider()
    caller = _frame()

    context = await provider.provide(
        caller,
        MTPCallRequest(target_alias="helper", task="summarize"),
    )

    assert context.shared_context == ""
    profile_resolver.resolve.assert_awaited_once_with(
        "helper",
        identity_scope=caller.identity_scope,
    )
    alias_resolver.resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_provide_compiles_atom_context_ref_for_callee():
    resolved = ResolveResult(
        kind="atom",
        requested_alias="fact_a",
        atom=_atom("Fact A", "context payload"),
    )
    provider, _, alias_resolver = _provider(alias_results=[resolved])
    caller = _frame(profile=OMNI_DOLL_PROFILE.model_copy(update={"language": "en"}))

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
    execution_context = alias_resolver.resolve.await_args.kwargs["context"]
    assert execution_context.identity == caller.identity


@pytest.mark.asyncio
async def test_provide_compiles_redirected_context_ref_as_canonical_atom():
    resolved = ResolveResult(
        kind="redirect",
        requested_alias="draft_ctx_1234",
        canonical_alias="fact_canonical",
        atom=_atom("Canonical Fact", "canonical context"),
    )
    provider, _, _ = _provider(alias_results=[resolved])

    context = await provider.provide(
        _frame(),
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
    resolved = ResolveResult(
        kind="atom",
        requested_alias="fact_b",
        atom=_atom("Fact B", "usable context"),
    )
    provider, _, alias_resolver = _provider(
        alias_results=[RuntimeError("storage unavailable"), resolved]
    )

    context = await provider.provide(
        _frame(),
        MTPCallRequest(
            target_alias="helper",
            task="summarize",
            context_refs=["fact_a", "fact_b"],
        ),
    )

    assert "Fact B" in context.shared_context
    assert "usable context" in context.shared_context
    assert alias_resolver.resolve.await_count == 2


@pytest.mark.asyncio
async def test_provide_returns_empty_context_when_no_ref_can_be_rendered():
    provider, _, _ = _provider(
        alias_results=[ResolveResult(kind="not_found", requested_alias="missing")]
    )

    context = await provider.provide(
        _frame(),
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
            _frame(),
            MTPCallRequest(
                target_alias="helper",
                task="summarize",
                context_refs=["fact_a"],
            ),
        )

    alias_resolver.resolve.assert_not_awaited()
