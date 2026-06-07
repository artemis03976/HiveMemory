from datetime import datetime

import pytest

from hivememory.agent_runtime.resolver import ResolveResult
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    PendingAtom,
    PendingAtomResolution,
    PendingAtomSettlement,
    PendingAtomStatus,
    UpdateFocus,
    VerificationStatus,
    WriteFocus,
)
from hivememory.engines.memory_compiler.builders.memory_atom import build_memory_atom_ir
from hivememory.engines.memory_compiler.builders.pending_atom import build_pending_atom_ir
from hivememory.engines.memory_compiler.builders.resolve_result import build_resolve_result_ir


@pytest.fixture
def memory_atom() -> MemoryAtom:
    return MemoryAtom(
        index=IndexLayer(
            title="API endpoint",
            summary="Endpoint summary",
            memory_type=MemoryType.FACT,
            tags=["api", "backend"],
            alias="fact_api",
        ),
        payload=PayloadLayer(content="Use /api/v1/memories", history_summary=["created"]),
        meta=MetaData(
            source_agent_id="agent",
            user_id="u1",
            updated_at=datetime(2026, 1, 1),
            confidence_score=0.8,
            verification_status=VerificationStatus.VERIFIED,
        ),
    )


def test_build_memory_atom_ir_maps_identity_content_and_metadata(memory_atom):
    unit = build_memory_atom_ir(memory_atom)

    assert unit.identity.source_kind == "atom"
    assert unit.identity.alias == "fact_api"
    assert unit.identity.memory_id == str(memory_atom.id)
    assert unit.content.title == "API endpoint"
    assert unit.content.summary == "Endpoint summary"
    assert unit.content.content == "Use /api/v1/memories"
    assert set(unit.content.tags) == {"api", "backend"}
    assert unit.content.memory_type == MemoryType.FACT.value
    assert unit.metadata["confidence_score"] == 0.8
    assert unit.metadata["verification_status"] == VerificationStatus.VERIFIED
    assert unit.metadata["history_summary"] == ["created"]


def test_build_pending_atom_ir_for_write_focus():
    pending = PendingAtom(
        pending_alias="draft_1",
        intent_id="intent-1",
        status=PendingAtomStatus.PENDING,
        source_verb="WRITE",
        focus=WriteFocus(title="Title", content="Body"),
    )

    unit = build_pending_atom_ir(pending)

    assert unit.identity.source_kind == "pending"
    assert unit.identity.alias == "draft_1"
    assert unit.content.title == "Title"
    assert unit.content.content == "Body"
    assert unit.status.source_state == "pending"
    assert unit.status.source_verb == "WRITE"
    assert unit.status.is_terminal is False


def test_build_pending_atom_ir_for_update_focus_and_discarded_settlement():
    pending = PendingAtom(
        pending_alias="rev_1",
        intent_id="intent-2",
        status=PendingAtomStatus.SETTLED,
        source_verb="UPDATE",
        focus=UpdateFocus(
            base_alias="fact_api",
            base_uuid="uuid-1",
            instruction="Revise it",
            content="New body",
        ),
        settlement=PendingAtomSettlement(
            pending_alias="rev_1",
            intent_id="intent-2",
            resolution=PendingAtomResolution.DISCARDED,
            message="duplicate",
            reason="already covered",
        ),
    )

    unit = build_pending_atom_ir(pending)

    assert unit.content.instruction == "Revise it"
    assert unit.content.content == "New body"
    assert unit.metadata["base_alias"] == "fact_api"
    assert unit.status.is_terminal is True
    assert unit.status.is_discarded is True
    assert unit.status.message == "duplicate"
    assert unit.status.reason == "already covered"


def test_build_resolve_result_ir_for_redirect_uses_canonical_content(memory_atom):
    resolve = ResolveResult(
        kind="redirect",
        requested_alias="draft_old",
        canonical_alias="fact_api",
        atom=memory_atom,
    )

    unit = build_resolve_result_ir(resolve)

    assert unit.identity.source_kind == "resolve_result"
    assert unit.identity.alias == "fact_api"
    assert unit.identity.redirected_from == "draft_old"
    assert unit.status.is_redirect is True
    assert unit.content.content == "Use /api/v1/memories"


def test_build_resolve_result_ir_for_terminal_result():
    resolve = ResolveResult(
        kind="failed",
        requested_alias="draft_failed",
        settlement=PendingAtomSettlement(
            pending_alias="draft_failed",
            intent_id="intent-3",
            resolution=PendingAtomResolution.CREATED,
            message="message",
            reason="reason",
            error="boom",
        ),
    )

    unit = build_resolve_result_ir(resolve)

    assert unit.identity.alias == "draft_failed"
    assert unit.status.is_terminal is True
    assert unit.status.error == "boom"
    assert unit.status.message == "message"
    assert unit.status.reason == "reason"


def test_build_resolve_result_ir_rejects_unsupported_kind():
    with pytest.raises(ValueError, match="not_found"):
        build_resolve_result_ir(ResolveResult(kind="not_found", requested_alias="missing"))
