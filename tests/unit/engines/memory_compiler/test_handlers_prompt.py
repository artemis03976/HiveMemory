import pytest

from hivememory.core.models import VerificationStatus
from hivememory.engines.memory_compiler.handlers.prompt import (
    _format_confidence_from_ir,
    _truncate_content,
    compile_prompt_full,
    compile_prompt_index,
    compile_shared_context,
)
from hivememory.engines.memory_compiler.ir import (
    MemoryContentIR,
    MemoryIdentityIR,
    MemoryStatusIR,
    MemoryUnitIR,
)
from hivememory.engines.memory_compiler.models import MemoryCompileOptions, MemoryCompileTarget
from hivememory.i18n import set_default_language


@pytest.fixture(autouse=True)
def reset_language():
    set_default_language("en")
    yield
    set_default_language("zh")


def _atom_unit() -> MemoryUnitIR:
    return MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="atom", alias="fact_api", memory_id="mem-1"),
        content=MemoryContentIR(
            title="API endpoint",
            summary="Use memory APIs for persistence",
            content="Full memory content",
            tags=["api"],
            memory_type="FACT",
        ),
        status=MemoryStatusIR(),
        metadata={
            "confidence_score": 0.95,
            "verification_status": VerificationStatus.VERIFIED,
            "history_summary": ["created"],
        },
    )


def test_compile_prompt_full_renders_atom_and_artifact_metadata():
    artifact = compile_prompt_full(
        _atom_unit(),
        MemoryCompileTarget.PROMPT_FULL,
        MemoryCompileOptions(language="en", requested_alias="requested"),
    )

    assert artifact.target == MemoryCompileTarget.PROMPT_FULL
    assert artifact.source_kind == "atom"
    assert artifact.alias == "fact_api"
    assert artifact.metadata["requested_alias"] == "requested"
    assert "Full memory content" in artifact.text
    assert "95% (High) [Verified]" in artifact.text


def test_compile_prompt_index_truncates_long_summary():
    unit = _atom_unit()
    unit.content.summary = "x" * 20

    artifact = compile_prompt_index(
        unit,
        MemoryCompileTarget.PROMPT_INDEX,
        MemoryCompileOptions(language="en", max_summary_length=5),
    )

    assert "xxxxx..." in artifact.text
    assert "x" * 20 not in artifact.text


def test_compile_shared_context_renders_pending_unit():
    pending = MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="pending", alias="draft_1"),
        content=MemoryContentIR(title="Draft", content="Draft content"),
        status=MemoryStatusIR(source_state="pending", source_verb="WRITE"),
    )

    artifact = compile_shared_context(
        pending,
        MemoryCompileTarget.SHARED_CONTEXT,
        MemoryCompileOptions(language="en"),
    )

    assert artifact.source_kind == "pending"
    assert "draft_1" in artifact.text
    assert "Draft content" in artifact.text


def test_compile_prompt_full_rejects_pending_source():
    pending = MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="pending", alias="draft_1"),
        content=MemoryContentIR(content="Draft content"),
        status=MemoryStatusIR(source_state="pending", source_verb="WRITE"),
    )

    with pytest.raises(ValueError, match="Unsupported source"):
        compile_prompt_full(pending, MemoryCompileTarget.PROMPT_FULL, MemoryCompileOptions())


def test_confidence_format_marks_low_unverified():
    unit = _atom_unit()
    unit.metadata["confidence_score"] = 0.5
    unit.metadata["verification_status"] = None

    assert "50% (Low)" in _format_confidence_from_ir(unit, "en")
    assert "[Unverified]" in _format_confidence_from_ir(unit, "en")


def test_truncate_content_prefers_sentence_boundary():
    content = "First paragraph.\n\nSecond paragraph that is too long."

    truncated = _truncate_content(content, max_length=25, language="en")

    assert truncated.startswith("First paragraph.")
    assert "content truncated" in truncated
