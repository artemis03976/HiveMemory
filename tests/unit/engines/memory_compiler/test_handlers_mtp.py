import pytest

from hivememory.engines.memory_compiler.handlers.mtp import compile_mtp_read
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


def test_compile_mtp_read_renders_redirect():
    unit = MemoryUnitIR(
        identity=MemoryIdentityIR(
            source_kind="resolve_result",
            alias="fact_api",
            redirected_from="draft_old",
        ),
        content=MemoryContentIR(content="Canonical content"),
        status=MemoryStatusIR(is_redirect=True),
    )

    artifact = compile_mtp_read(
        unit,
        MemoryCompileTarget.MTP_READ,
        MemoryCompileOptions(language="en"),
    )

    assert artifact.status == "redirect"
    assert "Alias Redirected" in artifact.text
    assert "draft_old" in artifact.text
    assert "fact_api" in artifact.text
    assert "Canonical content" in artifact.text


def test_compile_mtp_read_renders_pending_update():
    unit = MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="pending", alias="rev_1"),
        content=MemoryContentIR(instruction="Revise title", content="Updated content"),
        status=MemoryStatusIR(source_state="pending", source_verb="UPDATE"),
        metadata={"base_alias": "fact_api"},
    )

    artifact = compile_mtp_read(
        unit,
        MemoryCompileTarget.MTP_READ,
        MemoryCompileOptions(language="en"),
    )

    assert artifact.source_kind == "pending"
    assert "rev_1" in artifact.text
    assert "fact_api" in artifact.text
    assert "Revise title" in artifact.text
    assert "Updated content" in artifact.text


def test_compile_mtp_read_renders_failed_terminal_result():
    unit = MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="resolve_result", alias="draft_failed"),
        content=MemoryContentIR(),
        status=MemoryStatusIR(is_terminal=True, error="generation failed"),
    )

    artifact = compile_mtp_read(
        unit,
        MemoryCompileTarget.MTP_READ,
        MemoryCompileOptions(language="en"),
    )

    assert artifact.status == "failed"
    assert "generation failed" in artifact.text
    assert "draft_failed" in artifact.text


def test_compile_mtp_read_rejects_unknown_source_kind():
    unit = MemoryUnitIR(
        identity=MemoryIdentityIR(source_kind="pending", alias="draft_1"),
        content=MemoryContentIR(),
        status=MemoryStatusIR(source_state="pending", source_verb=None),
    )
    unit.identity.source_kind = "unknown"

    with pytest.raises(ValueError, match="Unsupported source"):
        compile_mtp_read(unit, MemoryCompileTarget.MTP_READ, MemoryCompileOptions(language="en"))
