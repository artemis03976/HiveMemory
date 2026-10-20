"""MTP READ 的输出顺序与部分缺失提示。"""

from datetime import UTC, datetime

import pytest

from hivememory.agent_runtime.frame import FrameExecutionPolicy
from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.core.models.reference import ReferenceResolution
from hivememory.workspace.contracts import ResolveReferencesRequest
from tests.helpers.call import StubOperationSubmitter
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_runtime_scope


def _atom(title: str, content: str) -> MemoryAtom:
    return MemoryAtom(
        index=IndexLayer(
            title=title, summary=f"{title} summary", memory_type=MemoryType.FACT, tags=["read"]
        ),
        payload=PayloadLayer(content=content),
        meta=make_memory_metadata(
            source_agent_id="writer",
            user_id="user-1",
            updated_at=datetime.now(UTC),
            confidence_score=0.9,
        ),
    )


def _context(results: list[ReferenceResolution]) -> MTPExecutionContext:
    async def operation(request):
        """只替代 workspace 的引用解析，READ 的解析与编译保持真实。"""
        assert isinstance(request, ResolveReferencesRequest)
        return results

    return MTPExecutionContext(
        submit_operation=StubOperationSubmitter(operation=operation),
        runtime_scope=make_runtime_scope(run_id="run-1", frame_id="frame-1"),
        execution_policy=FrameExecutionPolicy.from_verbs({"READ"}),
        language="en",
    )


@pytest.mark.asyncio
async def test_read_outputs_each_alias_in_request_order_and_warns_for_missing_ones() -> None:
    results = [
        ReferenceResolution(kind="atom", atom=_atom("Second fact", "content of fact_b")),
        ReferenceResolution(kind="failed", requested_alias="draft_failed"),
        ReferenceResolution(kind="not_found", requested_alias="missing"),
        ReferenceResolution(kind="atom", atom=_atom("First fact", "content of fact_a")),
    ]

    result = await KoakumaRuntime().execute_mtp(
        "⟪ READ | [fact_b, draft_failed, missing, fact_a] ⟫", _context(results)
    )

    assert result.response_status == "success"
    content = result.response_content
    positions = [
        content.index("content of fact_b"),
        content.index("draft_failed"),
        content.index("content of fact_a"),
    ]
    assert positions == sorted(positions)
    assert "missing" in result.formatted_response
