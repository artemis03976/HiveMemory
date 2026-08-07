"""Reproducible copy/projection baseline for data-model governance Phase I.

The benchmark is intentionally self-contained and does not require Qdrant, an
LLM provider, or a running HiveMemory service.  It measures current boundary
projection costs and the corresponding full-deep-copy costs for the three
model shapes called out by the Phase I plan: a large Topic, a Memory list, and
a long TurnEvent stream.
"""

from __future__ import annotations

import gc
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hivememory.core.models.artifact import (  # noqa: E402
    ArtifactRef,
    ArtifactType,
    MemoryEventLog,
    MemoryEventType,
)
from hivememory.core.models.interaction import (  # noqa: E402
    Identity,
    TraceItem,
    TurnEvent,
    TurnRecord,
)
from hivememory.core.models.memory import (  # noqa: E402
    Artifacts,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    RelationLayer,
)
from hivememory.core.models.pending import (  # noqa: E402
    PendingAtomMaterializeTask,
    WriteFocus,
)
from hivememory.core.models.topic import LogicalBlock, TopicData  # noqa: E402
from hivememory.core.protocol.models import InteractionPayload  # noqa: E402
from hivememory.patchouli.memory_library.buffer import SemanticBuffer  # noqa: E402
from hivememory.server.models.memory import MemoryResponse  # noqa: E402

TOPIC_BLOCKS = 1_000
MEMORY_ITEMS = 1_000
TURN_EVENTS = 10_000
QUEUE_EVENTS = 2_000
REPEATS = 7


def _time_operation(operation: Callable[[], Any], repeats: int = REPEATS) -> float:
    samples_ms: list[float] = []
    operation()
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter_ns()
        result = operation()
        elapsed = time.perf_counter_ns() - started
        if result is None:
            raise RuntimeError("benchmark operation unexpectedly returned None")
        samples_ms.append(elapsed / 1_000_000)
    return statistics.median(samples_ms)


def _peak_mib(operation: Callable[[], Any]) -> float:
    gc.collect()
    tracemalloc.start()
    result = operation()
    if result is None:
        raise RuntimeError("benchmark operation unexpectedly returned None")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def _build_turn(index: int, *, event_count: int = 2) -> TurnRecord:
    identity = Identity(user_id="benchmark-user", agent_id="benchmark-agent")
    events = tuple(
        TurnEvent(
            kind="tool_call" if offset % 2 == 0 else "tool_result",
            sequence=offset,
            role="assistant" if offset % 2 == 0 else "system",
            content=f"event-{index}-{offset}",
            action_id=f"action-{index}-{offset // 2}",
            tool_name="benchmark_tool",
            tool_kind="RUN",
            tool_args={"index": index, "nested": {"offsets": [offset, offset + 1]}},
            status="completed",
        )
        for offset in range(event_count)
    )
    return TurnRecord(
        turn_id=f"turn-{index}",
        identity=identity,
        user_query=f"benchmark question {index}",
        assistant_final_text=f"benchmark answer {index}",
        turn_events=events,
        semantic_traces=(
            TraceItem(
                action="RUN",
                action_id=f"action-{index}-0",
                tool="benchmark_tool",
                status="completed",
            ),
        ),
    )


def _build_topic_buffer() -> SemanticBuffer:
    return SemanticBuffer(
        topic_id="benchmark-topic",
        user_id="benchmark-user",
        topic_title="Phase I benchmark topic",
        topic_summary="Synthetic large topic used for boundary projection measurements.",
        blocks=[
            LogicalBlock(turn=_build_turn(index), total_tokens=64)
            for index in range(TOPIC_BLOCKS)
        ],
        total_tokens=TOPIC_BLOCKS * 64,
    )


def _project_topic(buffer: SemanticBuffer) -> TopicData:
    """Mirror ShortTermMemoryStore._to_topic_data without requiring a port."""

    return TopicData(
        topic_id=buffer.topic_id,
        user_id=buffer.user_id,
        current_agent_id=buffer.current_agent_id,
        topic_title=buffer.topic_title,
        topic_summary=buffer.topic_summary,
        state_summary=buffer.state_summary,
        blocks=tuple(buffer.blocks),
        state=buffer.state,
        last_update=buffer.last_update,
        last_accessed_at=buffer.last_accessed_at,
        total_tokens=buffer.total_tokens,
        model_used=buffer.model_used,
    )


def _build_memory(index: int) -> MemoryAtom:
    ref = ArtifactRef(
        artifact_id=f"artifact-{index}",
        artifact_type=ArtifactType.MEMORY_VERSION,
        uri=f"memory://benchmark/{index}",
        sha256="0" * 64,
    )
    event = MemoryEventLog(
        event_type=MemoryEventType.VERSIONED,
        artifact_refs=[ref],
        note="benchmark",
    )
    return MemoryAtom(
        meta=MetaData(
            source_agent_id="benchmark-agent",
            user_id="benchmark-user",
        ),
        index=IndexLayer(
            title=f"Benchmark memory {index}",
            summary="Synthetic memory used for Phase I projection benchmarking.",
            tags=["benchmark", "phase-i", f"item-{index % 10}"],
            memory_type=MemoryType.FACT,
            alias=f"benchmark_memory_{index}",
        ),
        payload=PayloadLayer(
            content=("benchmark content " * 32) + str(index),
            history_summary=["created", "updated"],
            artifacts=Artifacts(
                refs=[ref],
                events=[event],
                revival_keys=[f"key-{index}"],
            ),
        ),
        relations=RelationLayer(relates_to=[f"memory-{max(0, index - 1)}"]),
    )


def _build_long_event_stream(size: int) -> tuple[TurnEvent, ...]:
    return tuple(
        TurnEvent(
            kind="thought",
            sequence=index,
            role="assistant",
            content=f"event stream item {index}",
            action_id=f"action-{index // 2}",
            tool_args={"index": index, "values": [index, index + 1]},
        )
        for index in range(size)
    )


def _build_queue_payload(events: tuple[TurnEvent, ...]) -> InteractionPayload:
    identity = Identity(user_id="benchmark-user", agent_id="benchmark-agent")
    materialize_task = PendingAtomMaterializeTask(
        pending_alias="draft_benchmark",
        intent_id="intent_benchmark",
        source_verb="WRITE",
        identity=identity,
        focus=WriteFocus(content="benchmark memory intent"),
    )
    return InteractionPayload(
        identity=identity,
        user_message="benchmark interaction",
        assistant_final_text="benchmark response",
        turn_events=list(events),
        mtp_traces=[TraceItem(action="WRITE", action_id="action-0")],
        materialize_tasks=[materialize_task],
        worth_saving=True,
    )


def main() -> None:
    topic_buffer = _build_topic_buffer()
    topic_snapshot = _project_topic(topic_buffer)
    memories = [_build_memory(index) for index in range(MEMORY_ITEMS)]
    long_events = _build_long_event_stream(TURN_EVENTS)
    queue_payload = _build_queue_payload(long_events[:QUEUE_EVENTS])
    queue_json = queue_payload.model_dump_json()

    operations: list[tuple[str, str, Callable[[], Any]]] = [
        (
            "Large Topic",
            f"current TopicData projection ({TOPIC_BLOCKS} blocks)",
            lambda: _project_topic(topic_buffer),
        ),
        (
            "Large Topic",
            f"full deep copy ({TOPIC_BLOCKS} blocks)",
            lambda: topic_snapshot.model_copy(deep=True),
        ),
        (
            "Memory list",
            f"HTTP projection ({MEMORY_ITEMS} atoms)",
            lambda: [MemoryResponse.from_atom(memory) for memory in memories],
        ),
        (
            "Memory list",
            f"full deep copy ({MEMORY_ITEMS} atoms)",
            lambda: [memory.model_copy(deep=True) for memory in memories],
        ),
        (
            "TurnEvent stream",
            f"TurnRecord projection ({TURN_EVENTS} events)",
            lambda: TurnRecord(
                turn_id="long-event-stream",
                user_query="benchmark",
                assistant_final_text="benchmark",
                turn_events=long_events,
            ),
        ),
        (
            "TurnEvent stream",
            f"full deep copy ({TURN_EVENTS} events)",
            lambda: TurnRecord(
                turn_id="long-event-stream",
                user_query="benchmark",
                assistant_final_text="benchmark",
                turn_events=long_events,
            ).model_copy(deep=True),
        ),
        (
            "Queue candidate",
            f"InteractionPayload JSON encode ({QUEUE_EVENTS} events)",
            queue_payload.model_dump_json,
        ),
        (
            "Queue candidate",
            f"InteractionPayload JSON decode ({QUEUE_EVENTS} events)",
            lambda: InteractionPayload.model_validate_json(queue_json),
        ),
    ]

    print("# Data model Phase I performance baseline")
    print()
    print(f"- Python: {platform.python_version()}")
    print(f"- Pydantic: {version('pydantic')}")
    print(f"- Platform: {platform.platform()}")
    print(f"- Repeats: {REPEATS} (median reported)")
    print()
    print("| Shape | Operation | Median ms | Peak MiB |")
    print("|:--|:--|--:|--:|")
    for shape, description, operation in operations:
        median_ms = _time_operation(operation)
        peak_mib = _peak_mib(operation)
        print(f"| {shape} | {description} | {median_ms:.3f} | {peak_mib:.3f} |")


if __name__ == "__main__":
    main()
