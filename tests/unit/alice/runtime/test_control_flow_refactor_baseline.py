from __future__ import annotations

import ast
from pathlib import Path

from hivememory.agent_runtime.models import ExecutionFrame, MTPExecutionContext
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.alice.orchestration.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.core.models import OMNI_DOLL_PROFILE
from tests.helpers.workspace import make_execution_labels, make_runtime_scope


def _frame(run_id: str, frame_id: str) -> ExecutionFrame:
    return FrameFactory().create(
        FrameSpec(
            runtime_scope=FrameFactory.scope(
                labels=make_execution_labels(),
                run_id=run_id,
                frame_id=frame_id,
            ),
            profile=OMNI_DOLL_PROFILE,
            messages=[],
            topic_id="topic",
            execution_policy=FrameExecutionPolicy(),
        )
    )


def test_agent_runtime_package_does_not_import_alice() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    package_root = repo_root / "src" / "hivememory" / "agent_runtime"

    violations: list[str] = []
    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(
                name == "hivememory.alice" or name.startswith("hivememory.alice.") for name in names
            ):
                violations.append(str(source_path.relative_to(repo_root)))

    assert violations == []


def test_run_session_keeps_frames_and_calls_run_local() -> None:
    session = RunSession(agent_run_id="run-a")
    frame_a = _frame("run-a", "frame-a")
    frame_b = _frame("run-a", "frame-b")

    session.register_frame(frame_a)
    session.register_frame(frame_b)
    record = session.register_call(frame_a, "action-1")

    assert set(session.frames) == {"frame-a", "frame-b"}
    assert session.call_records[("frame-a", "action-1")] is record


def test_mtp_context_contains_workspace_and_frame_coordinates() -> None:
    context = MTPExecutionContext(runtime_scope=make_runtime_scope())

    assert context.runtime_scope.run_id == "test_run"
    assert context.runtime_scope.frame_id == "test_frame"
    assert context.runtime_scope.labels.workspace_id == "main_workspace"
