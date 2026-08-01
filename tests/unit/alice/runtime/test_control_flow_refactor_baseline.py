from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.agent_runtime.mtp.mtp_executor import KoakumaMTPExecutor
from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler
from hivememory.alice.runtime.agent.runtime import AgentRuntime
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope


def _frame(run_id: str, frame_id: str) -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(run_id=run_id, frame_id=frame_id),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic",
        identity=Identity(user_id="user"),
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


def test_agent_runtime_has_no_child_specific_public_api() -> None:
    public_methods = {
        name
        for name, member in inspect.getmembers(AgentRuntime, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    assert not any("child" in name or "sub_frame" in name for name in public_methods)


def test_frame_scheduler_resume_isolated_between_interleaved_runs() -> None:
    """The compatibility scheduler no longer owns a process-wide stack."""
    scheduler = FrameScheduler(prompt_assembler=SimpleNamespace())
    frame_a = _frame("run-a", "frame-a")
    frame_b = _frame("run-b", "frame-b")

    scheduler.suspend_frame(frame_a)
    assert scheduler.resume_frame() is frame_a
    scheduler.suspend_frame(frame_b)
    assert scheduler.resume_frame() is frame_b
    assert scheduler.get_current_depth() == 0


def test_mtp_cancel_event_is_invocation_local() -> None:
    koakuma = SimpleNamespace()
    executor = KoakumaMTPExecutor(koakuma)

    assert not hasattr(executor, "set_cancel_event")
    assert not hasattr(koakuma, "cancel_event")
