from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

from hivememory.agent_runtime.models import ExecutionFrame, MTPExecutionContext
from hivememory.agent_runtime.mtp import KoakumaMTPExecutor
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.application import AgentRunService
from hivememory.alice.orchestration.frame_factory import FrameFactory, FrameSpec
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.runtime.core import AliceRuntime
from hivememory.alice.runtime.profile_resolver import AgentProfileResolver
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity
from hivememory.system.config import HiveMemoryConfig


def _frame(run_id: str, frame_id: str) -> ExecutionFrame:
    return FrameFactory().create(
        FrameSpec(
            runtime_scope=FrameFactory.scope(run_id=run_id, frame_id=frame_id),
            profile=OMNI_DOLL_PROFILE,
            identity=Identity(user_id="user"),
            messages=[],
            topic_id="topic",
            execution_policy=FrameExecutionPolicy.from_profile(OMNI_DOLL_PROFILE),
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


def test_agent_runtime_has_no_child_specific_public_api() -> None:
    public_methods = {
        name
        for name, member in inspect.getmembers(AgentRuntime, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    assert not any("child" in name or "sub_frame" in name for name in public_methods)


def test_alice_runtime_does_not_own_agent_run_use_case() -> None:
    runtime_public_methods = {
        name
        for name, member in inspect.getmembers(AliceRuntime, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    assert "run_agent" not in runtime_public_methods
    assert "run_agent_stream" not in runtime_public_methods
    assert inspect.iscoroutinefunction(AgentRunService.run_agent)
    assert inspect.isasyncgenfunction(AgentRunService.run_agent_stream)


def test_alice_runtime_owns_process_scoped_profile_resolver() -> None:
    config = HiveMemoryConfig()
    runtime = AliceRuntime(config.alice, config.memory_compiler)

    assert isinstance(runtime.profile_resolver, AgentProfileResolver)
    assert runtime.profile_resolver is runtime.profile_resolver


def test_frame_factory_creates_ordinary_frames_without_topology_metadata() -> None:
    frame = _frame("run-a", "frame-a")

    assert frame.runtime_scope.run_id == "run-a"
    assert frame.runtime_scope.frame_id == "frame-a"
    assert not hasattr(frame.runtime_scope, "depth")
    assert not hasattr(frame.runtime_scope, "parent_frame_id")
    assert not hasattr(frame, "is_main_frame")
    assert not hasattr(frame, "is_sub_frame")


def test_run_session_keeps_frames_and_calls_run_local() -> None:
    session = RunSession(agent_run_id="run-a")
    frame_a = _frame("run-a", "frame-a")
    frame_b = _frame("run-a", "frame-b")

    session.register_frame(frame_a)
    session.register_frame(frame_b)
    record = session.register_call(frame_a, "action-1")

    assert set(session.frames) == {"frame-a", "frame-b"}
    assert session.call_records[("frame-a", "action-1")] is record


def test_mtp_executor_keeps_runtime_stateless() -> None:
    koakuma = SimpleNamespace()
    executor = KoakumaMTPExecutor(koakuma)

    assert vars(executor) == {"_koakuma": koakuma}
    assert vars(koakuma) == {}


def test_mtp_context_contains_only_frame_coordinates() -> None:
    context = MTPExecutionContext()

    assert context.runtime_scope.run_id == ""
    assert context.runtime_scope.frame_id == ""
    assert not hasattr(context.runtime_scope, "depth")
