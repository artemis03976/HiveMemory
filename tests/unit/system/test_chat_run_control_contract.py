"""Chat Run 取消控制契约测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from hivememory.core.errors import WorkspaceDomainError
from hivememory.system.application.chat_service import _run_interruptible
from hivememory.system.runtime.control import (
    ChatGenerationRun,
    ChatGenerationRunRegistry,
    ChatRunOutcome,
    ChatRunPhase,
)
from tests.helpers.workspace import make_access_context


def _run(generation_id: str) -> ChatGenerationRun:
    return ChatGenerationRun(
        generation_id=generation_id,
        access_context=make_access_context(interaction_id=f"interaction-{generation_id}"),
    )


@pytest.mark.asyncio
async def test_gateway_stop_cancels_bound_task_immediately() -> None:
    run = _run("generation-1")
    blocker = asyncio.Event()
    task = asyncio.create_task(blocker.wait())
    run.bind_phase(ChatRunPhase.GATEWAY, task)

    result = run.request_stop()

    assert result.accepted is True
    assert result.reason == "user_requested"
    assert run.outcome is ChatRunOutcome.STOP_REQUESTED
    assert run.phase is ChatRunPhase.GATEWAY
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_prepare_stop_records_request_without_cancelling_prepare_task() -> None:
    run = _run("generation-2")
    run.enter_phase(ChatRunPhase.PREPARE)
    blocker = asyncio.Event()
    prepare_task = asyncio.create_task(blocker.wait())

    result = run.request_stop("during_prepare")

    assert result.accepted is True
    assert run.outcome is ChatRunOutcome.STOP_REQUESTED
    assert run.stop_reason == "during_prepare"
    assert prepare_task.cancelled() is False
    prepare_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare_task


def test_finalize_and_terminal_stop_are_rejected() -> None:
    finalizing = _run("generation-3")
    finalizing.enter_phase(ChatRunPhase.FINALIZE)
    result = finalizing.request_stop()
    assert result.accepted is False
    assert result.reason == "already_finalizing"
    assert finalizing.outcome is ChatRunOutcome.RUNNING

    terminal = _run("generation-4")
    terminal.phase = ChatRunPhase.TERMINAL
    terminal.outcome = ChatRunOutcome.COMPLETED
    result = terminal.request_stop()
    assert result.accepted is False
    assert result.reason == "already_terminal"
    assert terminal.outcome is ChatRunOutcome.COMPLETED


def test_repeated_stop_keeps_first_reason_and_does_not_cancel_again() -> None:
    run = _run("generation-5")
    task = MagicMock()
    task.done.return_value = False
    run.bind_phase(ChatRunPhase.ALICE, task)

    first = run.request_stop("first_reason")
    second = run.request_stop("second_reason")

    assert first.accepted is True
    assert second.accepted is True
    assert second.reason == "first_reason"
    task.cancel.assert_called_once_with()


def test_registry_not_found_and_terminal_results_are_stable() -> None:
    registry = ChatGenerationRunRegistry()

    access_context = make_access_context()
    missing = registry.cancel("missing-generation", access_context)
    assert missing.cancelled is False
    assert missing.status == "not_found"

    run = _run("generation-6")
    run.phase = ChatRunPhase.TERMINAL
    run.outcome = ChatRunOutcome.FAILED
    registry.register(run)

    terminal = registry.cancel(run.generation_id, run.access_context)
    assert terminal.cancelled is False
    assert terminal.reason == "already_terminal"
    assert run.outcome is ChatRunOutcome.FAILED


def test_stop_after_bound_task_finished_is_accepted_without_second_cancel() -> None:
    run = _run("generation-7")
    task = MagicMock()
    task.done.return_value = True
    run.bind_phase(ChatRunPhase.GATEWAY, task)

    result = run.request_stop("late_stop")

    assert result.accepted is True
    assert result.reason == "late_stop"
    task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_owner_task_cancellation_is_not_translated_to_chat_run_cancelled() -> None:
    run = _run("generation-8")
    blocker = asyncio.Event()

    async def operation():
        await blocker.wait()

    task = asyncio.create_task(_run_interruptible(run, ChatRunPhase.GATEWAY, operation))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.outcome is ChatRunOutcome.RUNNING


def test_registry_hides_run_from_different_workspace_control_plane() -> None:
    """防止仅凭 generation_id 跨 Workspace 查询或取消另一条 run。"""
    registry = ChatGenerationRunRegistry()
    owner_context = make_access_context(
        user_id="u1",
        workspace_id="main_workspace",
        interaction_id="interaction-main",
    )
    other_context = make_access_context(
        user_id="u1",
        workspace_id="isolation_workspace",
        interaction_id="interaction-isolation",
    )
    run = ChatGenerationRun(
        generation_id="shared-generation-id",
        access_context=owner_context,
    )
    registry.register(run)

    assert registry.get(run.generation_id, other_context) is None
    assert registry.status(run.generation_id, other_context) is None
    rejected = registry.cancel(run.generation_id, other_context)
    assert rejected.status == "not_found"
    assert rejected.cancelled is False
    assert run.outcome is ChatRunOutcome.RUNNING


def test_registry_rejects_generation_id_collision_without_overwriting_owner() -> None:
    """防止重复 generation_id 覆盖既有 scope 并把控制权转给后注册者。"""
    registry = ChatGenerationRunRegistry()
    original = _run("collision")
    replacement = ChatGenerationRun(
        generation_id="collision",
        access_context=make_access_context(
            workspace_id="isolation_workspace",
            interaction_id="replacement",
        ),
    )
    registry.register(original)

    with pytest.raises(WorkspaceDomainError, match="拒绝覆盖"):
        registry.register(replacement)

    assert registry.get("collision", original.access_context) is original
    assert registry.get("collision", replacement.access_context) is None
