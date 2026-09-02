from unittest.mock import MagicMock

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.core.mtp import MTPCallRequest, MTPCommand, MTPResponseStatus, MTPVerb
from hivememory.core.mtp.exceptions import PermissionDeniedError
from tests.helpers.workspace import make_runtime_scope


def _context(*, allow_call: bool) -> MTPExecutionContext:
    policy = FrameExecutionPolicy.from_profile(
        OMNI_DOLL_PROFILE,
        denied_verbs=() if allow_call else {"CALL"},
    )
    return MTPExecutionContext(
        agent_profile=OMNI_DOLL_PROFILE,
        runtime_scope=make_runtime_scope(run_id="run-1", frame_id="frame-1"),
        execution_policy=policy,
    )


def _koakuma() -> MagicMock:
    koakuma = MagicMock(spec=KoakumaRuntime)
    koakuma._handle_call = KoakumaRuntime._handle_call.__get__(koakuma)
    return koakuma


@pytest.mark.asyncio
async def test_call_returns_suspend() -> None:
    koakuma = _koakuma()
    context = _context(allow_call=True)
    command = MagicMock(spec=MTPCommand)
    command.target = MagicMock(single_alias="coder_doll")
    command.args = {"task": "Write code", "context_refs": '["mem_spec"]'}

    response = await koakuma._handle_call(command, context=context)

    assert response.status == MTPResponseStatus.SUSPEND
    assert response.call_request == MTPCallRequest(
        target_alias="coder_doll",
        task="Write code",
        context_refs=["mem_spec"],
    )


@pytest.mark.asyncio
async def test_policy_blocks_call_without_depth_semantics() -> None:
    koakuma = _koakuma()
    command = MagicMock(spec=MTPCommand)
    command.target = MagicMock(single_alias="another_doll")
    command.args = {"task": "Forbidden task"}

    with pytest.raises(PermissionDeniedError):
        await koakuma._handle_call(command, context=_context(allow_call=False))


@pytest.mark.asyncio
async def test_call_missing_task() -> None:
    koakuma = _koakuma()
    command = MagicMock(spec=MTPCommand)
    command.verb = MTPVerb.CALL
    command.target = MagicMock(single_alias="coder_doll")
    command.args = {}
    response = await KoakumaRuntime._route_and_execute(
        koakuma,
        command,
        _context(allow_call=True),
    )

    assert response.status == MTPResponseStatus.ERROR
    assert response.error.code == "mtp.argument.invalid"


@pytest.mark.asyncio
async def test_call_missing_target() -> None:
    koakuma = _koakuma()
    command = MagicMock(spec=MTPCommand)
    command.verb = MTPVerb.CALL
    command.target = MagicMock(single_alias=None)
    command.args = {"task": "some task"}
    response = await KoakumaRuntime._route_and_execute(
        koakuma,
        command,
        _context(allow_call=True),
    )

    assert response.status == MTPResponseStatus.ERROR
    assert response.error.code == "mtp.argument.invalid"
