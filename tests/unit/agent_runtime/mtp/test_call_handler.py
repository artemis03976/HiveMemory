"""MTP CALL 的公开执行结果、参数校验与 frame 策略单元测试。"""

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.core.mtp import MTPCallRequest
from tests.helpers.workspace import make_runtime_scope


def _context(*, allow_call: bool) -> MTPExecutionContext:
    return MTPExecutionContext(
        runtime_scope=make_runtime_scope(run_id="run-1", frame_id="frame-1"),
        execution_policy=FrameExecutionPolicy.from_verbs({"CALL"} if allow_call else set()),
    )


@pytest.mark.asyncio
async def test_call_returns_suspend_with_context_refs() -> None:
    result = await KoakumaRuntime().execute_mtp(
        '⟪ CALL | coder_doll | task="Write code" context_refs=["mem_spec"] ⟫',
        _context(allow_call=True),
    )

    assert result.response_status == "suspend"
    assert result.call_request == MTPCallRequest(
        target_alias="coder_doll", task="Write code", context_refs=["mem_spec"]
    )


@pytest.mark.asyncio
async def test_policy_blocks_call_without_depth_semantics() -> None:
    result = await KoakumaRuntime().execute_mtp(
        '⟪ CALL | another_doll | task="Forbidden task" ⟫', _context(allow_call=False)
    )

    assert result.response_status == "error"
    assert 'code="mtp.permission.denied"' in result.formatted_response
    assert result.call_request is None


@pytest.mark.asyncio
async def test_call_missing_task_returns_argument_error() -> None:
    result = await KoakumaRuntime().execute_mtp(
        "⟪ CALL | coder_doll | ⟫", _context(allow_call=True)
    )

    assert result.response_status == "error"
    assert 'code="mtp.argument.invalid"' in result.formatted_response


@pytest.mark.asyncio
async def test_call_wildcard_target_returns_argument_error() -> None:
    result = await KoakumaRuntime().execute_mtp(
        '⟪ CALL | * | task="some task" ⟫', _context(allow_call=True)
    )

    assert result.response_status == "error"
    assert 'code="mtp.argument.invalid"' in result.formatted_response
