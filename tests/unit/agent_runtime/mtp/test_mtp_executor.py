from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.mtp_executor import KoakumaMTPExecutor
from hivememory.core.models import OMNI_DOLL_PROFILE, Identity, RuntimeScope
from hivememory.core.protocol.models import MTPExecutionResult


@pytest.mark.asyncio
async def test_koakuma_mtp_executor_delegates_to_runtime():
    result = MTPExecutionResult(
        command=None,
        response_status="success",
        response_content="ok",
        formatted_response="<mtp_response>ok</mtp_response>",
        success=True,
        execution_time_ms=1.0,
    )
    koakuma = MagicMock()
    koakuma.intercept_and_execute = AsyncMock(return_value=result)
    context = MTPExecutionContext(
        identity=Identity(user_id="u1", agent_id="agent_a"),
        agent_profile=OMNI_DOLL_PROFILE,
        runtime_scope=RuntimeScope(run_id="run-1", frame_id="frame-1"),
    )

    executor = KoakumaMTPExecutor(koakuma)
    actual = await executor.intercept_and_execute("assistant text", context)

    koakuma.intercept_and_execute.assert_awaited_once_with(
        "assistant text",
        context=context,
        cancel_event=None,
    )
    assert actual is result
