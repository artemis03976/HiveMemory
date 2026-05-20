from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.alice.runtime.models import MTPExecutionContext
from hivememory.alice.runtime.mtp_executor import KoakumaMTPExecutor
from hivememory.core.models import Identity, OMNI_DOLL_PROFILE
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
        depth=0,
    )

    executor = KoakumaMTPExecutor(koakuma)
    actual = await executor.intercept_and_execute("assistant text", context)

    koakuma.intercept_and_execute.assert_awaited_once_with(
        "assistant text",
        context=context,
    )
    assert actual is result
