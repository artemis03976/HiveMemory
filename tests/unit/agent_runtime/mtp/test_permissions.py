"""MTP 公开执行入口按 frame 策略拒绝动词。"""

import pytest

from hivememory.agent_runtime.frame import FrameExecutionPolicy
from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.core.models import AgentProfile
from tests.helpers.workspace import make_runtime_scope


@pytest.mark.asyncio
async def test_empty_frame_policy_rejects_verbs_before_submitting() -> None:
    """全部动词在分派前经同一处 frame 检查，拒绝先于任何资源提交。"""
    command = '⟪ WRITE | * | content="拒绝写入" ⟫'
    requests = []

    async def record_submission(request):
        requests.append(request)
        raise AssertionError("被拒绝的动词不能进入操作提交端口")

    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        agent_profile=AgentProfile(),
        execution_policy=FrameExecutionPolicy.from_verbs([]),
        submit_operation=record_submission,
    )

    result = await KoakumaRuntime().execute_mtp(command, context)

    assert result.response_status == "error"
    assert 'code="mtp.permission.denied"' in result.formatted_response
    assert result.response_content == ""
    assert requests == []
