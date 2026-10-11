"""MTP 公开执行入口按 frame 策略拒绝动词，并忽略旧版 Profile 权限。"""

from pathlib import Path

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.config.alice import KoakumaConfig
from hivememory.core.models import AgentProfile
from hivememory.core.mtp import MTPCallRequest
from tests.helpers.workspace import make_runtime_scope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "⟪ READ | memory | ⟫",
        '⟪ SEARCH | * | query="记忆" ⟫',
        '⟪ WRITE | * | content="拒绝写入" ⟫',
        '⟪ UPDATE | memory | instruction="修改" ⟫',
        '⟪ RUN | sys_python_repl | code="print(42)" ⟫',
        '⟪ CALL | another_agent | task="任务" ⟫',
    ],
)
async def test_empty_frame_policy_rejects_verbs_before_submitting(command: str) -> None:
    """frame 拒绝先于任何资源提交，Profile 无法扩大明确的空集合。"""
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


@pytest.mark.asyncio
async def test_frame_policy_allows_call_despite_legacy_profile_denial() -> None:
    """旧配置中的空动词列表不影响 frame 明确允许的 CALL。"""
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        agent_profile=AgentProfile.model_validate({"allowed_mtp_verbs": []}),
        execution_policy=FrameExecutionPolicy.from_verbs({"call"}),
    )

    result = await KoakumaRuntime().execute_mtp(
        '⟪ CALL | reviewer_doll | task="审查代码" ⟫', context
    )

    assert result.response_status == "suspend"
    assert result.call_request == MTPCallRequest(target_alias="reviewer_doll", task="审查代码")


@pytest.mark.asyncio
async def test_system_write_ignores_legacy_tool_denial(tmp_path: Path) -> None:
    """所有已注册系统工具都可执行，旧工具白名单不能阻止本地文件写入。"""
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        agent_profile=AgentProfile.model_validate({"allowed_sys_tools": []}),
        execution_policy=FrameExecutionPolicy.from_verbs({"RUN"}),
    )
    runtime = KoakumaRuntime(config=KoakumaConfig(workspace_path=str(tmp_path)))

    result = await runtime.execute_mtp(
        '⟪ RUN | sys_write_file | path="result.txt" content="系统工具无需权限" ⟫', context
    )

    assert result.response_status == "success"
    assert (tmp_path / "result.txt").read_text() == "系统工具无需权限"
