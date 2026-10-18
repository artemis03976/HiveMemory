"""Frame 动词集合驱动真实提示词与 MTP/Workspace 操作链路的集成测试。"""

from dataclasses import replace

import pytest

from hivememory.agent_runtime.frame import FrameExecutionPolicy
from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.core.models import AgentProfile
from hivememory.prompts.mtp import MTPPromptBuilder
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_identity_scope, make_runtime_scope


@pytest.mark.asyncio
async def test_prompt_and_runtime_reject_verbs_outside_frame_without_submission() -> None:
    """模型强制输出未列出的动词时，frame 在真实操作入口之前拒绝并不留副作用。"""
    command = '⟪ WRITE | * | content="越权写入" ⟫'
    harness = OperationsHarness()
    submit = await harness.submitter(make_identity_scope())
    requests = []

    async def record_submission(request):
        """观察公开提交端口，意外请求仍经过真实线程 context 与授权。"""
        requests.append(request)
        return await submit(request)

    policy = FrameExecutionPolicy.from_verbs({"READ", "SEARCH", "RUN"})
    profile = AgentProfile(language="en")
    prompt = MTPPromptBuilder(
        language=profile.language, allowed_verbs=policy.permitted_verbs
    ).build()
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        execution_policy=policy,
        submit_operation=record_submission,
        language=profile.language,
    )

    result = await KoakumaRuntime().execute_mtp(command, context)

    assert "- READ:" in prompt
    assert "- SEARCH:" in prompt
    assert "- RUN:" in prompt
    assert "- WRITE:" not in prompt
    assert "- UPDATE:" not in prompt
    assert "- CALL:" not in prompt
    assert result.response_status == "error"
    assert 'code="mtp.permission.denied"' in result.formatted_response
    assert result.response_content == ""
    assert requests == []
    assert harness.registry.size == 0
    assert harness.runtime.stats()["cold_reads"] == 0


@pytest.mark.asyncio
async def test_shared_runtime_takes_verb_availability_from_frame_policy() -> None:
    """同一个 runtime 不持有权限状态，只有换 frame 策略才改变动词可用性。"""
    harness = OperationsHarness()
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        execution_policy=FrameExecutionPolicy.from_verbs({"READ", "RUN"}),
        submit_operation=await harness.submitter(make_identity_scope()),
        language="zh",
    )
    runtime = KoakumaRuntime()
    command = '⟪ WRITE | * | content="另一个 frame 的意图" ⟫'

    denied = await runtime.execute_mtp(command, context)
    granted = await runtime.execute_mtp(
        command,
        replace(
            context, execution_policy=FrameExecutionPolicy.from_verbs({"READ", "RUN", "WRITE"})
        ),
    )

    assert denied.response_status == "error"
    assert 'code="mtp.permission.denied"' in denied.formatted_response
    assert granted.response_status == "ack"
    assert harness.registry.size == 1
    pending = harness.registry.get(granted.pending_alias, make_identity_scope().workspace_identity)
    assert pending.focus.content == "另一个 frame 的意图"


@pytest.mark.asyncio
async def test_system_tool_alias_requires_exact_registration_without_resource_submission() -> None:
    """系统工具不设 Agent 权限，但未注册的 sys_* 前缀不会当作系统工具或记忆执行。"""
    harness = OperationsHarness()
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        execution_policy=FrameExecutionPolicy.from_verbs({"RUN"}),
        submit_operation=await harness.submitter(make_identity_scope(), allowed_operations=[]),
        language="zh",
    )

    result = await KoakumaRuntime().execute_mtp("⟪ RUN | sys_clock_evil | ⟫", context)

    assert result.response_status == "error"
    assert 'code="mtp.alias.not_found"' in result.formatted_response
    assert harness.runtime.stats()["cold_reads"] == 0
    assert harness.citations == []
