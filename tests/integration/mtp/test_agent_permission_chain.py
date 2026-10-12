"""Frame 动词集合驱动真实提示词与 MTP/Workspace 操作链路的集成测试。"""

from dataclasses import replace

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.policy import FrameExecutionPolicy
from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import AgentProfile, IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.prompts.mtp import MTPPromptBuilder
from tests.helpers.memory import make_memory_metadata
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_identity_scope, make_runtime_scope


def _legacy_profile(*, allowed_verbs: list[str], allowed_tools: list[str]) -> AgentProfile:
    """从存量图纸的旧权限键解析，模型参数与人设仍由 Profile 提供。"""
    atom = MemoryAtom(
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            alias="legacy_agent", title="旧版人偶", memory_type=MemoryType.AGENT_PROFILE
        ),
        payload=PayloadLayer(
            content="专注代码审查",
            agent_config={
                "model_name": "test-model",
                "language": "en",
                "allowed_mtp_verbs": allowed_verbs,
                "allowed_sys_tools": allowed_tools,
            },
        ),
    )
    profile = AgentProfile.from_atom(atom)
    assert isinstance(profile, AgentProfile)
    return profile


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        '⟪ WRITE | * | content="越权写入" ⟫',
        '⟪ UPDATE | fact_missing | instruction="越权更新" ⟫',
        '⟪ CALL | another_agent | task="越权调用" ⟫',
    ],
)
async def test_prompt_and_runtime_reject_verbs_outside_frame_without_submission(
    command: str,
) -> None:
    """模型强制输出未列出的动词时，frame 在真实操作入口之前拒绝并不留副作用。"""
    harness = OperationsHarness()
    submit = await harness.submitter(make_identity_scope())
    requests = []

    async def record_submission(request):
        """观察公开提交端口，意外请求仍经过真实线程 context 与授权。"""
        requests.append(request)
        return await submit(request)

    policy = FrameExecutionPolicy.from_verbs({"READ", "SEARCH", "RUN"})
    profile = _legacy_profile(
        allowed_verbs=["READ", "SEARCH", "RUN", "WRITE", "UPDATE", "CALL"],
        allowed_tools=["sys_clock"],
    )
    prompt = MTPPromptBuilder(
        language=profile.language, allowed_verbs=policy.permitted_verbs
    ).build()
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        agent_profile=profile,
        execution_policy=policy,
        submit_operation=record_submission,
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
@pytest.mark.parametrize("legacy_verbs", [[], ["READ", "SEARCH"]])
async def test_legacy_profile_denial_cannot_restrict_authorized_write_or_tool_menu(
    legacy_verbs: list[str],
) -> None:
    """Profile 旧键不再裁剪 frame 教学或执行；真实 WRITE 按访问登记产生 ACK。"""
    harness = OperationsHarness()
    profile = _legacy_profile(allowed_verbs=legacy_verbs, allowed_tools=[])
    policy = FrameExecutionPolicy.from_verbs({"READ", "RUN", "WRITE"})
    prompt = MTPPromptBuilder(
        language=profile.language, allowed_verbs=policy.permitted_verbs
    ).build()
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        agent_profile=profile,
        execution_policy=policy,
        submit_operation=await harness.submitter(
            make_identity_scope(), allowed_operations={WorkspaceOperation.MEMORY_WRITE}
        ),
    )

    result = await KoakumaRuntime().execute_mtp('⟪ WRITE | * | content="由访问登记授权" ⟫', context)

    assert "- WRITE:" in prompt
    assert "- UPDATE:" not in prompt
    assert "- CALL:" not in prompt
    for tool in (
        "sys_clock",
        "sys_web_search",
        "sys_read_file",
        "sys_write_file",
        "sys_python_repl",
    ):
        assert f"`{tool}`" in prompt
    assert result.response_status == "ack"
    assert harness.registry.size == 1
    pending = harness.registry.get(result.pending_alias, make_identity_scope().workspace_identity)
    assert pending.focus.content == "由访问登记授权"
    assert pending.from_actor == make_identity_scope().actor_identity


@pytest.mark.asyncio
async def test_shared_runtime_keeps_frame_policies_independent_of_profile() -> None:
    """同一个 runtime 与 Profile 不持有权限状态，换 frame 策略才能改变动词可用性。"""
    harness = OperationsHarness()
    profile = _legacy_profile(allowed_verbs=[], allowed_tools=[])
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        agent_profile=profile,
        execution_policy=FrameExecutionPolicy.from_verbs({"READ", "RUN"}),
        submit_operation=await harness.submitter(make_identity_scope()),
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
    """系统工具不设 Agent 权限，但未注册的 sys_* 前缀不会当作内核工具或记忆执行。"""
    harness = OperationsHarness()
    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        execution_policy=FrameExecutionPolicy.from_verbs({"RUN"}),
        submit_operation=await harness.submitter(make_identity_scope(), allowed_operations=[]),
    )

    result = await KoakumaRuntime().execute_mtp("⟪ RUN | sys_clock_evil | ⟫", context)

    assert result.response_status == "error"
    assert 'code="mtp.alias.not_found"' in result.formatted_response
    assert harness.runtime.stats()["cold_reads"] == 0
    assert harness.citations == []
