"""
Agent 权限端到端测试

测试覆盖:
- 端到端权限链路: profile → prompt 过滤 → Koakuma 拦截
- 限制性 profile 阻止越权操作
- 无 profile 时的全权限兜底
- Prompt 不显示禁止的动词和工具
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from hivememory.core.models import AgentProfile, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.patchouli.kernel.core import PatchouliKernel
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.patchouli.mtp.exceptions import PermissionDeniedError
from hivememory.patchouli.mtp import MTPCommand, MTPVerb


def _make_profile_atom(
    agent_id: str,
    allowed_verbs: list,
    allowed_tools: list,
) -> MemoryAtom:
    """构建 AGENT_PROFILE 类型的 MemoryAtom"""
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            user_id="system",
            source_agent_id="system",
            visibility="PUBLIC",
        ),
        index=IndexLayer(
            alias=agent_id,
            title=f"Agent {agent_id}",
            summary=f"Profile for {agent_id}",
            tags=["agent", "profile"],
            memory_type=MemoryType.AGENT_PROFILE,
        ),
        payload=PayloadLayer(
            content=f"You are {agent_id}.",
            artifacts={
                "agent_config": {
                    "model_name": "gpt-4",
                    "temperature": 0.7,
                    "allowed_mtp_verbs": allowed_verbs,
                    "allowed_sys_tools": allowed_tools,
                }
            },
        ),
    )


def _create_kernel_with_koakuma():
    """创建带 Koakuma 的 PatchouliKernel"""
    with patch.object(PatchouliKernel, "_init_infrastructure"), \
         patch.object(PatchouliKernel, "_build_engines", return_value={}), \
         patch.object(PatchouliKernel, "_register_services"), \
         patch.object(PatchouliKernel, "_register_bus_routes"):

        mock_config = Mock()
        mock_config.koakuma.enabled = True
        mock_config.koakuma.mtp_prompt.enabled = True
        mock_config.koakuma.mtp_prompt.language = "en"
        mock_config.koakuma.mtp_prompt.include_demo = True
        mock_config.koakuma.mtp_prompt.include_error_handling = True

        kernel = PatchouliKernel(config=mock_config, bus=None)
        kernel.storage = Mock()

        # 创建真实的 Koakuma 实例
        koakuma = KoakumaRuntime(bus=None, config=None)
        kernel._services = {"koakuma": koakuma}

        return kernel, koakuma


@pytest.mark.asyncio
class TestProfileToPromptFiltering:
    """Profile → Prompt 过滤链路测试"""

    async def test_profile_filters_prompt_verbs(self):
        """Profile 的动词白名单正确过滤 prompt"""
        kernel, _ = _create_kernel_with_koakuma()

        # 加载限制性 profile
        profile_atom = _make_profile_atom(
            agent_id="reviewer_doll",
            allowed_verbs=["READ", "SEARCH"],
            allowed_tools=["sys_clock"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=profile_atom)

        profile = kernel.load_agent_profile("reviewer_doll")
        assert profile is not None

        # 生成 MTP prompt
        mtp_prompt = kernel.get_mtp_prompt(profile=profile)

        # 允许的动词应该出现
        assert "- READ:" in mtp_prompt
        assert "- SEARCH:" in mtp_prompt

        # 禁止的动词不应该出现
        assert "- WRITE:" not in mtp_prompt
        assert "- UPDATE:" not in mtp_prompt
        assert "- RUN:" not in mtp_prompt

    async def test_profile_filters_prompt_tools(self):
        """Profile 的工具白名单正确过滤 prompt"""
        kernel, _ = _create_kernel_with_koakuma()

        profile_atom = _make_profile_atom(
            agent_id="restricted_agent",
            allowed_verbs=["READ", "RUN"],
            allowed_tools=["sys_clock"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=profile_atom)

        profile = kernel.load_agent_profile("restricted_agent")
        mtp_prompt = kernel.get_mtp_prompt(profile=profile)

        # 允许的工具应该出现
        assert "sys_clock" in mtp_prompt

        # 禁止的工具不应该出现
        assert "sys_write_file" not in mtp_prompt
        assert "sys_python_repl" not in mtp_prompt
        assert "sys_web_search" not in mtp_prompt

    async def test_no_profile_renders_full_prompt(self):
        """无 profile 时渲染完整 prompt（兜底逻辑）"""
        kernel, _ = _create_kernel_with_koakuma()

        # 不加载 profile
        mtp_prompt = kernel.get_mtp_prompt(profile=None)

        # 所有动词都应该出现
        assert "- SEARCH:" in mtp_prompt
        assert "- READ:" in mtp_prompt
        assert "- RUN:" in mtp_prompt
        assert "- WRITE:" in mtp_prompt
        assert "- UPDATE:" in mtp_prompt

        # 所有默认工具都应该出现
        assert "sys_clock" in mtp_prompt
        assert "sys_web_search" in mtp_prompt


@pytest.mark.asyncio
class TestPromptToKoakumaEnforcement:
    """Prompt → Koakuma 运行时拦截链路测试"""

    async def test_koakuma_enforces_verb_permission(self):
        """Koakuma 运行时拦截越权动词"""
        kernel, koakuma = _create_kernel_with_koakuma()

        # 设置限制性 profile
        profile_atom = _make_profile_atom(
            agent_id="reader_only",
            allowed_verbs=["READ", "SEARCH"],
            allowed_tools=["sys_clock"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=profile_atom)

        profile = kernel.load_agent_profile("reader_only")
        koakuma.set_active_profile(profile)

        # 允许的动词应该通过
        koakuma._check_verb_permission("READ")
        koakuma._check_verb_permission("SEARCH")

        # 禁止的动词应该被拦截
        with pytest.raises(PermissionDeniedError):
            koakuma._check_verb_permission("WRITE")

        with pytest.raises(PermissionDeniedError):
            koakuma._check_verb_permission("UPDATE")

    async def test_koakuma_enforces_tool_permission(self):
        """Koakuma 运行时拦截越权工具"""
        kernel, koakuma = _create_kernel_with_koakuma()

        profile_atom = _make_profile_atom(
            agent_id="safe_agent",
            allowed_verbs=["READ", "RUN"],
            allowed_tools=["sys_clock"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=profile_atom)

        profile = kernel.load_agent_profile("safe_agent")
        koakuma.set_active_profile(profile)

        # 允许的工具应该通过
        koakuma._check_tool_permission("sys_clock")

        # 禁止的工具应该被拦截
        with pytest.raises(PermissionDeniedError):
            koakuma._check_tool_permission("sys_write_file")

        with pytest.raises(PermissionDeniedError):
            koakuma._check_tool_permission("sys_bash_exec")


@pytest.mark.asyncio
class TestEndToEndPermissionChain:
    """端到端权限链路测试"""

    async def test_reviewer_profile_blocks_write(self):
        """Reviewer profile: prompt 不显示 WRITE，运行时拦截 WRITE"""
        kernel, koakuma = _create_kernel_with_koakuma()

        # 加载 reviewer profile
        reviewer_atom = _make_profile_atom(
            agent_id="reviewer_doll",
            allowed_verbs=["READ", "SEARCH"],
            allowed_tools=["sys_clock"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=reviewer_atom)

        profile = kernel.load_agent_profile("reviewer_doll")

        # 1. Prompt 层：WRITE 不应该出现
        mtp_prompt = kernel.get_mtp_prompt(profile=profile)
        assert "- WRITE:" not in mtp_prompt
        assert "- UPDATE:" not in mtp_prompt

        # 2. Runtime 层：Koakuma 拦截 WRITE
        koakuma.set_active_profile(profile)

        with pytest.raises(PermissionDeniedError) as exc_info:
            koakuma._check_verb_permission("WRITE")

        assert "WRITE" in str(exc_info.value)

    async def test_coder_profile_allows_write(self):
        """Coder profile: prompt 显示 WRITE，运行时允许 WRITE"""
        kernel, koakuma = _create_kernel_with_koakuma()

        # 加载 coder profile
        coder_atom = _make_profile_atom(
            agent_id="coder_doll",
            allowed_verbs=["READ", "SEARCH", "WRITE", "RUN"],
            allowed_tools=["sys_clock", "sys_read_file", "sys_write_file"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=coder_atom)

        profile = kernel.load_agent_profile("coder_doll")

        # 1. Prompt 层：WRITE 应该出现
        mtp_prompt = kernel.get_mtp_prompt(profile=profile)
        assert "- WRITE:" in mtp_prompt
        assert "- RUN:" in mtp_prompt
        assert "sys_write_file" in mtp_prompt

        # 2. Runtime 层：Koakuma 允许 WRITE
        koakuma.set_active_profile(profile)

        # 应该不抛异常
        koakuma._check_verb_permission("WRITE")
        koakuma._check_verb_permission("RUN")
        koakuma._check_tool_permission("sys_write_file")

    async def test_no_profile_allows_all_operations(self):
        """无 profile 时：prompt 显示全部，运行时允许全部"""
        kernel, koakuma = _create_kernel_with_koakuma()

        # 1. Prompt 层：全部动词和工具都应该出现
        mtp_prompt = kernel.get_mtp_prompt(profile=None)
        assert "- SEARCH:" in mtp_prompt
        assert "- READ:" in mtp_prompt
        assert "- WRITE:" in mtp_prompt
        assert "- UPDATE:" in mtp_prompt
        assert "- RUN:" in mtp_prompt

        # 2. Runtime 层：Koakuma 允许全部（无 profile）
        koakuma.set_active_profile(None)

        # 应该不抛异常
        koakuma._check_verb_permission("WRITE")
        koakuma._check_verb_permission("UPDATE")
        koakuma._check_verb_permission("RUN")
        koakuma._check_tool_permission("sys_write_file")
        koakuma._check_tool_permission("sys_bash_exec")


@pytest.mark.asyncio
class TestSecurityScenarios:
    """安全场景测试"""

    async def test_prompt_injection_cannot_bypass_runtime(self):
        """Prompt 注入无法绕过运行时拦截"""
        kernel, koakuma = _create_kernel_with_koakuma()

        # 设置限制性 profile
        profile_atom = _make_profile_atom(
            agent_id="restricted",
            allowed_verbs=["READ"],
            allowed_tools=[],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=profile_atom)

        profile = kernel.load_agent_profile("restricted")
        koakuma.set_active_profile(profile)

        # 即使 LLM 幻觉输出了 WRITE 指令，运行时也会拦截
        with pytest.raises(PermissionDeniedError):
            koakuma._check_verb_permission("WRITE")

    async def test_tool_permission_exact_match(self):
        """工具权限精确匹配（防止前缀攻击）"""
        kernel, koakuma = _create_kernel_with_koakuma()

        profile_atom = _make_profile_atom(
            agent_id="limited",
            allowed_verbs=["RUN"],
            allowed_tools=["sys_clock"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=profile_atom)

        profile = kernel.load_agent_profile("limited")
        koakuma.set_active_profile(profile)

        # sys_clock 应该通过
        koakuma._check_tool_permission("sys_clock")

        # sys_clock_evil 不应该通过（不是前缀匹配）
        with pytest.raises(PermissionDeniedError):
            koakuma._check_tool_permission("sys_clock_evil")

    async def test_profile_switch_updates_permissions(self):
        """Profile 切换正确更新权限"""
        kernel, koakuma = _create_kernel_with_koakuma()

        # 第一个 profile：限制性
        restrictive_atom = _make_profile_atom(
            agent_id="restrictive",
            allowed_verbs=["READ"],
            allowed_tools=[],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=restrictive_atom)

        profile1 = kernel.load_agent_profile("restrictive")
        koakuma.set_active_profile(profile1)

        # WRITE 应该被拒绝
        with pytest.raises(PermissionDeniedError):
            koakuma._check_verb_permission("WRITE")

        # 切换到第二个 profile：宽松
        permissive_atom = _make_profile_atom(
            agent_id="permissive",
            allowed_verbs=["READ", "WRITE"],
            allowed_tools=["sys_write_file"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=permissive_atom)

        profile2 = kernel.load_agent_profile("permissive")
        koakuma.set_active_profile(profile2)

        # WRITE 现在应该通过
        koakuma._check_verb_permission("WRITE")
        koakuma._check_tool_permission("sys_write_file")


@pytest.mark.asyncio
class TestProfileLoadingErrors:
    """Profile 加载错误处理测试"""

    async def test_profile_not_found_uses_default(self):
        """Profile 不存在时使用 OMNI_DOLL_PROFILE（兜底）"""
        kernel, koakuma = _create_kernel_with_koakuma()

        # 模拟 profile 不存在
        kernel.storage.get_memory_by_alias = Mock(return_value=None)

        profile = kernel.load_agent_profile("nonexistent")
        # 应该返回 OMNI_DOLL_PROFILE，不是 None
        from hivememory.core.models import OMNI_DOLL_PROFILE
        assert profile is OMNI_DOLL_PROFILE

        # OMNI_DOLL 允许全部操作
        koakuma.set_active_profile(profile)
        koakuma._check_verb_permission("WRITE")
        koakuma._check_tool_permission("sys_bash_exec")

    async def test_malformed_profile_returns_none(self):
        """格式错误的 profile 返回 OMNI_DOLL_PROFILE（兜底）"""
        kernel, koakuma = _create_kernel_with_koakuma()

        # 模拟格式错误的 profile（缺少 artifacts）
        broken_atom = MemoryAtom(
            id=uuid4(),
            meta=MetaData(user_id="system", source_agent_id="system"),
            index=IndexLayer(
                alias="broken",
                title="Broken",
                summary="Broken profile",
                tags=["agent"],
                memory_type=MemoryType.AGENT_PROFILE,
            ),
            payload=PayloadLayer(content="Some content", artifacts={}),
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=broken_atom)

        profile = kernel.load_agent_profile("broken")
        from hivememory.core.models import OMNI_DOLL_PROFILE
        assert profile is OMNI_DOLL_PROFILE
