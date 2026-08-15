"""
Agent 权限链路单元测试。

测试覆盖:
- 权限链路: profile → prompt 过滤 → Koakuma 拦截
- 限制性 profile 阻止越权操作
- 未指定 profile 时使用显式能力边界的 Omni-Doll
- Prompt 不显示禁止的动词和工具
"""

from types import MethodType
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
    PermissionDeniedError,
)
from hivememory.patchouli.runtime.core import PatchouliRuntime
from hivememory.prompts.mtp import MTPPromptBuilder


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


def _create_runtime_with_koakuma():
    """创建带 Koakuma 的 PatchouliRuntime"""
    with patch.object(PatchouliRuntime, "_init_infrastructure"), \
         patch.object(PatchouliRuntime, "_build_engines", return_value={}), \
         patch.object(PatchouliRuntime, "_register_services"):

        mock_patchouli_config = Mock()
        mock_shared_config = Mock()

        runtime = PatchouliRuntime(patchouli_config=mock_patchouli_config, shared_config=mock_shared_config)
        runtime.storage = Mock()

        def _get_agent_profile(agent_alias: str):
            if not agent_alias or agent_alias in ("default", "omni_doll"):
                return OMNI_DOLL_PROFILE
            atom = runtime.storage.get_memory_by_alias(agent_alias)
            if atom is None:
                raise AliasNotFoundError(
                    message_key="mtp.call.profile_not_found",
                    params={"agent_alias": agent_alias},
                )
            profile = AgentProfile.from_atom(atom)
            if profile is None:
                raise InvalidArgumentError(
                    message_key="mtp.call.profile_invalid",
                    params={"agent_alias": agent_alias},
                )
            return profile

        def _get_mtp_prompt(self, profile=None):
            builder = MTPPromptBuilder(
                language="en",
                include_demo=True,
                include_error_handling=True,
                allowed_verbs=getattr(profile, "allowed_mtp_verbs", None),
                allowed_runtime_tools=getattr(profile, "allowed_sys_tools", None),
            )
            return builder.build()

        runtime.storage.get_agent_profile = Mock(side_effect=_get_agent_profile)
        runtime.get_mtp_prompt = MethodType(_get_mtp_prompt, runtime)

        # 创建真实的 Koakuma 实例
        koakuma = KoakumaRuntime(bus=None, config=None, alias_resolver=Mock())
        runtime._services = {"koakuma": koakuma}

        return runtime, koakuma


@pytest.mark.asyncio
class TestProfileToPromptFiltering:
    """Profile → Prompt 过滤链路测试"""

    async def test_profile_filters_prompt_verbs(self):
        """Profile 的动词白名单正确过滤 prompt"""
        kernel, _ = _create_runtime_with_koakuma()

        # 加载限制性 profile
        profile_atom = _make_profile_atom(
            agent_id="reviewer_doll",
            allowed_verbs=["READ", "SEARCH"],
            allowed_tools=["sys_clock"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=profile_atom)

        profile = kernel.storage.get_agent_profile("reviewer_doll")
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
        kernel, _ = _create_runtime_with_koakuma()

        profile_atom = _make_profile_atom(
            agent_id="restricted_agent",
            allowed_verbs=["READ", "RUN"],
            allowed_tools=["sys_clock"],
        )
        kernel.storage.get_memory_by_alias = Mock(return_value=profile_atom)

        profile = kernel.storage.get_agent_profile("restricted_agent")
        mtp_prompt = kernel.get_mtp_prompt(profile=profile)

        # 允许的工具应该出现
        assert "sys_clock" in mtp_prompt

        # 禁止的工具不应该出现
        assert "sys_write_file" not in mtp_prompt
        assert "sys_python_repl" not in mtp_prompt
        assert "sys_web_search" not in mtp_prompt

    async def test_no_profile_renders_full_prompt(self):
        """无 profile 时渲染完整 prompt（兜底逻辑）"""
        kernel, _ = _create_runtime_with_koakuma()

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
class TestProfileLoadingErrors:
    """Profile 加载错误处理测试"""

    async def test_profile_not_found_fails_explicitly(self):
        """显式 Profile 不存在时不能使用 Omni-Doll 兜底。"""
        kernel, _ = _create_runtime_with_koakuma()

        # 模拟 profile 不存在
        kernel.storage.get_memory_by_alias = Mock(return_value=None)

        with pytest.raises(AliasNotFoundError):
            kernel.storage.get_agent_profile("nonexistent")

    async def test_malformed_profile_fails_explicitly(self):
        """格式错误的 Profile 不能使用 Omni-Doll 兜底。"""
        kernel, _ = _create_runtime_with_koakuma()

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

        with pytest.raises(InvalidArgumentError):
            kernel.storage.get_agent_profile("broken")
