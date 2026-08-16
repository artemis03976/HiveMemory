"""
Agent 权限链路单元测试。

测试覆盖:
- Profile 的动词白名单正确过滤 MTP prompt
- Profile 的工具白名单正确过滤 MTP prompt
- 无 profile 时渲染完整 prompt（兜底逻辑）
"""

from uuid import uuid4

import pytest

from hivememory.core.models import (
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)
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


def _profile(agent_id: str, allowed_verbs: list, allowed_tools: list) -> AgentProfile:
    profile = AgentProfile.from_atom(
        _make_profile_atom(agent_id, allowed_verbs, allowed_tools)
    )
    assert profile is not None
    return profile


def _build_prompt(profile: AgentProfile | None) -> str:
    return MTPPromptBuilder(
        language="en",
        include_demo=True,
        include_error_handling=True,
        allowed_verbs=getattr(profile, "allowed_mtp_verbs", None),
        allowed_runtime_tools=getattr(profile, "allowed_sys_tools", None),
    ).build()


@pytest.mark.asyncio
class TestProfileToPromptFiltering:
    """Profile → Prompt 过滤链路测试"""

    async def test_profile_filters_prompt_verbs(self):
        """Profile 的动词白名单正确过滤 prompt"""
        profile = _profile(
            agent_id="reviewer_doll",
            allowed_verbs=["READ", "SEARCH"],
            allowed_tools=["sys_clock"],
        )
        mtp_prompt = _build_prompt(profile)

        # 允许的动词应该出现
        assert "- READ:" in mtp_prompt
        assert "- SEARCH:" in mtp_prompt

        # 禁止的动词不应该出现
        assert "- WRITE:" not in mtp_prompt
        assert "- UPDATE:" not in mtp_prompt
        assert "- RUN:" not in mtp_prompt

    async def test_profile_filters_prompt_tools(self):
        """Profile 的工具白名单正确过滤 prompt"""
        profile = _profile(
            agent_id="restricted_agent",
            allowed_verbs=["READ", "RUN"],
            allowed_tools=["sys_clock"],
        )
        mtp_prompt = _build_prompt(profile)

        # 允许的工具应该出现
        assert "sys_clock" in mtp_prompt

        # 禁止的工具不应该出现
        assert "sys_write_file" not in mtp_prompt
        assert "sys_python_repl" not in mtp_prompt
        assert "sys_web_search" not in mtp_prompt

    async def test_no_profile_renders_full_prompt(self):
        """无 profile 时渲染完整 prompt（兜底逻辑）"""
        mtp_prompt = _build_prompt(profile=None)

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
