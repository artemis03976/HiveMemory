"""
Agent 权限链路单元测试。

测试覆盖:
- Profile 的动词白名单正确过滤 MTP prompt
- Profile 的工具白名单正确过滤 MTP prompt
- 无 profile 时渲染完整 prompt（兜底逻辑）
- Omni-Doll 兜底语义回归：仅在未指定 agent 或显式选择 default/omni_doll 时生效
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
    MemoryTypeMismatchError,
)
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from hivememory.prompts.mtp import MTPPromptBuilder
from tests.helpers.workspace import make_identity_scope
from tests.helpers.memory import make_memory_metadata


def _make_profile_atom(
    agent_id: str,
    allowed_verbs: list,
    allowed_tools: list,
) -> MemoryAtom:
    """构建 AGENT_PROFILE 类型的 MemoryAtom"""
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(user_id="system", source_agent_id="system"),
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


def _make_retrieval_familiar() -> tuple[RetrievalFamiliar, AsyncMock]:
    """构造真实 RetrievalFamiliar + mock 存储边界（仅替换 get_by_alias）。"""
    memory_library = MagicMock()
    get_by_alias = AsyncMock()
    memory_library.mid_term.get_by_alias = get_by_alias
    service = RetrievalFamiliar(
        engine=MagicMock(),
        memory_library=memory_library,
    )
    return service, get_by_alias


@pytest.mark.asyncio
class TestProfileLoadingErrors:
    """Omni-Doll 兜底语义回归测试。

    核心契约：Omni-Doll profile 只在「未指定 agent 或显式选择 default/omni_doll」时使用；
    任何自定义 alias 的缺失、越权、类型错误或配置损坏都必须显式失败，不能静默回退。
    """

    async def test_unset_alias_uses_omni_doll_profile(self):
        """未指定 agent 时使用 Omni-Doll（不触发存储查询）。"""
        service, get_by_alias = _make_retrieval_familiar()

        for alias in (None, "", "  "):
            profile = await service.get_agent_profile(
                alias,
                identity_scope=make_identity_scope(user_id="u1"),
            )
            assert profile is OMNI_DOLL_PROFILE

        get_by_alias.assert_not_awaited()

    async def test_default_and_omni_doll_aliases_use_omni_doll_profile(self):
        """显式选择 default / omni_doll 时使用 Omni-Doll。"""
        service, get_by_alias = _make_retrieval_familiar()

        for alias in ("default", "omni_doll"):
            profile = await service.get_agent_profile(
                alias,
                identity_scope=make_identity_scope(user_id="u1"),
            )
            assert profile is OMNI_DOLL_PROFILE

        get_by_alias.assert_not_awaited()

    async def test_missing_custom_alias_fails_explicitly(self):
        """自定义 alias 缺失时必须显式失败，不能回退到 Omni-Doll。"""
        service, get_by_alias = _make_retrieval_familiar()
        get_by_alias.return_value = None

        with pytest.raises(AliasNotFoundError) as exc_info:
            await service.get_agent_profile(
                "nonexistent_agent",
                identity_scope=make_identity_scope(user_id="u1"),
            )

        assert exc_info.value.message_key == "mtp.call.profile_not_found"

    async def test_custom_alias_without_scope_is_denied(self):
        """自定义 alias 且无 scope 时拒绝（不触发存储查询）。"""
        service, get_by_alias = _make_retrieval_familiar()

        with pytest.raises(ScopeRequiredError):
            await service.get_agent_profile(
                "custom_agent",
                identity_scope=None,  # type: ignore[arg-type]
            )

        get_by_alias.assert_not_awaited()

    async def test_wrong_memory_type_fails_explicitly(self):
        """alias 指向非 AGENT_PROFILE 记忆时显式失败，不回退。"""
        service, get_by_alias = _make_retrieval_familiar()
        get_by_alias.return_value = MemoryAtom(
            meta=make_memory_metadata(user_id="u1", source_agent_id="system"),
            index=IndexLayer(
                alias="custom", title="custom title", summary="not a profile",
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="c"),
        )

        with pytest.raises(MemoryTypeMismatchError) as exc_info:
            await service.get_agent_profile(
                "custom",
                identity_scope=make_identity_scope(user_id="u1"),
            )

        assert exc_info.value.message_key == "mtp.call.profile_type_mismatch"

    async def test_malformed_profile_fails_explicitly(self):
        """AGENT_PROFILE 但 artifacts 损坏（from_atom 解析失败）时显式失败，不回退。"""
        service, get_by_alias = _make_retrieval_familiar()
        get_by_alias.return_value = MemoryAtom(
            id=uuid4(),
            meta=make_memory_metadata(user_id="u1", source_agent_id="system"),
            index=IndexLayer(
                alias="broken", title="Broken", summary="Broken profile",
                tags=["agent"], memory_type=MemoryType.AGENT_PROFILE,
            ),
            payload=PayloadLayer(content="c", artifacts={}),
        )

        with pytest.raises(InvalidArgumentError) as exc_info:
            await service.get_agent_profile(
                "broken",
                identity_scope=make_identity_scope(user_id="u1"),
            )

        assert exc_info.value.message_key == "mtp.call.profile_invalid"

    async def test_valid_custom_alias_returns_parsed_profile(self):
        """合法自定义 alias 返回真实解析的 AgentProfile。"""
        service, get_by_alias = _make_retrieval_familiar()
        get_by_alias.return_value = _make_profile_atom(
            "coder_doll", ["READ", "RUN"], ["sys_clock"]
        )

        profile = await service.get_agent_profile(
            "coder_doll",
            identity_scope=make_identity_scope(user_id="system"),
        )

        assert profile.persona == "You are coder_doll."
        assert profile.allowed_mtp_verbs == ["READ", "RUN"]
        assert profile.allowed_sys_tools == ["sys_clock"]
