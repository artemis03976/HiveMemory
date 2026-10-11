"""Agent Profile 加载错误与 Omni-Doll 显式回退语义单元测试。"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
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
from hivememory.patchouli.application.agent_profile_management_service import (
    AgentProfileManagementService,
)
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


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
            agent_config={
                "model_name": "gpt-4",
                "temperature": 0.7,
                "allowed_mtp_verbs": allowed_verbs,
                "allowed_sys_tools": allowed_tools,
            },
        ),
    )


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
        scope = make_identity_scope(user_id="u1")

        for alias in (None, "", "  "):
            resolved = await service.get_agent_profile(
                alias,
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
            )
            assert resolved.profile == OMNI_DOLL_PROFILE

        get_by_alias.assert_not_awaited()

    async def test_default_and_omni_doll_aliases_use_omni_doll_profile(self):
        """显式选择 default / omni_doll 时使用 Omni-Doll。"""
        service, get_by_alias = _make_retrieval_familiar()
        scope = make_identity_scope(user_id="u1")

        for alias in ("default", "omni_doll"):
            resolved = await service.get_agent_profile(
                alias,
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
            )
            assert resolved.profile == OMNI_DOLL_PROFILE

        get_by_alias.assert_not_awaited()

    async def test_missing_custom_alias_fails_explicitly(self):
        """自定义 alias 缺失时必须显式失败，不能回退到 Omni-Doll。"""
        service, get_by_alias = _make_retrieval_familiar()
        get_by_alias.return_value = None
        scope = make_identity_scope(user_id="u1")

        with pytest.raises(AliasNotFoundError) as exc_info:
            await service.get_agent_profile(
                "nonexistent_agent",
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
            )

        assert exc_info.value.message_key == "mtp.call.profile_not_found"

    async def test_custom_alias_without_scope_is_denied(self):
        """scope 校验位于公开处理者，缺少 scope 时不进入内部 Profile 解析。"""
        bus = AsyncMock()
        service = AgentProfileManagementService(bus=bus)

        with pytest.raises(ScopeRequiredError):
            await service.get_agent_profile(
                "custom_agent",
                identity_scope=None,  # type: ignore[arg-type]
            )

        bus.request.assert_not_awaited()

    async def test_wrong_memory_type_fails_explicitly(self):
        """alias 指向非 AGENT_PROFILE 记忆时显式失败，不回退。"""
        service, get_by_alias = _make_retrieval_familiar()
        scope = make_identity_scope(user_id="u1")
        get_by_alias.return_value = MemoryAtom(
            meta=make_memory_metadata(user_id="u1", source_agent_id="system"),
            index=IndexLayer(
                alias="custom",
                title="custom title",
                summary="not a profile",
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="c"),
        )

        with pytest.raises(MemoryTypeMismatchError) as exc_info:
            await service.get_agent_profile(
                "custom",
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
            )

        assert exc_info.value.message_key == "mtp.call.profile_type_mismatch"

    async def test_malformed_profile_fails_explicitly(self):
        """AGENT_PROFILE 但 artifacts 损坏（from_atom 解析失败）时显式失败，不回退。"""
        service, get_by_alias = _make_retrieval_familiar()
        scope = make_identity_scope(user_id="u1")
        get_by_alias.return_value = MemoryAtom(
            id=uuid4(),
            meta=make_memory_metadata(user_id="u1", source_agent_id="system"),
            index=IndexLayer(
                alias="broken",
                title="Broken",
                summary="Broken profile",
                tags=["agent"],
                memory_type=MemoryType.AGENT_PROFILE,
            ),
            payload=PayloadLayer(content="c", artifacts={}),
        )

        with pytest.raises(InvalidArgumentError) as exc_info:
            await service.get_agent_profile(
                "broken",
                belong_to=scope.workspace_identity,
                from_actor=scope.actor_identity,
            )

        assert exc_info.value.message_key == "mtp.call.profile_invalid"

    async def test_valid_custom_alias_returns_parsed_profile(self):
        """合法自定义 alias 返回真实解析的 AgentProfile。"""
        service, get_by_alias = _make_retrieval_familiar()
        get_by_alias.return_value = _make_profile_atom("coder_doll", ["READ", "RUN"], ["sys_clock"])
        scope = make_identity_scope(user_id="system")

        resolved = await service.get_agent_profile(
            "coder_doll",
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
        )

        profile = resolved.profile
        assert profile.persona == "You are coder_doll."
        assert profile.agent_id == "coder_doll"
        assert profile.model_name == "gpt-4"
        assert profile.temperature == pytest.approx(0.7)
