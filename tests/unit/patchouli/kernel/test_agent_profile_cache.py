"""
AgentProfile 加载与缓存单元测试

测试覆盖:
- load_agent_profile: 从存储加载图纸 / 缓存命中 / 图纸不存在时兜底
- get_agent_persona: 提取 persona 字段
- 缓存失效与重新加载
"""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, AgentProfileConfig, OMNI_DOLL_PROFILE
)
from hivememory.patchouli.kernel.core import PatchouliKernel


def _make_profile_atom(
    agent_id: str = "test_agent",
    persona: str = "You are a test agent.",
    allowed_verbs: list = None,
    allowed_tools: list = None,
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
            content=persona,
            artifacts={
                "agent_config": {
                    "model_name": "gpt-4",
                    "temperature": 0.7,
                    "allowed_mtp_verbs": ["READ", "SEARCH"] if allowed_verbs is None else allowed_verbs,
                    "allowed_sys_tools": ["sys_clock"] if allowed_tools is None else allowed_tools,
                }
            },
        ),
    )


def _create_kernel_with_storage(mock_storage=None):
    """创建带 mock storage 的 PatchouliKernel"""
    with patch.object(PatchouliKernel, "_init_infrastructure"), \
         patch.object(PatchouliKernel, "_build_engines", return_value={}), \
         patch.object(PatchouliKernel, "_register_services"), \
         patch.object(PatchouliKernel, "_register_bus_routes"):

        mock_config = Mock()
        mock_config.koakuma.enabled = True
        mock_config.koakuma.mtp_prompt.enabled = True
        mock_config.koakuma.mtp_prompt.language = "zh"
        mock_config.koakuma.mtp_prompt.include_demo = True
        mock_config.koakuma.mtp_prompt.include_error_handling = True

        kernel = PatchouliKernel(config=mock_config, bus=None)
        kernel.storage = mock_storage or Mock()
        kernel._services = {}
        return kernel


class TestLoadAgentProfile:
    """load_agent_profile() 测试"""

    def test_load_profile_from_storage_success(self):
        """从存储成功加载图纸"""
        profile_atom = _make_profile_atom(
            agent_id="coder_doll",
            persona="You are a senior Python developer.",
            allowed_verbs=["READ", "SEARCH", "RUN"],
            allowed_tools=["sys_read_file", "sys_write_file"],
        )

        mock_storage = Mock()
        mock_storage.get_memory_by_alias = Mock(return_value=profile_atom)
        kernel = _create_kernel_with_storage(mock_storage)

        profile = kernel.load_agent_profile("coder_doll")

        assert profile is not None
        assert isinstance(profile, AgentProfileConfig)
        assert profile.allowed_mtp_verbs == ["READ", "SEARCH", "RUN"]
        assert profile.allowed_sys_tools == ["sys_read_file", "sys_write_file"]
        mock_storage.get_memory_by_alias.assert_called_once_with("coder_doll")

    def test_load_profile_cache_hit(self):
        """缓存命中时不访问存储"""
        profile_atom = _make_profile_atom(agent_id="cached_agent")

        mock_storage = Mock()
        mock_storage.get_memory_by_alias = Mock(return_value=profile_atom)
        kernel = _create_kernel_with_storage(mock_storage)

        # 第一次加载
        profile1 = kernel.load_agent_profile("cached_agent")
        assert profile1 is not None
        assert mock_storage.get_memory_by_alias.call_count == 1

        # 第二次加载应命中缓存
        profile2 = kernel.load_agent_profile("cached_agent")
        assert profile2 is not None
        assert profile2.allowed_mtp_verbs == profile1.allowed_mtp_verbs
        assert mock_storage.get_memory_by_alias.call_count == 1  # 未增加

    def test_load_profile_not_found_returns_omni_doll(self):
        """图纸不存在时返回 OMNI_DOLL_PROFILE（兜底）"""
        mock_storage = Mock()
        mock_storage.get_memory_by_alias = Mock(return_value=None)
        kernel = _create_kernel_with_storage(mock_storage)

        profile = kernel.load_agent_profile("nonexistent_agent")

        assert profile is OMNI_DOLL_PROFILE
        mock_storage.get_memory_by_alias.assert_called_once_with("nonexistent_agent")

    def test_load_profile_wrong_type_returns_omni_doll(self):
        """非 AGENT_PROFILE 类型返回 OMNI_DOLL_PROFILE"""
        wrong_atom = MemoryAtom(
            id=uuid4(),
            meta=MetaData(user_id="test", source_agent_id="test"),
            index=IndexLayer(
                alias="not_a_profile",
                title="Regular Memory",
                summary="Not a profile",
                tags=["fact"],
                memory_type=MemoryType.FACT,  # 错误类型
            ),
            payload=PayloadLayer(content="Some content"),
        )

        mock_storage = Mock()
        mock_storage.get_memory_by_alias = Mock(return_value=wrong_atom)
        kernel = _create_kernel_with_storage(mock_storage)

        profile = kernel.load_agent_profile("not_a_profile")

        assert profile is OMNI_DOLL_PROFILE

    def test_load_profile_missing_artifacts_returns_omni_doll(self):
        """缺少 artifacts.agent_config 时返回 OMNI_DOLL_PROFILE"""
        broken_atom = MemoryAtom(
            id=uuid4(),
            meta=MetaData(user_id="system", source_agent_id="system"),
            index=IndexLayer(
                alias="broken_agent",
                title="Broken Profile",
                summary="Missing config",
                tags=["agent"],
                memory_type=MemoryType.AGENT_PROFILE,
            ),
            payload=PayloadLayer(
                content="Some persona",
                artifacts={},  # 缺少 agent_config
            ),
        )

        mock_storage = Mock()
        mock_storage.get_memory_by_alias = Mock(return_value=broken_atom)
        kernel = _create_kernel_with_storage(mock_storage)

        profile = kernel.load_agent_profile("broken_agent")

        assert profile is OMNI_DOLL_PROFILE

    def test_load_profile_default_alias_returns_omni_doll(self):
        """别名为 'default' 时直接返回 OMNI_DOLL_PROFILE"""
        mock_storage = Mock()
        kernel = _create_kernel_with_storage(mock_storage)

        profile = kernel.load_agent_profile("default")

        assert profile is OMNI_DOLL_PROFILE
        mock_storage.get_memory_by_alias.assert_not_called()

    def test_load_profile_empty_alias_returns_omni_doll(self):
        """空别名时直接返回 OMNI_DOLL_PROFILE"""
        mock_storage = Mock()
        kernel = _create_kernel_with_storage(mock_storage)

        profile = kernel.load_agent_profile("")

        assert profile is OMNI_DOLL_PROFILE
        mock_storage.get_memory_by_alias.assert_not_called()


class TestGetAgentPersona:
    """get_agent_persona() 测试"""

    def test_get_persona_from_cached_profile(self):
        """从缓存的图纸提取 persona"""
        profile_atom = _make_profile_atom(
            agent_id="test_agent",
            persona="You are a helpful assistant specialized in testing.",
        )

        mock_storage = Mock()
        mock_storage.get_memory_by_alias = Mock(return_value=profile_atom)
        kernel = _create_kernel_with_storage(mock_storage)

        # 先加载 profile（触发缓存）
        kernel.load_agent_profile("test_agent")

        # 获取 persona
        persona = kernel.get_agent_persona("test_agent")

        assert persona == "You are a helpful assistant specialized in testing."

    def test_get_persona_not_found_returns_empty(self):
        """图纸不存在时返回空字符串"""
        mock_storage = Mock()
        mock_storage.get_memory_by_alias = Mock(return_value=None)
        kernel = _create_kernel_with_storage(mock_storage)

        persona = kernel.get_agent_persona("nonexistent_agent")

        assert persona == ""

    def test_get_persona_default_alias_returns_empty(self):
        """default 别名返回空字符串"""
        mock_storage = Mock()
        kernel = _create_kernel_with_storage(mock_storage)

        persona = kernel.get_agent_persona("default")

        assert persona == ""


class TestProfilePermissions:
    """AgentProfileConfig 权限方法测试"""

    def test_is_verb_allowed_with_whitelist(self):
        """白名单模式：只允许列表中的动词"""
        profile = AgentProfileConfig(
            allowed_mtp_verbs=["READ", "SEARCH"],
            allowed_sys_tools=[],
        )

        assert profile.is_verb_allowed("READ") is True
        assert profile.is_verb_allowed("SEARCH") is True
        assert profile.is_verb_allowed("WRITE") is False
        assert profile.is_verb_allowed("UPDATE") is False

    def test_is_verb_allowed_empty_list_denies_all(self):
        """空列表：禁止所有动词"""
        profile = AgentProfileConfig(
            allowed_mtp_verbs=[],
            allowed_sys_tools=[],
        )

        assert profile.is_verb_allowed("READ") is False
        assert profile.is_verb_allowed("WRITE") is False
        assert profile.is_verb_allowed("UPDATE") is False

    def test_is_tool_allowed_with_whitelist(self):
        """白名单模式：只允许列表中的工具"""
        profile = AgentProfileConfig(
            allowed_mtp_verbs=[],
            allowed_sys_tools=["sys_clock", "sys_read_file"],
        )

        assert profile.is_tool_allowed("sys_clock") is True
        assert profile.is_tool_allowed("sys_read_file") is True
        assert profile.is_tool_allowed("sys_write_file") is False
        assert profile.is_tool_allowed("sys_bash_exec") is False

    def test_is_tool_allowed_empty_list_denies_all(self):
        """空列表：禁止所有工具"""
        profile = AgentProfileConfig(
            allowed_mtp_verbs=[],
            allowed_sys_tools=[],
        )

        assert profile.is_tool_allowed("sys_clock") is False
        assert profile.is_tool_allowed("sys_bash_exec") is False
        assert profile.is_tool_allowed("sys_python_repl") is False
