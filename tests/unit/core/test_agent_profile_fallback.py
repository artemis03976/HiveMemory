"""Agent Profile 的默认模型参数，以及存量图纸中已删除的键（权限、语言）在解析时被忽略。"""

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from tests.helpers.memory import make_memory_metadata


def test_omni_doll_profile_uses_default_model_parameters_without_permissions() -> None:
    """内置图纸提供固定身份与默认运行配置，权限不附着在 Profile 上。"""
    assert OMNI_DOLL_PROFILE.model_dump() == {
        "agent_id": "omni_doll",
        "persona": "",
        "model_name": "default",
        "temperature": None,
        "top_p": None,
        "enabled": True,
    }


def test_profile_parsing_ignores_legacy_permission_keys() -> None:
    """权限旧键（即使含非列表值）与已删除的 language 键都被忽略，真实配置和 alias 身份仍正常解析；
    没有启用字段的存量配置读作启用。"""
    atom = MemoryAtom(
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            alias="legacy_agent", title="旧版人偶", memory_type=MemoryType.AGENT_PROFILE
        ),
        payload=PayloadLayer(
            content="专注代码审查",
            agent_config={
                "model_name": "legacy-model",
                "language": "en",
                "allowed_mtp_verbs": {"obsolete": "value"},
                "allowed_sys_tools": 42,
            },
        ),
    )

    profile = AgentProfile.from_atom(atom)

    assert isinstance(profile, AgentProfile)
    assert profile.model_dump() == {
        "agent_id": "legacy_agent",
        "persona": "专注代码审查",
        "model_name": "legacy-model",
        "temperature": None,
        "top_p": None,
        "enabled": True,
    }
