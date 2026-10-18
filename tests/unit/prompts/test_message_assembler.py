"""CALL 子帧按动词集合组装教学，Profile 提供人设与语言。"""

from types import SimpleNamespace

import pytest

from hivememory.core.models import AgentProfile
from hivememory.i18n import get_default_language, set_default_language
from hivememory.prompts.assembler import AgentPromptAssembler


@pytest.fixture(autouse=True)
def reset_i18n():
    previous = get_default_language()
    set_default_language("zh")
    yield
    set_default_language(previous)


def _make_koakuma_config():
    mtp_prompt_config = SimpleNamespace(
        enabled=True,
        include_demo=False,
        include_error_handling=False,
    )
    return SimpleNamespace(
        enabled=True,
        mtp_prompt=mtp_prompt_config,
    )


def test_build_sub_agent_messages_disables_call():
    assembler = AgentPromptAssembler(_make_koakuma_config())
    profile = AgentProfile(
        persona="子 Agent",
        language="zh",
    )

    messages = assembler.build_sub_agent_messages(
        profile=profile,
        task="Write unit tests",
        shared_context="[Shared Context]",
        permitted_verbs=frozenset({"READ", "RUN", "CALL"}),
    )

    assert messages[0]["role"] == "system"
    assert "CALL" not in messages[0]["content"]
    assert "- READ:" in messages[0]["content"]
    assert "- RUN:" in messages[0]["content"]
    assert "SEARCH" not in messages[0]["content"]
    assert "sys_write_file" in messages[0]["content"]
    assert "sys_python_repl" in messages[0]["content"]
    assert "[Shared Context]" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Write unit tests"}


def test_mtp_prompt_uses_resolved_profile_language():
    assembler = AgentPromptAssembler(_make_koakuma_config())
    profile = AgentProfile(
        persona="",
        language="en",
    )

    messages = assembler.build_sub_agent_messages(
        profile=profile,
        task="Search memory",
        shared_context="",
        permitted_verbs=frozenset({"SEARCH", "RUN"}),
    )

    assert "You are an intelligent Agent running on HiveMemory" in messages[0]["content"]
    assert "你是运行在 HiveMemory" not in messages[0]["content"]


def test_mtp_prompt_uses_global_language_without_profile_language():
    set_default_language("en")
    assembler = AgentPromptAssembler(_make_koakuma_config())

    messages = assembler.build_sub_agent_messages(
        profile=None,
        task="Search memory",
        shared_context="",
        permitted_verbs=frozenset({"SEARCH", "RUN"}),
    )

    assert "You are an intelligent Agent running on HiveMemory" in messages[0]["content"]
