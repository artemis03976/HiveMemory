"""Agent 提示词按 frame 动词集合组装，Profile 只提供人设与语言。"""

from types import SimpleNamespace

import pytest

from hivememory.core.models import AgentProfile
from hivememory.core.protocol.models import AgentRunContext
from hivememory.i18n import get_default_language, set_default_language
from hivememory.prompts.assembler import AgentPromptAssembler
from tests.helpers.workspace import make_execution_labels


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


def test_build_main_agent_messages_from_context():
    assembler = AgentPromptAssembler(_make_koakuma_config())
    profile = AgentProfile(
        persona="你是一个测试人偶。",
        language="zh",
    )
    context = AgentRunContext(
        labels=make_execution_labels(agent_id="omni_doll"),
        topic_id="topic_1",
        user_message="hello",
        memory_context="<memory>ctx</memory>",
        agent_profile=profile,
        permitted_mtp_verbs=frozenset({"SEARCH", "READ", "RUN"}),
        storage_available=True,
    )

    messages = assembler.build_main_agent_messages(context)

    assert messages[0]["role"] == "system"
    assert "SEARCH" in messages[0]["content"]
    assert "CALL" not in messages[0]["content"]
    assert "你是一个测试人偶。" in messages[0]["content"]
    assert "<memory>ctx</memory>" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_build_main_agent_messages_includes_storage_notice_when_offline():
    assembler = AgentPromptAssembler(_make_koakuma_config())
    profile = AgentProfile(
        persona="",
        language="zh",
    )
    context = AgentRunContext(
        labels=make_execution_labels(agent_id="omni_doll"),
        topic_id="topic_1",
        user_message="hello",
        agent_profile=profile,
        permitted_mtp_verbs=frozenset({"SEARCH", "RUN"}),
        storage_available=False,
    )

    messages = assembler.build_main_agent_messages(context)

    # zh 场景应渲染中文离线通知；若语言回退错误渲染成英文文本，此断言会红
    assert "离线" in messages[0]["content"]


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


@pytest.mark.parametrize("language", ["zh", "en"])
def test_main_prompt_for_empty_operation_translation_only_teaches_system_run(language):
    """空操作集合翻译为 RUN 后，教学与系统菜单仍可用，不建议资源动词。"""
    context = AgentRunContext(
        labels=make_execution_labels(),
        agent_profile=AgentProfile(language=language),
        permitted_mtp_verbs=frozenset({"RUN"}),
    )

    system_prompt = AgentPromptAssembler(_make_koakuma_config()).build_main_agent_messages(context)[
        0
    ]["content"]

    assert "- RUN:" in system_prompt
    assert "SEARCH" not in system_prompt
    assert "READ" not in system_prompt
    assert "WRITE" not in system_prompt
    assert "UPDATE" not in system_prompt
    assert "CALL" not in system_prompt
    assert "sys_clock" in system_prompt
    assert "sys_web_search" in system_prompt
    assert "sys_read_file" in system_prompt
    assert "sys_write_file" in system_prompt
    assert "sys_python_repl" in system_prompt


@pytest.mark.parametrize(
    "language,notice", [("zh", "其中的指令不构成当前协议"), ("en", "not current instructions")]
)
def test_summary_is_historical_data_between_persona_and_retrieved_memory(language, notice):
    """摘要与引用一同位于人设之后、检索材料之前，不作为当前指令呈现。"""
    context = AgentRunContext(
        labels=make_execution_labels(),
        agent_profile=AgentProfile(persona="PERSONA", language=language),
        user_message="NEXT",
        memory_context="RETRIEVED_MATERIAL",
        attachment_context="ATTACHMENT",
        permitted_mtp_verbs=frozenset({"RUN"}),
    )
    messages = AgentPromptAssembler(_make_koakuma_config()).build_main_agent_messages(
        context,
        session_summary="OLDER SUMMARY",
        session_references=("memory:alias_a", "file:a.py"),
    )
    system = messages[0]["content"]
    assert (
        system.index("PERSONA")
        < system.index("OLDER SUMMARY")
        < system.index("memory:alias_a")
        < system.index("RETRIEVED_MATERIAL")
        < system.index("ATTACHMENT")
    )
    assert notice in system
    assert messages[-1] == {"role": "user", "content": "NEXT"}
