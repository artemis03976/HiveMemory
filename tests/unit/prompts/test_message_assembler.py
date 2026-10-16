"""Agent 提示词按 frame 动词集合组装，Profile 只提供人设与语言。"""

from types import SimpleNamespace

import pytest

from hivememory.core.models import AgentProfile
from hivememory.core.protocol.models import AgentRunContext
from hivememory.i18n import get_default_language, set_default_language
from hivememory.prompts.assembler import AgentPromptAssembler
from tests.helpers.sessions import make_conversation_turn
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


@pytest.mark.parametrize(
    ("registered_agent_id", "expected_reply"),
    [
        ("registered_agent", "历史回复"),
        ("another_agent", "[From: registered_agent]\n历史回复"),
    ],
)
def test_history_role_prefix_uses_registered_label_instead_of_profile_display_alias(
    registered_agent_id, expected_reply
):
    """历史角色归并取注册标签，Profile 的展示 alias 不改变当前发起角色。"""
    history = (
        make_conversation_turn(
            agent_id="registered_agent", user_message="历史问题", assistant_final_text="历史回复"
        ),
    )
    context = AgentRunContext(
        labels=make_execution_labels(agent_id=registered_agent_id),
        topic_id="topic_1",
        user_message="当前问题",
        session_history=history,
        agent_profile=AgentProfile(agent_id="display_agent"),
        permitted_mtp_verbs=frozenset({"RUN"}),
    )

    messages = AgentPromptAssembler(_make_koakuma_config()).build_main_agent_messages(context)

    assert messages[-3:] == [
        {"role": "user", "content": "历史问题"},
        {"role": "assistant", "content": expected_reply},
        {"role": "user", "content": "当前问题"},
    ]


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
@pytest.mark.parametrize("storage_available", [True, False], ids=["online", "offline"])
def test_main_prompt_uses_frame_verbs_and_ignores_legacy_profile_permissions(
    language, storage_available
):
    """旧 Profile 的扩权键不能放入 WRITE，工具禁用键不能缩减系统菜单。"""
    profile = AgentProfile.model_validate(
        {
            "persona": "当前人设",
            "language": language,
            "allowed_mtp_verbs": ["WRITE", "CALL"],
            "allowed_sys_tools": [],
        }
    )
    context = AgentRunContext(
        labels=make_execution_labels(),
        agent_profile=profile,
        permitted_mtp_verbs=frozenset({"READ", "SEARCH", "RUN"}),
        storage_available=storage_available,
    )
    config = SimpleNamespace(
        enabled=True,
        mtp_prompt=SimpleNamespace(enabled=True, include_demo=True, include_error_handling=True),
    )

    system_prompt = AgentPromptAssembler(config).build_main_agent_messages(context)[0]["content"]

    assert "- READ:" in system_prompt
    assert "- SEARCH:" in system_prompt
    assert "- RUN:" in system_prompt
    assert "WRITE" not in system_prompt
    assert "UPDATE" not in system_prompt
    assert "CALL" not in system_prompt
    assert "sys_clock" in system_prompt
    assert "sys_web_search" in system_prompt
    assert "sys_read_file" in system_prompt
    assert "sys_write_file" in system_prompt
    assert "sys_python_repl" in system_prompt


@pytest.mark.parametrize("language", ["zh", "en"])
def test_sub_prompt_inherits_verbs_without_legacy_profile_restrictions(language):
    """被调用方旧空白名单不再缩减父 frame 的 READ、RUN 与完整系统工具。"""
    profile = AgentProfile.model_validate(
        {"language": language, "allowed_mtp_verbs": [], "allowed_sys_tools": []}
    )

    system_prompt = AgentPromptAssembler(_make_koakuma_config()).build_sub_agent_messages(
        profile=profile,
        task="子任务",
        permitted_verbs=frozenset({"READ", "RUN"}),
    )[0]["content"]

    assert "- READ:" in system_prompt
    assert "- RUN:" in system_prompt
    assert "SEARCH" not in system_prompt
    assert "WRITE" not in system_prompt
    assert "CALL" not in system_prompt
    assert "sys_write_file" in system_prompt
    assert "sys_python_repl" in system_prompt


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
