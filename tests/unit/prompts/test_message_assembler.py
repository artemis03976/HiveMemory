from types import SimpleNamespace

from hivememory.core.models import AgentProfile, Identity
from hivememory.core.protocol.models import AgentRunContext, RetrievalResponse
from hivememory.patchouli.memory_library.models import TopicData
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.i18n import set_default_language
import pytest


@pytest.fixture(autouse=True)
def reset_i18n():
    set_default_language("zh")
    yield
    set_default_language("zh")


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


def _make_topic_data(state_summary="state"):
    return TopicData(
        topic_id="topic_1",
        user_id="u1",
        topic_title="测试话题",
        state_summary=state_summary,
        blocks=(),
        last_update=1.0,
        last_accessed_at=1.0,
    )


def test_build_main_agent_messages_from_context():
    assembler = AgentPromptAssembler(_make_koakuma_config())
    profile = AgentProfile(
        persona="你是一个测试人偶。",
        allowed_mtp_verbs=["SEARCH", "READ"],
        allowed_sys_tools=["sys_clock"],
        language="zh",
    )
    context = AgentRunContext(
        identity=Identity(user_id="u1", agent_id="omni_doll"),
        topic_id="topic_1",
        user_message="hello",
        topic_context=_make_topic_data("state"),
        retrieval_result=RetrievalResponse(memories=[]),
        memory_context="<memory>ctx</memory>",
        agent_profile=profile,
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
        allowed_mtp_verbs=["SEARCH"],
        allowed_sys_tools=[],
        language="zh",
    )
    context = AgentRunContext(
        identity=Identity(user_id="u1", agent_id="omni_doll"),
        topic_id="topic_1",
        user_message="hello",
        topic_context=None,
        retrieval_result=RetrievalResponse(),
        agent_profile=profile,
        storage_available=False,
    )

    messages = assembler.build_main_agent_messages(context)

    assert "OFFLINE" in messages[0]["content"] or "离线" in messages[0]["content"]


def test_build_sub_agent_messages_disables_call():
    assembler = AgentPromptAssembler(_make_koakuma_config())
    profile = AgentProfile(
        persona="子 Agent",
        allowed_mtp_verbs=None,
        allowed_sys_tools=None,
        language="zh",
    )

    messages = assembler.build_sub_agent_messages(
        profile=profile,
        task="Write unit tests",
        shared_context="[Shared Context]",
        depth=1,
    )

    assert messages[0]["role"] == "system"
    assert "CALL" not in messages[0]["content"]
    assert "[Shared Context]" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Write unit tests"}


def test_mtp_prompt_uses_resolved_profile_language():
    assembler = AgentPromptAssembler(_make_koakuma_config())
    profile = AgentProfile(
        persona="",
        allowed_mtp_verbs=["SEARCH"],
        allowed_sys_tools=[],
        language="en",
    )

    messages = assembler.build_sub_agent_messages(
        profile=profile,
        task="Search memory",
        shared_context="",
        depth=0,
    )

    assert "You are an intelligent Agent running on HiveOS" in messages[0]["content"]
    assert "你是运行在 HiveOS" not in messages[0]["content"]


def test_mtp_prompt_uses_global_language_without_profile_language():
    set_default_language("en")
    assembler = AgentPromptAssembler(_make_koakuma_config())

    prompt = assembler._build_mtp_prompt(profile=None)

    assert "You are an intelligent Agent running on HiveOS" in prompt
