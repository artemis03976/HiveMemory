"""AgentPromptAssembler 附件 section 注入顺序的快照测试（计划 10.6 节）。

冻结规则：system prompt 各 section 固定为 MTP/系统通知 → persona →
memory_context → attachment_context → topic state；附件 section 只作为
普通上下文注入，不改变 MTP 权限、工具白名单或 MemoryCompiler target。
"""

from types import SimpleNamespace

import pytest

from hivememory.core.models import AgentProfile, TopicData
from hivememory.core.protocol.models import AgentRunContext, RetrievalResponse
from hivememory.engines.attachment_compiler import AttachmentCompileResult
from hivememory.i18n import set_default_language
from hivememory.prompts.assembler import AgentPromptAssembler
from tests.helpers.workspace import make_identity_scope


@pytest.fixture(autouse=True)
def reset_i18n():
    set_default_language("zh")
    yield
    set_default_language("zh")


def _make_koakuma_config():
    return SimpleNamespace(
        enabled=True,
        mtp_prompt=SimpleNamespace(
            enabled=True,
            include_demo=False,
            include_error_handling=False,
        ),
    )


def _make_topic_data(state_summary="state"):
    return TopicData(
        topic_id="topic_1",
        workspace_identity=make_identity_scope(user_id="u1").workspace_identity,
        topic_title="测试话题",
        state_summary=state_summary,
        blocks=(),
        last_update=1.0,
    )


def _context(attachment_context: str | None) -> AgentRunContext:
    compile_result = (
        AttachmentCompileResult(attachment_context=attachment_context)
        if attachment_context is not None
        else None
    )
    return AgentRunContext(
        identity_scope=make_identity_scope(user_id="u1", agent_id="omni_doll"),
        interaction_id="interaction-order",
        topic_id="topic_1",
        user_message="hello",
        topic_context=_make_topic_data("TOPIC-STATE"),
        retrieval_result=RetrievalResponse(memories=[]),
        memory_context="MEMORY-CONTEXT",
        agent_profile=AgentProfile(persona="PERSONA", language="zh"),
        storage_available=True,
        attachment_compile_result=compile_result,
    )


def _koakuma_off():
    return SimpleNamespace(enabled=False, mtp_prompt=None)


def test_attachment_section_is_injected_between_memory_and_topic_state() -> None:
    """捕获附件 section 注入位置漂移或丢失。"""
    assembler = AgentPromptAssembler(_koakuma_off())
    system_prompt = assembler.build_main_agent_messages(
        _context("ATTACHMENT-SECTION"),
    )[
        0
    ]["content"]

    memory_pos = system_prompt.index("MEMORY-CONTEXT")
    attachment_pos = system_prompt.index("ATTACHMENT-SECTION")
    topic_pos = system_prompt.index("TOPIC-STATE")
    persona_pos = system_prompt.index("PERSONA")

    assert persona_pos < memory_pos < attachment_pos < topic_pos


def test_none_compile_result_keeps_prompt_unchanged() -> None:
    """捕获未选择附件时注入空附件 section 或占位符。"""
    assembler = AgentPromptAssembler(_koakuma_off())
    with_attachment = assembler.build_main_agent_messages(_context("ATTACHMENT-SECTION"))
    without_attachment = assembler.build_main_agent_messages(_context(None))

    assert "ATTACHMENT-SECTION" not in without_attachment[0]["content"]
    # 移除附件 section（连同其前导连接符）后，其余内容与带附件版本一致。
    with_content = with_attachment[0]["content"]
    separator = "\n" + "\n"
    assert with_content.replace(separator + "ATTACHMENT-SECTION", "") == (
        without_attachment[0]["content"]
    )


def test_koakuma_prompt_precedes_attachment_section() -> None:
    """捕获 MTP 教学片段被排到附件 section 之后。"""
    assembler = AgentPromptAssembler(_make_koakuma_config())
    system_prompt = assembler.build_main_agent_messages(
        _context("ATTACHMENT-SECTION"),
    )[
        0
    ]["content"]

    assert system_prompt.index("ATTACHMENT-SECTION") > 0
    # MTP prompt（含"工具"教学文本）必须出现在附件 section 之前。
    assert system_prompt.find("MTP") < system_prompt.index("ATTACHMENT-SECTION")
