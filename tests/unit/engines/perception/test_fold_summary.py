"""
折叠摘要写作者的单元测试

测试覆盖:
- 这一次的摘要只为被折叠的轮次而写：标题写明覆盖的轮次，此前的摘要只作模型的上下文、不被改写
- 规则抽取的部分总是写出：逐轮的用户消息（压成一行、截断），工具调用参数中的路径与能力调用的
  作用对象、产出的 alias，失败的工具调用（工具名与结果的第一行），用到的工具
- 模型以 JSON 写判断类的五栏，代码按固定顺序渲染，空的栏不写
- 没有可用模型、提供商失败、超时或输出不合格时只写标题与规则抽取的部分，并注明没有模型写的部分；
  程序错误不降级
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.config.patchouli import (
    GenerationTranscriptConfig,
    LLMSummaryWriterConfig,
)
from hivememory.core.errors import ModelProviderError
from hivememory.core.models import TurnEvent
from hivememory.engines.perception.fold_summary import (
    LLMFoldSummaryWriter,
    RuleFoldSummaryWriter,
    create_fold_summary_writer,
)
from hivememory.i18n import get_fold_summary_prompt_text
from tests.helpers.llm import DEFAULT_LLM_PARAMS, ScriptedLLMService
from tests.helpers.sessions import make_logical_block

TRANSCRIPT = GenerationTranscriptConfig()


def _process_block():
    """一轮带过程的交互：思考、外部工具读文件与改文件（改文件失败）、能力调用的读取与写入。"""
    return make_logical_block(
        turn_events=[
            TurnEvent(kind="user_message", sequence=0, content="创建认证模块\n  先看现有代码"),
            TurnEvent(kind="thought", sequence=1, content="先看看已有的鉴权代码"),
            TurnEvent(
                kind="tool_call",
                sequence=2,
                content="",
                action_id="toolu-1",
                tool_name="Read",
                tool_args={"file_path": "src/auth/session.py"},
                status="success",
                result="class Session: ...",
            ),
            TurnEvent(
                kind="tool_call",
                sequence=3,
                content="",
                action_id="toolu-2",
                tool_name="apply_patch",
                tool_args={"paths": ["src/auth/login.py", "src/auth/session.py"]},
                status="error",
                result="patch rejected: context mismatch\nhunk 2 of 3",
            ),
            TurnEvent(
                kind="tool_call",
                sequence=4,
                content="⟪ READ | mem_login ⟫",
                action_id="read",
                verb="READ",
                targets=("mem_login",),
                status="success",
                result="登录流程的旧记忆",
            ),
            TurnEvent(
                kind="tool_call",
                sequence=5,
                content="⟪ WRITE ⟫",
                action_id="write",
                verb="WRITE",
                status="ack",
                result="已登记",
                produced_aliases=("pending_auth",),
            ),
            TurnEvent(kind="assistant_message", sequence=6, content="已创建 auth.py"),
        ],
        total_tokens=50,
    )


# 规则抽取的部分：被折叠的两轮从第 3 轮开始，第二轮是 _process_block。
RULE_PART = (
    "### 用户消息\n"
    "- 第 3 轮：查询天气\n"
    "- 第 4 轮：创建认证模块 先看现有代码\n"
    "### 文件与标识符\n"
    "- src/auth/session.py\n"
    "- src/auth/login.py\n"
    "- mem_login\n"
    "- pending_auth\n"
    "### 失败的工具调用\n"
    "- 第 4 轮 apply_patch：patch rejected: context mismatch\n"
    "### 用到的工具\n"
    "Read, apply_patch, READ, WRITE"
)


def _folded_blocks():
    return [make_logical_block("查询天气", "晴"), _process_block()]


def _llm_writer(llm, *, timeout_seconds: float = 60) -> LLMFoldSummaryWriter:
    return LLMFoldSummaryWriter(
        summary_llm=llm,
        llm_params=DEFAULT_LLM_PARAMS,
        summary_timeout_seconds=timeout_seconds,
        transcript_config=TRANSCRIPT,
    )


class TestRuleFoldSummaryWriter:
    """按规则写折叠摘要：只有标题与规则抽取的部分"""

    @pytest.mark.asyncio
    async def test_summary_is_the_title_and_the_rule_extracted_part(self):
        """用户消息逐轮列出；路径取自工具参数，作用对象与 alias 取自能力调用，按首次出现去重。"""
        summary = await RuleFoldSummaryWriter().generate_summary(
            _folded_blocks(), first_turn_number=3, earlier_summary="此前的摘要"
        )

        assert summary == "## 第 3–4 轮\n" + RULE_PART

    @pytest.mark.asyncio
    async def test_long_user_message_is_cut_and_a_single_turn_has_a_one_turn_title(self):
        summary = await RuleFoldSummaryWriter().generate_summary(
            [make_logical_block("长" * 250, "好")], first_turn_number=1, earlier_summary=""
        )

        assert summary == "## 第 1 轮\n### 用户消息\n- 第 1 轮：" + "长" * 200 + "…"


class TestLLMFoldSummaryWriter:
    """用模型写判断类的五栏，标题与规则抽取的部分由代码写"""

    @pytest.mark.asyncio
    async def test_model_writes_the_columns_of_the_folded_turns_only(self):
        """模型看到此前的摘要与标上轮次号的被折叠轮次；五栏按固定顺序渲染，空的栏不写。"""
        llm = ScriptedLLMService(
            '{"decisions": ["认证放在 src/auth 下，因为要与会话共用"], '
            '"rejected": [], "facts": [" 现有会话类可以复用 ", " "], '
            '"preferences": ["用户要求先读现有代码"], '
            '"files": ["src/auth/session.py：复用其中的会话类"]}'
        )

        summary = await _llm_writer(llm).generate_summary(
            _folded_blocks(), first_turn_number=3, earlier_summary="## 第 1–2 轮\n此前的摘要"
        )

        assert summary == (
            "## 第 3–4 轮\n"
            "### 已定的决定及理由\n- 认证放在 src/auth 下，因为要与会话共用\n"
            "### 确认的事实与结论\n- 现有会话类可以复用\n"
            "### 用户的偏好与要求\n- 用户要求先读现有代码\n"
            "### 涉及的文件及其作用\n- src/auth/session.py：复用其中的会话类\n" + RULE_PART
        )
        [call] = llm.calls
        assert call.params.json_output is True
        system, user = (message["content"] for message in call.messages)
        assert system == get_fold_summary_prompt_text("system_prompt")
        assert user.startswith("<earlier_summary>\n## 第 1–2 轮\n此前的摘要\n</earlier_summary>")
        for line in (
            "[Turn 3]\n[User]: 查询天气",
            "[Turn 4]\n[User]: 创建认证模块",
            "- THOUGHT: 先看看已有的鉴权代码",
            '- TOOL: Read {"file_path": "src/auth/session.py"}',
            "  RESULT (error): patch rejected: context mismatch",
            "[Assistant]: 已创建 auth.py",
        ):
            assert line in user

    @pytest.mark.asyncio
    async def test_first_fold_tells_the_model_there_is_no_earlier_summary(self):
        llm = ScriptedLLMService('{"decisions": []}')

        summary = await _llm_writer(llm).generate_summary(
            [make_logical_block("q", "a")], first_turn_number=1, earlier_summary=""
        )

        assert summary == "## 第 1 轮\n### 用户消息\n- 第 1 轮：q"
        user = llm.calls[0].messages[1]["content"]
        assert user.startswith(
            "<earlier_summary>\n"
            + get_fold_summary_prompt_text("earlier_summary_empty")
            + "\n</earlier_summary>"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "llm",
        [
            None,
            ScriptedLLMService(ModelProviderError("provider down")),
            ScriptedLLMService("已定：认证放在 src/auth 下"),
            ScriptedLLMService('{"decisions": "认证放在 src/auth 下"}'),
            ScriptedLLMService('{"decision": ["认证放在 src/auth 下"]}'),
        ],
        ids=["no_model", "provider_failure", "not_json", "column_not_a_list", "unknown_column"],
    )
    async def test_without_a_usable_model_output_only_the_rule_part_is_written(self, llm):
        """模型部分写不成时注明没有模型写的部分，标题与规则抽取的部分照样写出。"""
        summary = await _llm_writer(llm).generate_summary(
            _folded_blocks(), first_turn_number=3, earlier_summary="此前的摘要"
        )

        assert summary == (
            "## 第 3–4 轮\n" + get_fold_summary_prompt_text("model_part_missing") + "\n" + RULE_PART
        )

    @pytest.mark.asyncio
    async def test_summary_timeout_from_config_writes_only_the_rule_part(self):
        """摘要调用超过配置的时限时只写规则抽取的部分，折叠不会无限等待模型。"""

        async def never_returns(messages, params):
            await asyncio.Event().wait()

        hanging_llm = Mock()
        hanging_llm.complete = AsyncMock(side_effect=never_returns)
        writer = create_fold_summary_writer(
            LLMSummaryWriterConfig(summary_timeout_seconds=0.01),
            llm_service=hanging_llm,
            llm_params=DEFAULT_LLM_PARAMS,
            transcript_config=TRANSCRIPT,
        )

        summary = await asyncio.wait_for(
            writer.generate_summary(
                [make_logical_block("测试", "响应")], first_turn_number=1, earlier_summary=""
            ),
            timeout=2,
        )

        assert summary == (
            "## 第 1 轮\n"
            + get_fold_summary_prompt_text("model_part_missing")
            + "\n### 用户消息\n- 第 1 轮：测试"
        )

    @pytest.mark.asyncio
    async def test_program_error_is_not_masked_by_the_rule_part(self):
        """程序错误不属于提供商失败，原样传播而不降级。"""
        failure = TypeError("broken prompt wiring")

        with pytest.raises(TypeError) as raised:
            await _llm_writer(ScriptedLLMService(failure)).generate_summary(
                [make_logical_block("测试", "响应")], first_turn_number=1, earlier_summary=""
            )

        assert raised.value is failure
