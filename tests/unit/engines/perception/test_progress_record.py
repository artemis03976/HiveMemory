"""ProgressRecordWriter 的单元测试：写进展记录一条的模型输入与输出校验。

被测对象：``engines.perception.progress_record.ProgressRecordWriter``，模型是边界外的替身。
保护的契约：
- 输入是旧进展记录、还没整理的轮次中已经被折叠的那部分的摘要，与其后的交互原文，每一轮标上它在
  话题中的轮次号；
- 提示词只有一份：第一条没有旧进展记录时，那一节写“暂无”；没有被折叠的轮次时，折叠摘要一节写“无”；
- 输出必须是带非空正文的 JSON 对象，否则抛出；模型调用超过时限时抛出，不回退。
"""

import asyncio

import pytest

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import TopicProgressEntry
from hivememory.engines.perception.progress_record import ProgressRecordWriter
from hivememory.i18n import get_progress_record_prompt_text
from hivememory.utils.json_parser import JSONParseError
from tests.helpers.llm import DEFAULT_LLM_PARAMS, CallbackLLMService, ScriptedLLMService
from tests.helpers.sessions import make_logical_block


def _writer(llm, *, timeout_seconds: float = 60) -> ProgressRecordWriter:
    return ProgressRecordWriter(
        llm,
        DEFAULT_LLM_PARAMS,
        timeout_seconds=timeout_seconds,
        transcript_config=GenerationTranscriptConfig(),
    )


def _entry(text: str, covered_until: int) -> TopicProgressEntry:
    return TopicProgressEntry(
        text=text,
        from_turn_id="t-from",
        to_turn_id="t-to",
        to_event_seq=1,
        covered_until=covered_until,
        written_at=1.0,
    )


@pytest.mark.asyncio
async def test_input_is_the_previous_record_and_the_unrecorded_turns_with_their_numbers():
    """旧进展记录逐条列出；新交互从话题中的轮次号接着编号，未完成的轮次在轮首注明。"""
    llm = ScriptedLLMService('{"entry": "  定下了发布顺序  "}')
    blocks = [
        make_logical_block("先发 staging 吗", "先发 staging"),
        make_logical_block("那就这样", "好", outcome="cancelled"),
    ]

    text = await _writer(llm).write_entry(
        [_entry("讨论了发布流程", 2)], blocks, first_turn_number=3
    )

    assert text == "定下了发布顺序"
    [call] = llm.calls
    assert call.params.json_output is True
    system, user = (message["content"] for message in call.messages)
    assert system == get_progress_record_prompt_text("system_prompt")
    assert user == (
        "[旧进展记录]\n- 讨论了发布流程\n\n[已折叠轮次的摘要]\n无\n\n[新交互记录原文]\n"
        "[Turn 3]\n[User]: 先发 staging 吗\n[Assistant]: 先发 staging\n\n"
        "[Turn 4]\n[Outcome: cancelled] This turn did not complete normally.\n"
        "[User]: 那就这样\n[Assistant]: 好"
    )


@pytest.mark.asyncio
async def test_first_entry_writes_none_for_the_previous_record():
    """还没有写过一条：提示词不变，旧进展记录一节写“暂无”。"""
    llm = ScriptedLLMService('{"entry": "开始讨论发布流程"}')

    await _writer(llm).write_entry(
        [], [make_logical_block("怎么发布", "分两步")], first_turn_number=1
    )

    system, user = (message["content"] for message in llm.calls[0].messages)
    assert system == get_progress_record_prompt_text("system_prompt")
    assert user == (
        "[旧进展记录]\n暂无\n\n[已折叠轮次的摘要]\n无\n\n"
        "[新交互记录原文]\n[Turn 1]\n[User]: 怎么发布\n[Assistant]: 分两步"
    )


@pytest.mark.asyncio
async def test_folded_part_of_the_unrecorded_turns_is_given_as_its_fold_summaries():
    """已经被折叠的轮次只给折叠记录的摘要，按折叠顺序；原文从折叠之后的轮次接着编号。"""
    llm = ScriptedLLMService('{"entry": "定下了发布顺序"}')

    await _writer(llm).write_entry(
        [_entry("讨论了发布流程", 2)],
        [make_logical_block("那就这样", "好")],
        first_turn_number=6,
        folded_summaries=["## 第 1–3 轮\n较早的摘要", "## 第 4–5 轮\n较近的摘要"],
    )

    user = llm.calls[0].messages[1]["content"]
    assert user == (
        "[旧进展记录]\n- 讨论了发布流程\n\n"
        "[已折叠轮次的摘要]\n## 第 1–3 轮\n较早的摘要\n\n## 第 4–5 轮\n较近的摘要\n\n"
        "[新交互记录原文]\n[Turn 6]\n[User]: 那就这样\n[Assistant]: 好"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "error"),
    [("这一段讨论了发布", JSONParseError), ('{"entry": "  "}', ValueError)],
    ids=["not_json", "blank_entry"],
)
async def test_unacceptable_output_raises(output, error):
    llm = ScriptedLLMService(output)

    with pytest.raises(error):
        await _writer(llm).write_entry([], [make_logical_block("q", "a")], first_turn_number=1)


@pytest.mark.asyncio
async def test_model_call_past_the_time_limit_raises():
    async def never_returns(messages, params):
        await asyncio.Event().wait()

    llm = CallbackLLMService(complete=never_returns)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            _writer(llm, timeout_seconds=0.01).write_entry(
                [], [make_logical_block("q", "a")], first_turn_number=1
            ),
            timeout=2,
        )
