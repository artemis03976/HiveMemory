"""
记忆提取器 (LLMMemoryExtractor) 单元测试

测试覆盖:
- LLM 消息格式与按语言选择的提示词
- 话题提取的草稿列表解析、“没有产出”与输出截断
- 模型调用经 LLM 接口的重试：提供商失败再请求一次（重试策略本身由接口模块的测试覆盖）
- WRITE、修订与合并返回模型原文；重试时把上一次的输出与被拒的说明接在对话之后
- 等待 LLM 期间不阻塞事件循环
"""

import asyncio
import json
import re
from datetime import UTC, datetime

import pytest

from hivememory.config.patchouli import ExtractorConfig
from hivememory.core.errors import ModelProviderError
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.engines.generation.extractor import LLMMemoryExtractor, NoOpMemoryExtractor
from hivememory.i18n import set_language
from hivememory.infrastructure.llm.base import (
    BaseLLMService,
    LLMCompletion,
    LLMRequestParams,
)
from tests.helpers.llm import DEFAULT_LLM_PARAMS, ScriptedLLMService
from tests.helpers.memory import make_memory_metadata

_DRAFT = {
    "title": "Extracted",
    "summary": "Summary",
    "tags": ["t1"],
    "memory_type": "FACT",
    "content": "Content",
    "confidence_score": 0.95,
    "has_value": True,
}
_DRAFT_JSON = json.dumps(_DRAFT)
_EXTRACTION_JSON = json.dumps(
    {"memories": [_DRAFT, {**_DRAFT, "title": "Second", "content": "Other content"}]}
)
_EDIT_SET_JSON = json.dumps(
    {"edits": [{"op": "append", "text": "Later progress"}], "changelog": "Appended progress"}
)


# CJK 标点、汉字与全角字符；英文提示词中出现任何一个，模型都可能跟着用中文写记忆。
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


@pytest.fixture(autouse=True)
def reset_i18n():
    set_language("zh")
    yield
    set_language("zh")


def _extractor(llm: BaseLLMService) -> LLMMemoryExtractor:
    return LLMMemoryExtractor(
        config=ExtractorConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )


class TestLLMMemoryExtractorPrompts:
    """提取、修订与合并发送的消息与组件请求参数。"""

    @pytest.mark.asyncio
    async def test_extract_sends_system_and_user_messages_with_component_params(self):
        llm = ScriptedLLMService(_EXTRACTION_JSON)

        await _extractor(llm).extract("User: Hi\nAssistant: Hello")

        call = llm.calls[-1]
        assert [message["role"] for message in call.messages] == ["system", "user"]
        assert call.params == DEFAULT_LLM_PARAMS
        # 话题提取要求输出草稿列表，格式说明必须是列表的 Schema。
        assert '"memories"' in call.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_language_selects_english_passive_prompts(self):
        """全局语言为英文时，被动提取的提示词全程英文，输出格式的 Schema 说明也是英文。"""
        set_language("en")
        llm = ScriptedLLMService(_EXTRACTION_JSON)

        await _extractor(llm).extract("User: Hi")

        messages = llm.calls[-1].messages
        assert "memory manager" in messages[0]["content"]
        assert "Conversation" in messages[1]["content"]
        assert _CJK.findall("".join(message["content"] for message in messages)) == []

    @pytest.mark.asyncio
    async def test_language_selects_english_write_prompts(self):
        """全局语言为英文时，WRITE 的提示词全程英文；没有标题时标题一栏写"未提供"。"""
        set_language("en")
        llm = ScriptedLLMService(_DRAFT_JSON)

        await _extractor(llm).extract_write(
            "User: Save this", content="Content to save", reason="Reuse later", title=None
        )

        messages = llm.calls[-1].messages
        assert "Active Response Mode" in messages[0]["content"]
        assert "Agent-submitted Memory Draft" in messages[1]["content"]
        assert "**Agent-proposed Title**:\n(Not provided)" in messages[1]["content"]
        assert _CJK.findall("".join(message["content"] for message in messages)) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("language", "keep_rule"),
        [
            ("zh", "Agent 拟定了标题时沿用它，不改写措辞"),
            ("en", "When the Agent proposed a title, keep it and do not reword it"),
        ],
    )
    async def test_write_prompts_carry_agent_title_and_keep_rule(self, language, keep_rule):
        """Mode B 的系统提示词带沿用规则，用户提示词带 agent 拟定的标题。"""
        set_language(language)
        llm = ScriptedLLMService(_DRAFT_JSON)

        await _extractor(llm).extract_write(
            "User: Save this",
            content="正文",
            reason=None,
            title="报表服务的每日汇总改由进程内 APScheduler 调度",
        )

        system, user = (message["content"] for message in llm.calls[-1].messages)
        assert keep_rule in system
        assert "报表服务的每日汇总改由进程内 APScheduler 调度" in user

    @pytest.mark.asyncio
    async def test_language_selects_english_update_prompts(self):
        """全局语言为英文时，UPDATE 的提示词全程英文；目标记忆带摘要与标签。"""
        set_language("en")
        llm = ScriptedLLMService(_EDIT_SET_JSON)

        await _extractor(llm).revise(
            _merge_memories()[0],
            instruction="Update it",
            new_content=None,
            transcript="",
        )

        messages = llm.calls[-1].messages
        assert "Editor Mode" in messages[0]["content"]
        assert all(
            text in messages[1]["content"]
            for text in (
                "Target Memory",
                "**Summary**: Decision",
                '**Tags**: ["design"]',
                "No new material",
                "No background conversation",
            )
        )
        assert _CJK.findall("".join(message["content"] for message in messages)) == []


class TestLLMMemoryExtractorResults:
    """模型输出到草稿的解析，以及没有产出与提供商失败的区分。"""

    @pytest.mark.asyncio
    async def test_extract_parses_model_output_into_drafts_in_order(self):
        drafts = await _extractor(ScriptedLLMService(_EXTRACTION_JSON)).extract("User: Hi")

        assert [(d.title, d.content) for d in drafts] == [
            ("Extracted", "Content"),
            ("Second", "Other content"),
        ]

    @pytest.mark.asyncio
    async def test_malformed_citations_are_dropped_without_voiding_the_output(self):
        """段引用只保留形如 2.3 的字符串；格式不对的引用只丢弃引用，不让草稿或整份输出作废。"""
        output = json.dumps(
            {
                "memories": [
                    {**_DRAFT, "source_segments": ["1.2", " 2.3 ", "Segment 2.4", 3.1, "2"]},
                    {**_DRAFT, "title": "Second", "source_segments": "1.1"},
                ]
            }
        )

        drafts = await _extractor(ScriptedLLMService(output)).extract("User: Hi")

        assert [(d.title, d.source_segments) for d in drafts] == [
            ("Extracted", ["1.2", "2.3"]),
            ("Second", []),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "output",
        ["", "not json", _DRAFT_JSON],
        ids=["empty", "unparseable", "bare-draft"],
    )
    async def test_empty_or_unparseable_output_is_no_output(self, output):
        """模型给出空文本、无法解析的文本或缺少草稿列表时没有产出，而不是失败。"""
        assert await _extractor(ScriptedLLMService(output)).extract("User: Hi") == []

    @pytest.mark.asyncio
    async def test_truncated_output_fails_without_retry(self):
        """截断的列表不能记为没有产出，否则整段话题不留记忆而任务成功；重发会再次截断。"""
        llm = ScriptedLLMService(
            LLMCompletion(text='{"memories": [' + _DRAFT_JSON, finish_reason="length"),
            _EXTRACTION_JSON,
        )

        with pytest.raises(ModelProviderError):
            await _extractor(llm).extract("User: Hi")

        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_provider_failure_is_retried_once(self):
        llm = ScriptedLLMService(ModelProviderError("transient"), _EXTRACTION_JSON)

        drafts = await _extractor(llm).extract("User: Hi")

        assert [d.title for d in drafts] == ["Extracted", "Second"]
        assert len(llm.calls) == 2


class _GatedLLMService(BaseLLMService):
    """在放行前一直挂起的完整调用，用于观察等待期间事件循环是否推进。"""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def _complete(
        self, messages: list[dict[str, str]], params: LLMRequestParams
    ) -> LLMCompletion:
        self.started.set()
        await self.release.wait()
        return LLMCompletion(text=_EXTRACTION_JSON)

    def _stream(self, messages, params):
        raise AssertionError("提取器不使用流式调用")


@pytest.mark.asyncio
async def test_waiting_for_llm_lets_the_event_loop_run_other_tasks():
    llm = _GatedLLMService()
    extraction = asyncio.create_task(_extractor(llm).extract("User: Hi"))

    async with asyncio.timeout(5):
        await llm.started.wait()
    # 提取仍在等待模型时，测试协程照常推进并负责放行。
    assert not extraction.done()
    llm.release.set()
    drafts = await extraction

    assert drafts[0].title == "Extracted"


def _merge_memories():
    """提供更新时间与内容都不同的两条正式记忆。"""
    retained = MemoryAtom(
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title="Design", summary="Decision", tags=["design"], memory_type=MemoryType.FACT
        ),
        payload=PayloadLayer(content="Earlier decision"),
    )
    retained.meta.updated_at = datetime(2026, 10, 1, tzinfo=UTC)
    absorbed = retained.model_copy(deep=True)
    absorbed.index.title = "Progress"
    absorbed.payload.content = "Later progress"
    absorbed.meta.updated_at = datetime(2026, 10, 2, tzinfo=UTC)
    return retained, absorbed


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["zh", "en"])
async def test_merge_sends_both_memories_and_newer_marker(language):
    """两种语言都携带两条记忆的完整内容、较新标记与 Librarian 请求参数。"""
    set_language(language)
    llm = ScriptedLLMService(_EDIT_SET_JSON)
    result = await _extractor(llm).merge(*_merge_memories())
    call = llm.calls[0]
    assert call.params == DEFAULT_LLM_PARAMS
    assert result == _EDIT_SET_JSON
    user_prompt = call.messages[1]["content"]
    assert all(
        text in user_prompt
        for text in (
            "Design",
            "Decision",
            "design",
            "FACT",
            "Earlier decision",
            "Progress",
            "Later progress",
        )
    )
    assert ("较新的是：N" if language == "zh" else "newer content update is: N") in user_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ask", "wording"),
    [
        (
            lambda extractor, retry: extractor.extract_write(
                "User: Save this", content="正文", reason=None, title=None, retry=retry
            ),
            "记忆草稿",
        ),
        (lambda extractor, retry: extractor.merge(*_merge_memories(), retry=retry), "编辑集"),
    ],
    ids=["write-draft", "merge-edit-set"],
)
async def test_retry_appends_the_previous_output_and_why_it_was_rejected(ask, wording):
    """重试时原对话不变，其后接上一次的输出与被拒的说明；说明按输出是草稿还是编辑集取各自的写法。"""
    reason = "the output is not a JSON object"
    llm = ScriptedLLMService("first output", "second output")
    extractor = _extractor(llm)

    await ask(extractor, None)
    await ask(extractor, ("previous output", reason))

    first, retry = llm.calls
    assert retry.messages[:2] == first.messages
    assert [(m["role"], m["content"]) for m in retry.messages[2:3]] == [
        ("assistant", "previous output")
    ]
    request = retry.messages[3]
    assert (request["role"], reason in request["content"], wording in request["content"]) == (
        "user",
        True,
        True,
    )


@pytest.mark.asyncio
async def test_noop_extractor_has_no_revision_or_merge():
    retained, absorbed = _merge_memories()
    noop = NoOpMemoryExtractor()

    assert (
        await noop.merge(retained, absorbed),
        await noop.revise(retained, instruction="Update", new_content=None, transcript=""),
    ) == (None, None)
