"""
记忆提取器 (LLMMemoryExtractor) 单元测试

测试覆盖:
- LLM 消息格式与按语言选择的提示词
- 提取结果解析与“没有产出”
- 只对提供商失败重试，耗尽后显式失败
- 等待 LLM 期间不阻塞事件循环
"""

import asyncio
import json
from datetime import UTC, datetime

import pytest

from hivememory.config.patchouli import ExtractorConfig
from hivememory.core.errors import ModelContextWindowExceededError, ModelProviderError
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.core.models.memory import MEMORY_SUMMARY_MAX_LENGTH, MEMORY_TITLE_MAX_LENGTH
from hivememory.engines.generation.extractor import LLMMemoryExtractor, NoOpMemoryExtractor
from hivememory.i18n import set_default_language
from hivememory.infrastructure.llm.base import (
    BaseLLMService,
    LLMCompletion,
    LLMRequestParams,
)
from tests.helpers.llm import DEFAULT_LLM_PARAMS, ScriptedLLMService
from tests.helpers.memory import make_memory_metadata

_DRAFT_JSON = json.dumps(
    {
        "title": "Extracted",
        "summary": "Summary",
        "tags": ["t1"],
        "memory_type": "FACT",
        "content": "Content",
        "confidence_score": 0.95,
        "has_value": True,
    }
)


@pytest.fixture(autouse=True)
def reset_i18n():
    set_default_language("zh")
    yield
    set_default_language("zh")


def _extractor(llm: BaseLLMService) -> LLMMemoryExtractor:
    return LLMMemoryExtractor(
        config=ExtractorConfig(), llm_service=llm, llm_params=DEFAULT_LLM_PARAMS
    )


class TestLLMMemoryExtractorPrompts:
    """提取、修订与合并发送的消息与组件请求参数。"""

    @pytest.mark.asyncio
    async def test_extract_sends_system_and_user_messages_with_component_params(self):
        llm = ScriptedLLMService(_DRAFT_JSON)

        await _extractor(llm).extract("User: Hi\nAssistant: Hello", {})

        call = llm.calls[-1]
        assert [message["role"] for message in call.messages] == ["system", "user"]
        assert call.params == DEFAULT_LLM_PARAMS

    @pytest.mark.asyncio
    async def test_default_language_selects_english_passive_prompts(self):
        """测试全局语言驱动被动提取模板"""
        set_default_language("en")
        llm = ScriptedLLMService(_DRAFT_JSON)

        await _extractor(llm).extract("User: Hi", {})

        messages = llm.calls[-1].messages
        assert "memory manager" in messages[0]["content"]
        assert "Conversation" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_default_language_selects_english_write_prompts(self):
        """测试全局语言驱动 WRITE 模板"""
        set_default_language("en")
        llm = ScriptedLLMService(_DRAFT_JSON)

        await _extractor(llm).extract(
            "User: Save this",
            {
                "mode": "write",
                "write_content": "Content to save",
            },
        )

        messages = llm.calls[-1].messages
        assert "Active Response Mode" in messages[0]["content"]
        assert "Agent-submitted Memory Draft" in messages[1]["content"]
        assert "(Not provided)" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_default_language_selects_english_update_prompts(self):
        """测试全局语言驱动 UPDATE 模板"""
        set_default_language("en")
        llm = ScriptedLLMService(
            json.dumps({"new_content": "New content", "changelog": "Updated content"})
        )

        await _extractor(llm).revise(
            old_content="Old content",
            metadata={
                "instruction": "Update it",
                "memory_title": "Title",
                "memory_alias": "alias",
            },
        )

        messages = llm.calls[-1].messages
        assert "Editor Mode" in messages[0]["content"]
        assert "Target Memory" in messages[1]["content"]
        assert "No new material" in messages[1]["content"]
        assert "No background conversation" in messages[1]["content"]


class TestLLMMemoryExtractorResults:
    """模型输出到草稿的解析，以及没有产出与提供商失败的区分。"""

    @pytest.mark.asyncio
    async def test_extract_parses_model_output_into_draft(self):
        draft = await _extractor(ScriptedLLMService(_DRAFT_JSON)).extract("User: Hi", {})

        assert draft is not None
        assert (draft.title, draft.content) == ("Extracted", "Content")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("output", ["", "not json"], ids=["empty", "unparseable"])
    async def test_empty_or_unparseable_output_is_no_output(self, output):
        """模型给出空文本或无法解析的文本时没有产出，而不是失败。"""
        assert await _extractor(ScriptedLLMService(output)).extract("User: Hi", {}) is None

    @pytest.mark.asyncio
    async def test_provider_failure_is_retried_once(self):
        llm = ScriptedLLMService(ModelProviderError("transient"), _DRAFT_JSON)

        draft = await _extractor(llm).extract("User: Hi", {})

        assert draft is not None
        assert draft.title == "Extracted"
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_exhausted_provider_retries_fail_explicitly(self):
        """重试耗尽后抛出最后一次的提供商失败，不被记录为没有产出。"""
        last_failure = ModelProviderError("still failing")
        llm = ScriptedLLMService(ModelProviderError("transient"), last_failure)

        with pytest.raises(ModelProviderError) as raised:
            await _extractor(llm).merge(*_merge_memories())

        assert raised.value is last_failure
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [ModelContextWindowExceededError("too long"), TypeError("broken prompt wiring")],
        ids=["context-window", "program-error"],
    )
    async def test_non_retryable_failures_propagate_after_one_attempt(self, failure):
        """超窗时同一请求必然再次失败，程序错误不属于提供商失败；两者都不重试。"""
        llm = ScriptedLLMService(failure, _DRAFT_JSON)

        with pytest.raises(type(failure)) as raised:
            await _extractor(llm).extract("User: Hi", {})

        assert raised.value is failure
        assert len(llm.calls) == 1


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
        return LLMCompletion(text=_DRAFT_JSON)

    def _stream(self, messages, params):
        raise AssertionError("提取器不使用流式调用")


@pytest.mark.asyncio
async def test_waiting_for_llm_lets_the_event_loop_run_other_tasks():
    llm = _GatedLLMService()
    extraction = asyncio.create_task(_extractor(llm).extract("User: Hi", {}))

    async with asyncio.timeout(5):
        await llm.started.wait()
    # 提取仍在等待模型时，测试协程照常推进并负责放行。
    assert not extraction.done()
    llm.release.set()
    draft = await extraction

    assert draft is not None
    assert draft.title == "Extracted"


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


def _merge_json(**changes):
    fields = dict(
        title="Merged",
        summary="Decision and progress",
        tags=["work"],
        memory_type="FACT",
        content="Both pieces",
        changelog="Preserved both stages",
    )
    fields.update(changes)
    return json.dumps(fields)


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["zh", "en"])
async def test_merge_sends_both_memories_and_newer_marker(language):
    """两种语言都携带完整内容、较新标记与 Librarian 请求参数。"""
    set_default_language(language)
    llm = ScriptedLLMService(_merge_json())
    result = await _extractor(llm).merge(*_merge_memories())
    call = llm.calls[0]
    assert call.params == DEFAULT_LLM_PARAMS
    assert result.content == "Both pieces"
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
async def test_merge_parses_and_truncates_model_output_after_provider_retry():
    """合并重用提供商重试，输出在进入索引前收敛长度。"""
    llm = ScriptedLLMService(
        ModelProviderError("transient"),
        _merge_json(
            title="题" * (MEMORY_TITLE_MAX_LENGTH + 50),
            summary="摘" * (MEMORY_SUMMARY_MAX_LENGTH + 50),
        ),
    )
    result = await _extractor(llm).merge(*_merge_memories())
    assert (result.title, result.summary, result.tags, result.memory_type, result.changelog) == (
        "题" * MEMORY_TITLE_MAX_LENGTH,
        "摘" * MEMORY_SUMMARY_MAX_LENGTH,
        ["work"],
        "FACT",
        "Preserved both stages",
    )
    assert len(llm.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    ["", "not json", '{"content":"partial"}', _merge_json(title="", content="")],
    ids=["empty", "unparseable", "missing-fields", "blank-title-and-body"],
)
async def test_merge_invalid_output_returns_none_without_retry(output):
    """没有完整合并内容时不能用追加或原始材料替代。"""
    llm = ScriptedLLMService(output)
    assert await _extractor(llm).merge(*_merge_memories()) is None
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_noop_merge_has_no_output():
    assert await NoOpMemoryExtractor().merge(*_merge_memories()) is None
