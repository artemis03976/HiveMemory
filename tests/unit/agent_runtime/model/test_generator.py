"""
MTPGenerator 单元测试：LLM 服务使用继承真实接口的替身。

测试覆盖:
- 请求附带 MTP 停止符，帧的其余参数原样传给服务
- 自然停止 / MTP 中断检测与右定界符补齐、prefix_text / mtp_fragment 切分
- 流式分块：检测到左定界符后不再推出正文，未检测到时补推尾部
- 服务的中立错误、程序错误与取消原样传播
- 结束、提前关闭与取消时关闭服务返回的流
"""

import asyncio

import pytest

from hivememory.agent_runtime.model.generator import GenerationResult, MTPGenerator, StreamChunk
from hivememory.core.errors import ModelContextWindowExceededError
from hivememory.core.mtp import MTP_LEFT_DELIMITER, MTP_RIGHT_DELIMITER, MTP_STOP_SEQUENCE
from hivememory.infrastructure.llm.base import (
    BaseLLMService,
    LLMCompletion,
    LLMRequestParams,
    LLMStreamChunk,
)
from tests.helpers.llm import ScriptedLLMService

_MESSAGES = [{"role": "user", "content": "hi"}]
_PARAMS = LLMRequestParams(
    model="provider/model",
    api_key="test-key",
    temperature=0.3,
    top_p=0.9,
    max_tokens=256,
)


def _stream_of(*deltas: str, finish_reason: str = "stop") -> list[LLMStreamChunk]:
    return [
        *(LLMStreamChunk(delta=delta) for delta in deltas),
        LLMStreamChunk(finish_reason=finish_reason),
    ]


async def _generate_once(text: str, finish_reason: str = "stop") -> GenerationResult:
    llm = ScriptedLLMService(LLMCompletion(text=text, finish_reason=finish_reason))
    return await MTPGenerator(llm).generate(_MESSAGES, _PARAMS)


async def _drain(generator: MTPGenerator) -> list[StreamChunk]:
    return [chunk async for chunk in generator.generate(_MESSAGES, _PARAMS, stream=True)]


@pytest.mark.asyncio
async def test_request_adds_mtp_stop_sequence_and_keeps_frame_params():
    """帧的模型、凭证与采样参数原样交给服务，只附加 MTP 停止符。"""
    llm = ScriptedLLMService("ok")

    result = await MTPGenerator(llm).generate(_MESSAGES, _PARAMS)

    assert llm.calls[0].params == _PARAMS.model_copy(update={"stop": (MTP_STOP_SEQUENCE,)})
    assert result.model_used == "provider/model"


class TestGenerateOnce:
    """完整生成的 MTP 中断检测与文本切分。"""

    @pytest.mark.asyncio
    async def test_natural_completion_is_not_mtp(self):
        result = await _generate_once("普通回复")

        assert result == GenerationResult(
            text="普通回复",
            finish_reason="stop",
            was_mtp_interrupted=False,
            prefix_text="普通回复",
            mtp_fragment="",
            model_used="provider/model",
        )

    @pytest.mark.asyncio
    async def test_unreported_finish_reason_is_treated_as_stop(self):
        """服务未报告结束原因时按自然停止解释，含左定界符的文本仍识别为 MTP。"""
        text = f"前缀{MTP_LEFT_DELIMITER}READ|mem_doc"
        llm = ScriptedLLMService(LLMCompletion(text=text, finish_reason=None))

        result = await MTPGenerator(llm).generate(_MESSAGES, _PARAMS)

        assert result.finish_reason == "stop"
        assert result.was_mtp_interrupted is True

    @pytest.mark.asyncio
    async def test_mtp_interrupted_text_gets_right_delimiter(self):
        """stop 截断的 MTP 文本补齐右定界符，并切分前缀与指令片段。"""
        result = await _generate_once(f"前面的文本{MTP_LEFT_DELIMITER}READ|mem_doc")

        assert result.was_mtp_interrupted is True
        assert result.prefix_text == "前面的文本"
        assert result.text == f"前面的文本{MTP_LEFT_DELIMITER}READ|mem_doc {MTP_RIGHT_DELIMITER}"
        assert result.mtp_fragment == f"{MTP_LEFT_DELIMITER}READ|mem_doc {MTP_RIGHT_DELIMITER}"

    @pytest.mark.asyncio
    async def test_length_limited_text_is_not_mtp(self):
        """finish_reason=length 时即使含左定界符也不标记为 MTP。"""
        text = f"文本{MTP_LEFT_DELIMITER}something"

        result = await _generate_once(text, finish_reason="length")

        assert result.was_mtp_interrupted is False
        assert (result.prefix_text, result.mtp_fragment) == (text, "")

    @pytest.mark.asyncio
    async def test_complete_mtp_text_is_not_completed_twice(self):
        text = f"{MTP_LEFT_DELIMITER}READ|mem_doc {MTP_RIGHT_DELIMITER}"

        result = await _generate_once(text)

        assert result.was_mtp_interrupted is True
        assert (result.text, result.mtp_fragment) == (text, text)

    @pytest.mark.asyncio
    async def test_last_left_delimiter_starts_the_fragment(self):
        text = f"a{MTP_LEFT_DELIMITER}first{MTP_LEFT_DELIMITER}second"

        result = await _generate_once(text)

        assert result.prefix_text == f"a{MTP_LEFT_DELIMITER}first"
        assert result.mtp_fragment == f"{MTP_LEFT_DELIMITER}second {MTP_RIGHT_DELIMITER}"


class TestGenerateStream:
    """流式生成的分块序列。"""

    @pytest.mark.asyncio
    async def test_text_before_mtp_is_streamed_once_and_protocol_text_is_withheld(self):
        left = MTP_LEFT_DELIMITER
        llm = ScriptedLLMService(_stream_of("你好", f"世界{left}READ", "|x"))

        chunks = await _drain(MTPGenerator(llm))

        assert [(c.delta, c.mtp_detected) for c in chunks if not c.is_final] == [
            ("你好", False),
            ("世界", False),
            ("", True),
        ]
        final = chunks[-1]
        assert final.is_final is True
        assert final.result.was_mtp_interrupted is True
        assert final.result.text == f"你好世界{left}READ|x {MTP_RIGHT_DELIMITER}"
        assert llm.calls[0].stream is True

    @pytest.mark.asyncio
    async def test_natural_stream_ends_with_final_result(self):
        llm = ScriptedLLMService(_stream_of("A", "B", finish_reason="length"))

        chunks = await _drain(MTPGenerator(llm))

        assert [c.delta for c in chunks if not c.is_final] == ["A", "B"]
        assert chunks[-1].result == GenerationResult(
            text="AB",
            finish_reason="length",
            prefix_text="AB",
            model_used="provider/model",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["once", "stream"])
async def test_service_errors_propagate_unchanged(path):
    """生成器不识别也不改写错误；中立错误的分类由服务在 SDK 边界完成。"""
    failure = ModelContextWindowExceededError("Model context window exceeded.")
    generator = MTPGenerator(ScriptedLLMService(failure))

    with pytest.raises(ModelContextWindowExceededError) as raised:
        if path == "once":
            await generator.generate(_MESSAGES, _PARAMS)
        else:
            await _drain(generator)

    assert raised.value is failure


class _TrackedStreamService(BaseLLMService):
    """流在第一项后挂起，记录服务流是否被关闭；可让关闭本身失败。"""

    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.pull_blocked = asyncio.Event()
        self.closed = False
        self._close_error = close_error

    async def _complete(self, messages, params) -> LLMCompletion:
        raise AssertionError("流式测试不使用完整调用")

    def _stream(self, messages, params):
        return _TrackedStream(self)


class _TrackedStream:
    def __init__(self, service: _TrackedStreamService) -> None:
        self._service = service
        self._pulled = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> LLMStreamChunk:
        if not self._pulled:
            self._pulled = True
            return LLMStreamChunk(delta="partial text")
        self._service.pull_blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        self._service.closed = True
        if self._service._close_error is not None:
            raise self._service._close_error


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["closed_by_caller", "cancelled_during_pull"])
async def test_service_stream_is_closed_when_generation_stops_early(ending):
    service = _TrackedStreamService()
    stream = MTPGenerator(service).generate(_MESSAGES, _PARAMS, stream=True)
    assert (await anext(stream)).delta == "partial text"

    if ending == "closed_by_caller":
        await stream.aclose()
    else:
        pull = asyncio.create_task(anext(stream))
        await service.pull_blocked.wait()
        pull.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pull

    assert service.closed is True


@pytest.mark.asyncio
async def test_service_stream_close_failure_does_not_replace_cancellation():
    service = _TrackedStreamService(close_error=RuntimeError("close failed"))
    stream = MTPGenerator(service).generate(_MESSAGES, _PARAMS, stream=True)
    await anext(stream)

    pull = asyncio.create_task(anext(stream))
    await service.pull_blocked.wait()
    pull.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pull
    assert service.closed is True
