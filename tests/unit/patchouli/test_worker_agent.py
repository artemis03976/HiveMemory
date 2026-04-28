"""
WorkerAgentService 单元测试

测试覆盖:
- 正常完成 / MTP 中断检测 / finish_reason 分支
- prefix_text / mtp_fragment 分割逻辑
- 异常传播 / stop sequence 注入 / kwargs 透传
"""

import pytest
from unittest.mock import Mock, patch

from hivememory.patchouli.worker_agent import WorkerAgentService
from hivememory.patchouli.config import LLMConfig
from hivememory.patchouli.mtp import MTP_LEFT_DELIMITER, MTP_STOP_SEQUENCE


def _make_config() -> LLMConfig:
    return LLMConfig(model="test-model", api_key="test-key")


def _make_response(text="hello", finish_reason="stop", has_usage=True):
    """构建 mock litellm response"""
    choice = Mock()
    choice.message.content = text
    choice.finish_reason = finish_reason
    resp = Mock()
    resp.choices = [choice]
    if has_usage:
        resp.usage = Mock(total_tokens=100)
    else:
        resp.usage = None
    return resp


@pytest.mark.asyncio
class TestWorkerAgentGenerateAsync:
    """generate_async() 方法测试"""

    def setup_method(self):
        self.config = _make_config()
        self.service = WorkerAgentService(config=self.config)

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_normal_completion(self, mock_completion):
        """正常完成，无 MTP 中断"""
        mock_completion.return_value = _make_response("普通回复", "stop")

        result = await self.service.generate_async(
            [{"role": "user", "content": "hi"}]
        )

        assert result.text == "普通回复"
        assert result.finish_reason == "stop"
        assert result.was_mtp_interrupted is False
        assert result.prefix_text == "普通回复"
        assert result.mtp_fragment == ""

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_mtp_interrupted(self, mock_completion):
        """finish_reason=stop 且文本含 ⟪ 时检测为 MTP 中断"""
        text = f"前面的文本{MTP_LEFT_DELIMITER}READ|mem_doc"
        mock_completion.return_value = _make_response(text, "stop")

        result = await self.service.generate_async(
            [{"role": "user", "content": "test"}]
        )

        assert result.was_mtp_interrupted is True
        assert result.prefix_text == "前面的文本"
        assert result.mtp_fragment == f"{MTP_LEFT_DELIMITER}READ|mem_doc"

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_length_limited_not_mtp(self, mock_completion):
        """finish_reason=length 时即使含 ⟪ 也不标记为 MTP"""
        text = f"文本{MTP_LEFT_DELIMITER}something"
        mock_completion.return_value = _make_response(text, "length")

        result = await self.service.generate_async(
            [{"role": "user", "content": "test"}]
        )

        assert result.was_mtp_interrupted is False
        assert result.prefix_text == text
        assert result.mtp_fragment == ""

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_empty_text_response(self, mock_completion):
        """LLM 返回空文本"""
        mock_completion.return_value = _make_response("", "stop")

        result = await self.service.generate_async(
            [{"role": "user", "content": "test"}]
        )

        assert result.text == ""
        assert result.was_mtp_interrupted is False

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_mtp_delimiter_at_start(self, mock_completion):
        """⟪ 在位置 0，prefix_text 为空"""
        text = f"{MTP_LEFT_DELIMITER}SEARCH|query=\"test\""
        mock_completion.return_value = _make_response(text, "stop")

        result = await self.service.generate_async(
            [{"role": "user", "content": "test"}]
        )

        assert result.was_mtp_interrupted is True
        assert result.prefix_text == ""
        assert result.mtp_fragment == text

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_multiple_delimiters_uses_last(self, mock_completion):
        """多个 ⟪ 使用 rfind 取最后一个"""
        text = f"a{MTP_LEFT_DELIMITER}first{MTP_LEFT_DELIMITER}second"
        mock_completion.return_value = _make_response(text, "stop")

        result = await self.service.generate_async(
            [{"role": "user", "content": "test"}]
        )

        assert result.was_mtp_interrupted is True
        last_pos = text.rfind(MTP_LEFT_DELIMITER)
        assert result.prefix_text == text[:last_pos]
        assert result.mtp_fragment == text[last_pos:]

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_llm_exception_propagated(self, mock_completion):
        """litellm.acompletion 抛异常时向上传播"""
        mock_completion.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            await self.service.generate_async(
                [{"role": "user", "content": "test"}]
            )

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_stop_sequence_injected(self, mock_completion):
        """验证 stop=[MTP_STOP_SEQUENCE] 被传入"""
        mock_completion.return_value = _make_response("ok", "stop")

        await self.service.generate_async([{"role": "user", "content": "test"}])

        call_kwargs = mock_completion.call_args
        assert call_kwargs[1]["stop"] == [MTP_STOP_SEQUENCE]

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_kwargs_passed_through(self, mock_completion):
        """额外 kwargs 正确传递给 litellm"""
        mock_completion.return_value = _make_response("ok", "stop")

        await self.service.generate_async(
            [{"role": "user", "content": "test"}],
            top_p=0.9,
            presence_penalty=0.5,
        )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["presence_penalty"] == 0.5


class _MockStreamResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for item in self._chunks:
            yield item


def _make_stream_chunk(delta: str = "", finish_reason=None):
    choice = Mock()
    choice.delta.content = delta
    choice.finish_reason = finish_reason
    chunk = Mock()
    chunk.choices = [choice]
    return chunk


@pytest.mark.asyncio
class TestWorkerAgentGenerateStream:
    def setup_method(self):
        self.config = _make_config()
        self.service = WorkerAgentService(config=self.config)

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_stream_no_duplicate_before_mtp(self, mock_completion):
        left = MTP_LEFT_DELIMITER
        chunks = [
            _make_stream_chunk("你好"),
            _make_stream_chunk("世界"),
            _make_stream_chunk(f"{left}READ|x", finish_reason="stop"),
        ]
        mock_completion.return_value = _MockStreamResponse(chunks)

        stream_chunks = []
        async for chunk in self.service.generate_stream([{"role": "user", "content": "hi"}]):
            stream_chunks.append(chunk)

        non_final_text = "".join(
            c.delta for c in stream_chunks if (not c.is_final and not c.mtp_detected)
        )
        assert non_final_text == "你好世界"
        assert stream_chunks[-1].is_final is True
        assert stream_chunks[-1].result is not None
        assert stream_chunks[-1].result.was_mtp_interrupted is True

    @patch("hivememory.patchouli.worker_agent.litellm.acompletion")
    async def test_stream_flushes_pending_tail_without_mtp(self, mock_completion):
        chunks = [
            _make_stream_chunk("A"),
            _make_stream_chunk("B"),
            _make_stream_chunk("", finish_reason="stop"),
        ]
        mock_completion.return_value = _MockStreamResponse(chunks)

        stream_chunks = []
        async for chunk in self.service.generate_stream([{"role": "user", "content": "hi"}]):
            stream_chunks.append(chunk)

        non_final_text = "".join(
            c.delta for c in stream_chunks if (not c.is_final and not c.mtp_detected)
        )
        assert non_final_text == "AB"
        assert stream_chunks[-1].is_final is True
        assert stream_chunks[-1].result is not None
        assert stream_chunks[-1].result.text == "AB"
