"""
无状态 MTP 生成器：在 LLM 服务接口之上完成 MTP 停止符、定界符检测与补齐。

生成器只负责 MTP 停止符和文本分段，不导入任何提供商 SDK：模型调用、提供商
失败的中立转换与底层响应的关闭都由注入的 ``BaseLLMService`` 完成。取消由调用
它的 asyncio task 直接传播，不在本层维护额外的控制信号。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Coroutine
from dataclasses import dataclass
from typing import Any, Literal, overload

from hivememory.core.mtp.models import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTP_STOP_SEQUENCE,
)
from hivememory.infrastructure.llm.base import BaseLLMService, LLMRequestParams

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """一次 LLM 生成的结构化结果。"""

    text: str = ""
    finish_reason: str = "stop"
    was_mtp_interrupted: bool = False
    prefix_text: str = ""
    mtp_fragment: str = ""
    # 实际发给提供商的模型标识符（如 "deepseek/deepseek-chat"），取自请求参数。
    model_used: str = ""


@dataclass
class StreamChunk:
    """流式生成发出的单个 chunk。"""

    delta: str = ""
    full_text: str = ""
    is_final: bool = False
    result: GenerationResult | None = None
    mtp_detected: bool = False


def _normalize_mtp_interrupted_text(text: str, last_open: int) -> str:
    """为被 stop sequence 截断的 MTP 文本补全右定界符。"""
    if last_open == -1:
        return text
    mtp_fragment = text[last_open:]
    if MTP_RIGHT_DELIMITER in mtp_fragment:
        return text
    return text.rstrip() + " " + MTP_RIGHT_DELIMITER


def _generation_result(text: str, finish_reason: str, model: str) -> GenerationResult:
    """自然停止且含左定界符时视为 MTP 中断，补全右定界符并切分前缀与指令片段。"""
    last_open = text.rfind(MTP_LEFT_DELIMITER)
    was_mtp = finish_reason == "stop" and last_open != -1
    if was_mtp:
        text = _normalize_mtp_interrupted_text(text, last_open)
    return GenerationResult(
        text=text,
        finish_reason=finish_reason,
        was_mtp_interrupted=was_mtp,
        prefix_text=text[:last_open] if was_mtp else text,
        mtp_fragment=text[last_open:] if was_mtp else "",
        model_used=model,
    )


class MTPGenerator:
    """无状态 MTP 生成器；唯一入口以 ``stream`` 参数选择完整生成或流式生成。"""

    def __init__(self, llm_service: BaseLLMService) -> None:
        self._llm_service = llm_service

    @overload
    def generate(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
        *,
        stream: Literal[False] = False,
    ) -> Coroutine[Any, Any, GenerationResult]: ...

    @overload
    def generate(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
        *,
        stream: Literal[True],
    ) -> AsyncGenerator[StreamChunk, None]: ...

    def generate(
        self,
        messages: list[dict[str, str]],
        params: LLMRequestParams,
        *,
        stream: bool = False,
    ) -> Coroutine[Any, Any, GenerationResult] | AsyncGenerator[StreamChunk, None]:
        """以帧的请求参数生成：``stream=False`` 返回可 await 的结果，否则返回流。

        请求总是附带 MTP 停止符，其余参数原样交给 LLM 服务；提供商失败以
        服务转换后的中立错误原样传播。
        """
        request = params.model_copy(update={"stop": (MTP_STOP_SEQUENCE,)})
        if stream:
            return self._generate_stream(messages, request)
        return self._generate_once(messages, request)

    async def _generate_once(
        self,
        messages: list[dict[str, str]],
        request: LLMRequestParams,
    ) -> GenerationResult:
        completion = await self._llm_service.complete(messages, request)
        return _generation_result(
            completion.text, completion.finish_reason or "stop", request.model
        )

    async def _generate_stream(
        self,
        messages: list[dict[str, str]],
        request: LLMRequestParams,
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式生成文本；结束、关闭或取消时关闭服务返回的流，由服务关闭底层响应。"""
        full_text = ""
        mtp_detected = False
        finish_reason = "stop"
        pending = ""
        delimiter_tail = max(0, len(MTP_LEFT_DELIMITER) - 1)

        chunks = self._llm_service.complete(messages, request, stream=True)
        try:
            async for chunk in chunks:
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                delta_content = chunk.delta
                if not delta_content:
                    continue

                full_text += delta_content
                if mtp_detected:
                    continue

                pending += delta_content
                if MTP_LEFT_DELIMITER in pending:
                    mtp_detected = True
                    index = pending.index(MTP_LEFT_DELIMITER)
                    before = pending[:index]
                    if before:
                        yield StreamChunk(
                            delta=before,
                            full_text=full_text,
                            mtp_detected=False,
                        )
                    yield StreamChunk(delta="", full_text=full_text, mtp_detected=True)
                    pending = ""
                    continue

                # 保留可能构成左定界符前缀的尾部，避免把半个定界符作为正文推出。
                if delimiter_tail == 0:
                    emit_text = pending
                    pending = ""
                elif len(pending) > delimiter_tail:
                    emit_text = pending[:-delimiter_tail]
                    pending = pending[-delimiter_tail:]
                else:
                    emit_text = ""
                if emit_text:
                    yield StreamChunk(
                        delta=emit_text,
                        full_text=full_text,
                        mtp_detected=False,
                    )
        finally:
            # 关闭失败只记录日志，不能替换正在传播的取消或生成错误。
            try:
                await chunks.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("关闭 LLM 服务流失败", exc_info=True)

        if not mtp_detected and pending:
            yield StreamChunk(delta=pending, full_text=full_text, mtp_detected=False)

        result = _generation_result(full_text, finish_reason, request.model)
        yield StreamChunk(
            delta="",
            full_text=result.text,
            is_final=True,
            result=result,
            mtp_detected=mtp_detected,
        )


__all__ = [
    "GenerationResult",
    "MTPGenerator",
    "StreamChunk",
]
