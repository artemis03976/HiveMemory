"""
Worker Agent Service - Agent Runtime 的无状态 LLM 文本生成服务。

Worker 只负责 LLM 调用、MTP stop sequence 和文本分段；取消由调用它的
asyncio task 直接传播，不在本层维护额外的控制信号。
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncGenerator
from typing import Any

import litellm

from hivememory.agent_runtime.models import GenerationResult, StreamChunk
from hivememory.core.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from hivememory.core.mtp.models import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTP_STOP_SEQUENCE,
)

logger = logging.getLogger(__name__)


class WorkerAgentService:
    """无状态 LLM 文本生成服务。"""

    def __init__(self) -> None:
        logger.info("WorkerAgentService 初始化完成（无状态）")

    def _extract_runtime_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """提取本次 LLM 调用的运行时参数。"""
        model = kwargs.pop("model", None)
        if not model:
            raise ValueError(
                "WorkerAgentService: 缺少必填参数 'model'。"
                "请通过 generation_options 显式指定，或在 ModelRegistry 中为 Agent "
                "配置对应模型。不允许静默回落到其他模型。"
            )

        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        max_tokens = kwargs.pop("max_tokens", None)
        return {
            "model": model,
            "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
            "max_tokens": DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
            "top_p": DEFAULT_TOP_P if top_p is None else top_p,
            "api_key": kwargs.pop("api_key", None),
            "api_base": kwargs.pop("api_base", None),
        }

    @staticmethod
    def _normalize_mtp_interrupted_text(text: str, last_open: int) -> str:
        """为被 stop sequence 截断的 MTP 文本补全右定界符。"""
        if last_open == -1:
            return text
        mtp_fragment = text[last_open:]
        if MTP_RIGHT_DELIMITER in mtp_fragment:
            return text
        return text.rstrip() + " " + MTP_RIGHT_DELIMITER

    async def generate_async(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> GenerationResult:
        """执行一次完整 LLM 生成。"""
        runtime_params = self._extract_runtime_params(kwargs)
        try:
            response = await litellm.acompletion(
                model=runtime_params["model"],
                messages=messages,
                api_key=runtime_params["api_key"],
                api_base=runtime_params["api_base"],
                temperature=runtime_params["temperature"],
                max_tokens=runtime_params["max_tokens"],
                stop=[MTP_STOP_SEQUENCE],
                top_p=runtime_params["top_p"],
                **kwargs,
            )
        except Exception as error:
            logger.error("LLM 异步生成失败: %s", error)
            raise

        text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"
        if hasattr(response, "usage") and response.usage:
            logger.info(
                "LLM 异步生成完成 (model=%s, tokens=%s, finish_reason=%s)",
                runtime_params["model"],
                response.usage.total_tokens,
                finish_reason,
            )

        last_open = text.rfind(MTP_LEFT_DELIMITER)
        was_mtp = finish_reason == "stop" and last_open != -1
        if was_mtp:
            text = self._normalize_mtp_interrupted_text(text, last_open)

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            was_mtp_interrupted=was_mtp,
            prefix_text=text[:last_open] if was_mtp else text,
            mtp_fragment=text[last_open:] if was_mtp else "",
            model_used=runtime_params["model"],
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式生成文本，并在结束或取消时关闭底层 async iterator。"""
        runtime_params = self._extract_runtime_params(kwargs)
        try:
            response = await litellm.acompletion(
                model=runtime_params["model"],
                messages=messages,
                api_key=runtime_params["api_key"],
                api_base=runtime_params["api_base"],
                temperature=runtime_params["temperature"],
                max_tokens=runtime_params["max_tokens"],
                top_p=runtime_params["top_p"],
                stop=[MTP_STOP_SEQUENCE],
                stream=True,
                **kwargs,
            )
        except Exception as error:
            logger.error("LLM 流式生成启动失败: %s", error)
            raise

        full_text = ""
        mtp_detected = False
        finish_reason = "stop"
        pending = ""
        delimiter_tail = max(0, len(MTP_LEFT_DELIMITER) - 1)

        try:
            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta_content = choice.delta.content or ""
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
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

            if not mtp_detected and pending:
                yield StreamChunk(delta=pending, full_text=full_text, mtp_detected=False)

            last_open = full_text.rfind(MTP_LEFT_DELIMITER)
            was_mtp = finish_reason == "stop" and last_open != -1
            if was_mtp:
                full_text = self._normalize_mtp_interrupted_text(full_text, last_open)

            result = GenerationResult(
                text=full_text,
                finish_reason=finish_reason,
                was_mtp_interrupted=was_mtp,
                prefix_text=full_text[:last_open] if was_mtp else full_text,
                mtp_fragment=full_text[last_open:] if was_mtp else "",
                model_used=runtime_params["model"],
            )
            yield StreamChunk(
                delta="",
                full_text=full_text,
                is_final=True,
                result=result,
                mtp_detected=mtp_detected,
            )
        finally:
            close = getattr(response, "aclose", None)
            if callable(close):
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result


__all__ = [
    "GenerationResult",
    "StreamChunk",
    "WorkerAgentService",
]
