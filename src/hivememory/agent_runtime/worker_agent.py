"""
Worker Agent Service - 无状态 LLM 文本生成服务

定位：Alice 子系统的生成引擎，纯粹的文本生成器。
职责：
    - 封装 LLM API 调用 (via litellm)
    - 使用 MTP Stop Sequence (⟫) 实现生成中断
    - 检测 MTP 指令并返回结构化结果
    - 不持有任何业务状态（由 AliceRuntime / AgentRuntime 调度）

架构定位：
    AgentRuntime 持有 WorkerAgentService，并由 AgentLoopExecutor 调用。
    WorkerAgentService 不知道 MTP 协议的具体语义，只负责检测 ⟪ 定界符。

    AliceSystem
    ├── AliceRuntime
    │   ├── KoakumaRuntime
    │   ├── AgentRuntime
    │   │   ├── AgentLoopExecutor
    │   │   └── WorkerAgentService (LLM Engine)  ← 本模块
    └── AliceService

对应设计文档: MemoryToolProtocol.md Section 3.1 & 6.4
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncio
import litellm

from hivememory.system.config import LLMConfig
from hivememory.agent_runtime.models import GenerationResult, StreamChunk
from hivememory.core.mtp.models import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTP_STOP_SEQUENCE,
)

logger = logging.getLogger(__name__)


class WorkerAgentService:
    """
    无状态 LLM 文本生成服务

    封装 litellm.acompletion() 调用，自动注入 MTP Stop Sequence，
    并对返回结果进行 MTP 中断检测。

    使用示例:
        >>> from hivememory.agent_runtime.worker_agent import WorkerAgentService
        >>> from hivememory.system.config import LLMConfig
        >>>
        >>> service = WorkerAgentService(config=LLMConfig(model="gpt-4o"))
        >>> result = await service.generate_async([{"role": "user", "content": "Hello"}])
        >>> if result.was_mtp_interrupted:
        ...     print(f"MTP detected: {result.mtp_fragment}")
    """

    def __init__(self, config: LLMConfig):
        """
        Args:
            config: LLM 配置 (model, api_key, api_base, temperature, max_tokens)
        """
        self._config = config
        logger.info(
            f"WorkerAgentService 初始化完成 (model={config.model})"
        )

    def _extract_runtime_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        model = kwargs.pop("model", None)
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        max_tokens = kwargs.pop("max_tokens", None)

        params: Dict[str, Any] = {
            "model": model or self._config.model,
            "temperature": self._config.temperature if temperature is None else temperature,
            "max_tokens": self._config.max_tokens if max_tokens is None else max_tokens,
        }
        if top_p is not None:
            params["top_p"] = top_p
        return params

    @staticmethod
    def _normalize_mtp_interrupted_text(text: str, last_open: int) -> str:
        """Normalize stop-sequence-truncated MTP text before downstream parsing."""
        if last_open == -1:
            return text
        mtp_fragment = text[last_open:]
        if MTP_RIGHT_DELIMITER in mtp_fragment:
            return text
        return text.rstrip() + " " + MTP_RIGHT_DELIMITER

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> GenerationResult:
        runtime_params = self._extract_runtime_params(kwargs)
        try:
            response = await litellm.acompletion(
                model=runtime_params["model"],
                messages=messages,
                api_key=self._config.api_key,
                api_base=self._config.api_base,
                temperature=runtime_params["temperature"],
                max_tokens=runtime_params["max_tokens"],
                stop=[MTP_STOP_SEQUENCE],
                top_p=runtime_params.get("top_p"),
                **kwargs,
            )
        except Exception as e:
            logger.error(f"LLM 异步生成失败: {e}")
            raise

        text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"

        if hasattr(response, "usage") and response.usage:
            logger.info(
                f"LLM 异步生成完成 (model={self._config.model}, "
                f"tokens={response.usage.total_tokens}, "
                f"finish_reason={finish_reason})"
            )

        last_open = text.rfind(MTP_LEFT_DELIMITER)
        was_mtp = finish_reason == "stop" and last_open != -1

        if was_mtp:
            text = self._normalize_mtp_interrupted_text(text, last_open)
            logger.info(
                f"MTP 中断检测: ⟪ 位于 offset={last_open}, "
                f"prefix_len={last_open}, fragment_len={len(text) - last_open}"
            )

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            was_mtp_interrupted=was_mtp,
            prefix_text=text[:last_open] if was_mtp else text,
            mtp_fragment=text[last_open:] if was_mtp else "",
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        cancel_event: Optional[asyncio.Event] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        流式 LLM 生成，逐 chunk yield

        使用 litellm.acompletion(stream=True)，实时检测 MTP 定界符 ⟪。
        - 检测到 ⟪ 之前：每个 chunk 作为 delta yield (mtp_detected=False)
        - 检测到 ⟪ 之后：继续缓冲但不 yield delta (mtp_detected=True)
        - 流结束时：yield is_final=True 的 StreamChunk，携带完整 GenerationResult

        Args:
            messages: OpenAI 格式的消息列表
            **kwargs: 传递给 litellm 的额外参数

        Yields:
            StreamChunk: 流式 chunk
        """
        runtime_params = self._extract_runtime_params(kwargs)
        try:
            response = await litellm.acompletion(
                model=runtime_params["model"],
                messages=messages,
                api_key=self._config.api_key,
                api_base=self._config.api_base,
                temperature=runtime_params["temperature"],
                max_tokens=runtime_params["max_tokens"],
                top_p=runtime_params.get("top_p"),
                stop=[MTP_STOP_SEQUENCE],
                stream=True,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"LLM 流式生成启动失败: {e}")
            raise

        full_text = ""
        mtp_detected = False
        finish_reason = "stop"
        # 缓冲区：用于处理 ⟪ 可能跨 chunk 边界的情况
        pending = ""
        delimiter_tail = max(0, len(MTP_LEFT_DELIMITER) - 1)

        async for chunk in response:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("LLM 流式生成被用户取消")
                finish_reason = "cancelled"
                break

            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue

            delta_content = choice.delta.content or ""
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if not delta_content:
                continue

            full_text += delta_content

            if not mtp_detected:
                pending += delta_content
                if MTP_LEFT_DELIMITER in pending:
                    mtp_detected = True
                    idx = pending.index(MTP_LEFT_DELIMITER)
                    # ⟪ 之前的文本作为最后一个正常 delta 推送
                    before = pending[:idx]
                    if before:
                        yield StreamChunk(delta=before, full_text=full_text, mtp_detected=False)
                    # ⟪ 及之后的内容不推送，标记 mtp_detected
                    yield StreamChunk(delta="", full_text=full_text, mtp_detected=True)
                    pending = ""
                else:
                    if delimiter_tail == 0:
                        emit_text = pending
                        pending = ""
                    elif len(pending) > delimiter_tail:
                        emit_text = pending[:-delimiter_tail]
                        pending = pending[-delimiter_tail:]
                    else:
                        emit_text = ""
                    if emit_text:
                        yield StreamChunk(delta=emit_text, full_text=full_text, mtp_detected=False)
            # mtp_detected=True 后不再 yield 中间 chunk，静默缓冲

        if not mtp_detected and pending:
            yield StreamChunk(delta=pending, full_text=full_text, mtp_detected=False)

        # 流结束，构建最终 GenerationResult
        last_open = full_text.rfind(MTP_LEFT_DELIMITER)
        was_mtp = finish_reason == "stop" and last_open != -1

        if was_mtp:
            full_text = self._normalize_mtp_interrupted_text(full_text, last_open)
            logger.info(
                f"流式 MTP 中断检测: ⟪ 位于 offset={last_open}, "
                f"prefix_len={last_open}, fragment_len={len(full_text) - last_open}"
            )

        result = GenerationResult(
            text=full_text,
            finish_reason=finish_reason,
            was_mtp_interrupted=was_mtp,
            prefix_text=full_text[:last_open] if was_mtp else full_text,
            mtp_fragment=full_text[last_open:] if was_mtp else "",
        )

        yield StreamChunk(
            delta="",
            full_text=full_text,
            is_final=True,
            result=result,
            mtp_detected=mtp_detected,
        )


__all__ = [
    "GenerationResult",
    "StreamChunk",
    "WorkerAgentService",
]
