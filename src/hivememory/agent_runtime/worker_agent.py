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
from contextlib import suppress
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncio
import litellm

from hivememory.agent_runtime.models import GenerationResult, StreamChunk
from hivememory.core.mtp.models import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTP_STOP_SEQUENCE,
)

logger = logging.getLogger(__name__)

# 温度/token 上限的服务级默认值，不影响"使用哪个模型"的语义。
# 实际值通常由 ModelRegistry 中的模型定义覆盖。
_DEFAULT_TEMPERATURE: float = 1.0
_DEFAULT_MAX_TOKENS: int = 32768


class WorkerAgentService:
    """
    无状态 LLM 文本生成服务

    封装 litellm.acompletion() 调用，自动注入 MTP Stop Sequence，
    并对返回结果进行 MTP 中断检测。

    所有 LLM 参数（model、api_key、api_base 等）必须在每次调用时通过
    generation_options 显式传入，不保存任何实例级配置。
    model 缺失时立即抛出 ValueError，拒绝静默降级。

    使用示例:
        >>> service = WorkerAgentService()
        >>> result = await service.generate_async(
        ...     messages,
        ...     model="deepseek/deepseek-chat",
        ...     api_key="sk-...",
        ... )
    """

    def __init__(self) -> None:
        logger.info("WorkerAgentService 初始化完成（无状态）")

    def _extract_runtime_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """从 generation_options（kwargs）中提取本次 LLM 调用的运行时参数。

        设计原则：
        - model 是必填项，缺失即抛 ValueError。不允许隐式回落到"某个"模型——
          用错模型比报错更糟糕。
        - api_key / api_base 允许为 None：litellm 会从对应环境变量读取，
          这是合法的密钥管理方式。
        - temperature / max_tokens 缺失时使用服务级默认值，不影响模型正确性。
        """
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
        # api_key / api_base 为 None 是合法状态（litellm 从环境变量读取）
        api_key = kwargs.pop("api_key", None)
        api_base = kwargs.pop("api_base", None)

        params: Dict[str, Any] = {
            "model": model,
            "temperature": _DEFAULT_TEMPERATURE if temperature is None else temperature,
            "max_tokens": _DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
            "api_key": api_key,
            "api_base": api_base,
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
        cancel_event: Optional[asyncio.Event] = None,
        **kwargs,
    ) -> GenerationResult:
        runtime_params = self._extract_runtime_params(kwargs)
        try:
            response = await self._completion_with_cancel(
                cancel_event=cancel_event,
                completion_kwargs=dict(
                    model=runtime_params["model"],
                    messages=messages,
                    api_key=runtime_params["api_key"],
                    api_base=runtime_params["api_base"],
                    temperature=runtime_params["temperature"],
                    max_tokens=runtime_params["max_tokens"],
                    stop=[MTP_STOP_SEQUENCE],
                    top_p=runtime_params.get("top_p"),
                    **kwargs,
                ),
            )
        except Exception as e:
            logger.error(f"LLM 异步生成失败: {e}")
            raise

        if response is None:
            return GenerationResult(
                text="",
                finish_reason="cancelled",
                was_mtp_interrupted=False,
                prefix_text="",
                mtp_fragment="",
                model_used=runtime_params["model"],
            )

        text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"

        if cancel_event is not None and cancel_event.is_set():
            text = ""
            finish_reason = "cancelled"

        if hasattr(response, "usage") and response.usage:
            logger.info(
                f"LLM 异步生成完成 (model={runtime_params['model']}, "
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
            model_used=runtime_params["model"],
        )

    async def _completion_with_cancel(
        self,
        *,
        cancel_event: Optional[asyncio.Event],
        completion_kwargs: Dict[str, Any],
    ) -> Any:
        completion_task = asyncio.create_task(litellm.acompletion(**completion_kwargs))
        if cancel_event is None:
            return await completion_task

        cancel_task = asyncio.create_task(cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {completion_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                completion_task.cancel()
                with suppress(asyncio.CancelledError):
                    await completion_task
                return None
            return await completion_task
        finally:
            cancel_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancel_task

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
                api_key=runtime_params["api_key"],
                api_base=runtime_params["api_base"],
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
            model_used=runtime_params["model"],
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
