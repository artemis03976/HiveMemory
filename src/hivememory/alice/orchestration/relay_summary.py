"""用本次解析的模型增量折叠完整轮次；每次请求先检查自身预算。"""

import asyncio
from typing import Any

import litellm

from hivememory.agent_runtime.context_budget import (
    MESSAGE_OVERHEAD_TOKENS,
    ModelContextWindowExceededError,
    estimate_context_tokens,
)
from hivememory.agent_runtime.execution.worker import WorkerAgentService
from hivememory.agent_runtime.model_resolution import ResolvedFrameModel
from hivememory.config.alice import CompactionConfig
from hivememory.core.constants import DEFAULT_MAX_TOKENS
from hivememory.core.mtp.models import MTP_LEFT_DELIMITER, MTP_STOP_SEQUENCE
from hivememory.prompts.session_compaction import SummaryRequest, encode_summary_turn
from hivememory.utils.token_estimator import TokenEstimator


class SummaryRejectedError(Exception):
    """可保守降级的摘要失败，reason 是观测用稳定原因，不携带模型正文。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RelaySummarizer:
    """摘要只生成文本，不创建 frame、执行工具或写入长期记忆。"""

    def __init__(self, worker: WorkerAgentService) -> None:
        self._worker = worker

    async def fold(
        self,
        previous_summary: str,
        turns: list[dict[str, Any]],
        *,
        model: ResolvedFrameModel,
        config: CompactionConfig,
        language: str,
    ) -> str:
        """局部结果只有全部折叠成功后才返回；取消与程序错误直接传播。"""
        summary = previous_summary
        cursor = 0
        output_limit = min(
            config.summary_max_tokens,
            model.generation_options.get("max_tokens") or DEFAULT_MAX_TOKENS,
        )
        options = {**model.generation_options, "max_tokens": output_limit}
        # 每轮只编码、计数一次；候选批次的估算按字符数累加，开销随轮数线性增长。
        encoded = [encode_summary_turn(turn) for turn in turns]
        turn_counts = [_char_counts(text) for text in encoded]
        separator_counts = _char_counts(SummaryRequest.separator)
        while cursor < len(turns):
            request = SummaryRequest(summary, language=language, max_tokens=output_limit)
            instruction_tokens = estimate_context_tokens([{"content": request.instruction}])
            cjk, other = _char_counts(request.head + request.tail)
            end = cursor
            while end < len(turns):
                turn_cjk, turn_other = turn_counts[end]
                if end > cursor:
                    turn_cjk += separator_counts[0]
                    turn_other += separator_counts[1]
                request_tokens = (
                    instruction_tokens
                    + TokenEstimator.estimate_from_counts(cjk + turn_cjk, other + turn_other)
                    + MESSAGE_OVERHEAD_TOKENS
                )
                if request_tokens > model.input_token_limit:
                    break
                cjk, other, end = cjk + turn_cjk, other + turn_other, end + 1
            if end == cursor:
                raise SummaryRejectedError("summary_input_too_large")
            messages = request.messages(encoded[cursor:end])
            try:
                async with asyncio.timeout(config.summary_timeout_seconds):
                    result = await self._worker.generate_async(messages, **options)
            except TimeoutError as error:
                raise SummaryRejectedError("summary_timeout") from error
            except ModelContextWindowExceededError as error:
                raise SummaryRejectedError("summary_model_context_exceeded") from error
            except (
                litellm.APIError,
                litellm.OpenAIError,
                litellm.BadGatewayError,
                litellm.APIConnectionError,
                litellm.APIResponseValidationError,
                litellm.BadRequestError,
                litellm.AuthenticationError,
                litellm.PermissionDeniedError,
                litellm.NotFoundError,
                litellm.RateLimitError,
                litellm.Timeout,
                litellm.UnprocessableEntityError,
                litellm.InternalServerError,
                litellm.ServiceUnavailableError,
                litellm.BudgetExceededError,
            ) as error:
                raise SummaryRejectedError("summary_provider_error") from error
            text = result.text.strip()
            if not text:
                raise SummaryRejectedError("summary_empty")
            if result.finish_reason in {"length", "content_filter"}:
                raise SummaryRejectedError(f"summary_{result.finish_reason}")
            if (
                result.was_mtp_interrupted
                or MTP_LEFT_DELIMITER in text
                or MTP_STOP_SEQUENCE in text
            ):
                raise SummaryRejectedError("summary_contains_mtp")
            if TokenEstimator.estimate(text) > output_limit:
                raise SummaryRejectedError("summary_output_too_large")
            summary = text
            cursor = end
        return summary


def _char_counts(text: str) -> tuple[int, int]:
    """返回 (CJK 字符数, 其他字符数)，与整段估算使用同一分类。"""
    cjk = TokenEstimator.count_cjk_chars(text)
    return cjk, len(text) - cjk
