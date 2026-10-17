"""用本次解析的模型增量折叠完整轮次；每次请求先检查自身预算。"""

import asyncio
from typing import Any

import litellm

from hivememory.agent_runtime.context_budget import (
    ModelContextWindowExceededError,
    estimate_context_tokens,
)
from hivememory.agent_runtime.execution.worker import WorkerAgentService
from hivememory.agent_runtime.model_resolution import ResolvedFrameModel
from hivememory.config.alice import CompactionConfig
from hivememory.core.constants import DEFAULT_MAX_TOKENS
from hivememory.core.mtp.models import MTP_LEFT_DELIMITER, MTP_STOP_SEQUENCE
from hivememory.prompts.session_compaction import build_summary_messages
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
        while cursor < len(turns):
            batch: list[dict[str, Any]] = []
            messages: list[dict[str, str]] = []
            for turn in turns[cursor:]:
                candidate = build_summary_messages(
                    summary, [*batch, turn], language=language, max_tokens=output_limit
                )
                if estimate_context_tokens(candidate) > model.input_token_limit:
                    break
                batch.append(turn)
                messages = candidate
            if not batch:
                raise SummaryRejectedError("summary_input_too_large")
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
            cursor += len(batch)
        return summary
