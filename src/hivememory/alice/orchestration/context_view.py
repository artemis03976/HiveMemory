"""Alice 的会话派生视图：固定减重边界，只在触发压缩时向后推进。"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from hivememory.agent_runtime.context_budget import estimate_context_tokens
from hivememory.config.alice import CompactionConfig
from hivememory.core.protocol.models import AgentRunContext, SessionCompactionCheckpoint
from hivememory.i18n import resolve_language
from hivememory.prompts.transcript import HistoryTranscriptBuilder


class CompactionFailedError(Exception):
    """历史压缩后仍无法放入本次模型的输入预算。"""


@dataclass(frozen=True)
class CompactionResult:
    """一次压缩的消息、检查点与统计，不包含任何会话原文写入。"""

    messages: list[dict[str, str]]
    checkpoint: SessionCompactionCheckpoint
    tokens_before: int
    tokens_after: int
    covered_turns: int


class SessionContextView:
    """单次 run 的历史视图；关闭压缩时完整重放，始终保留原始轮次。"""

    def __init__(
        self,
        *,
        context: AgentRunContext,
        fixed_messages: list[dict[str, str]],
        checkpoint: SessionCompactionCheckpoint | None,
        input_token_limit: int,
        config: CompactionConfig,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._context = context
        self._fixed_messages = fixed_messages
        self._input_token_limit = input_token_limit
        self._config = config
        self._clock = clock
        self._language = resolve_language(profile_language=context.agent_profile.language).value
        self._builder = HistoryTranscriptBuilder()
        self._reduced_count = 0
        # 无法解释的格式必须回到原文，不能把未知摘要当成空历史。
        if (
            config.enabled
            and checkpoint is not None
            and checkpoint.producer == "alice"
            and checkpoint.format_version == 1
            and checkpoint.summarized_through is None
            and not checkpoint.summary
        ):
            self._reduced_count = next(
                (
                    index + 1
                    for index, turn in enumerate(context.session_history)
                    if turn.interaction_id == checkpoint.reduced_through
                ),
                0,
            )
        self.messages = self._render(self._reduced_count)
        self.estimated_tokens = estimate_context_tokens(self.messages)

    def should_compact(self, *, force: bool = False) -> bool:
        """内部强制参数跳过阈值；空历史或关闭压缩时无需生成检查点。"""
        return bool(
            self._config.enabled
            and self._context.session_history
            and (
                force
                or self.estimated_tokens > self._input_token_limit * self._config.trigger_ratio
            )
        )

    def compact(self) -> CompactionResult:
        """推进到近期原文预算内，以完整轮次为边界，不发出模型请求。"""
        turns = self._context.session_history
        recent_budget = self._input_token_limit * self._config.recent_raw_ratio
        # 从最近一轮倒序选取能放下的完整后缀，已有边界只前进。
        recent_tokens = 0
        boundary = len(turns)
        for index in range(len(turns) - 1, self._reduced_count - 1, -1):
            raw = self._builder.build_messages(
                [turns[index]],
                current_agent_id=self._context.labels.agent_id,
                language=self._language,
            )
            turn_tokens = estimate_context_tokens(raw)
            if recent_tokens + turn_tokens > recent_budget:
                break
            recent_tokens += turn_tokens
            boundary = index
        messages = self._render(boundary)
        checkpoint = SessionCompactionCheckpoint(
            producer="alice",
            format_version=1,
            reduced_through=turns[boundary - 1].interaction_id if boundary else None,
            created_at=self._clock(),
        )
        return CompactionResult(
            messages=messages,
            checkpoint=checkpoint,
            tokens_before=self.estimated_tokens,
            tokens_after=estimate_context_tokens(messages),
            covered_turns=boundary,
        )

    def _render(self, reduced_count: int) -> list[dict[str, str]]:
        """在系统提示词和本轮用户输入之间重放历史，原始载荷保持不变。"""
        turns = self._context.session_history
        history = self._builder.build_messages(
            turns,
            current_agent_id=self._context.labels.agent_id,
            language=self._language,
            reduced_through=turns[reduced_count - 1].interaction_id if reduced_count else None,
            tool_result_excerpt_chars=self._config.tool_result_excerpt_chars,
        )
        return [*self._fixed_messages[:-1], *history, *self._fixed_messages[-1:]]
