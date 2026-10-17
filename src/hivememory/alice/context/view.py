"""Alice 的会话派生视图：按完整轮次推进减重与摘要覆盖，不修改历史原文。"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from hivememory.agent_runtime.context_budget import InputTooLargeError, estimate_context_tokens
from hivememory.agent_runtime.model_resolution import ResolvedFrameModel
from hivememory.alice.context.references import extend_references
from hivememory.alice.context.summarizer import RelaySummarizer, SummaryRejectedError
from hivememory.config.alice import CompactionConfig
from hivememory.core.protocol.models import AgentRunContext, SessionCompactionCheckpoint
from hivememory.i18n import resolve_language
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.prompts.transcript import HistoryTranscriptBuilder


class CompactionFailedError(Exception):
    """历史压缩后仍无法放入本次模型的输入预算。"""


@dataclass(frozen=True)
class CompactionResult:
    """只有完整压缩结束才交回检查点；摘要失败可保留已完成的减重。"""

    messages: list[dict[str, str]]
    checkpoint: SessionCompactionCheckpoint
    tokens_before: int
    tokens_after: int
    covered_turns: int
    summarized_turns: int = 0
    layer: str = "reduction"
    summary_status: str = "not_needed"
    reason: str | None = None
    error: InputTooLargeError | CompactionFailedError | None = None


class SessionContextView:
    """单次 run 的历史视图；固定部分、摘要段落和完整后缀使用同一组装规则。"""

    def __init__(
        self,
        *,
        context: AgentRunContext,
        prompt_assembler: AgentPromptAssembler,
        checkpoint: SessionCompactionCheckpoint | None,
        input_token_limit: int,
        config: CompactionConfig,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._context = context
        self._assembler = prompt_assembler
        self._input_token_limit = input_token_limit
        self._config = config
        self._clock = clock
        self._language = resolve_language(profile_language=context.agent_profile.language).value
        self._builder = HistoryTranscriptBuilder()
        self._positions = {
            turn.interaction_id: index + 1 for index, turn in enumerate(context.session_history)
        }
        self._checkpoint = self._usable_checkpoint(checkpoint) if config.enabled else None
        self._reduced_count = (
            self._position(self._checkpoint.reduced_through) if self._checkpoint else 0
        )
        self._summarized_count = (
            self._position(self._checkpoint.summarized_through) if self._checkpoint else 0
        )
        self._fixed_messages = self._fixed()
        if estimate_context_tokens(self._fixed_messages) > input_token_limit:
            raise InputTooLargeError("Fixed input exceeds the model input budget.")
        self.messages = self._render(self._reduced_count, self._checkpoint)
        self.estimated_tokens = estimate_context_tokens(self.messages)
        if not config.enabled and self.estimated_tokens > input_token_limit:
            # 关闭压缩时超限的历史不会自行缩小；按历史放不下结束，而不是在首次生成前
            # 被轮内守卫拦截、误报为本轮工具结果超出预算。
            raise CompactionFailedError(
                "History exceeds the input budget with compaction disabled."
            )

    def _position(self, cursor: str | None) -> int:
        return self._positions.get(cursor, 0) if cursor is not None else 0

    def _usable_checkpoint(
        self, checkpoint: SessionCompactionCheckpoint | None
    ) -> SessionCompactionCheckpoint | None:
        """未知格式或无法匹配原文的视图回退到全量历史，不能静默丢失轮次。"""
        if checkpoint is None or checkpoint.producer != "alice" or checkpoint.format_version != 1:
            return None
        for cursor in (checkpoint.reduced_through, checkpoint.summarized_through):
            if cursor is not None and cursor not in self._positions:
                return None
        if self._position(checkpoint.summarized_through) > self._position(
            checkpoint.reduced_through
        ):
            return None
        if bool(checkpoint.summary.strip()) != bool(checkpoint.summarized_through):
            return None
        return checkpoint

    def should_compact(self, *, force: bool = False) -> bool:
        """内部强制参数跳过阈值；空历史或关闭压缩时不生成检查点。"""
        return bool(
            self._config.enabled
            and self._context.session_history
            and (
                force
                or self.estimated_tokens > self._input_token_limit * self._config.trigger_ratio
            )
        )

    def reduce(self) -> CompactionResult:
        """先将原文后缀缩到近期预算内，保留旧摘要与覆盖范围。"""
        turns = self._context.session_history
        recent_budget = self._input_token_limit * self._config.recent_raw_ratio
        recent_tokens = 0
        boundary = len(turns)
        for index in range(len(turns) - 1, self._reduced_count - 1, -1):
            turn_tokens = estimate_context_tokens(self._turn_messages(index, reduced=False))
            if recent_tokens + turn_tokens > recent_budget:
                break
            recent_tokens += turn_tokens
            boundary = index
        checkpoint = (
            self._checkpoint or SessionCompactionCheckpoint(producer="alice", format_version=1)
        ).model_copy(
            update={
                "reduced_through": turns[boundary - 1].interaction_id if boundary else None,
                "created_at": self._clock(),
            }
        )
        return self._result(checkpoint)

    async def compact(
        self, *, summarizer: RelaySummarizer, model: ResolvedFrameModel
    ) -> CompactionResult:
        """减重后按需接力摘要；任一折叠失败都不提交局部摘要，取消不降级。"""
        reduced = self.reduce()
        target = self._input_token_limit * self._config.target_ratio
        turns = self._context.session_history
        if reduced.tokens_after <= target or self._summarized_count == len(turns):
            return self._guard(reduced)

        # 每个候选前缀独立保留最近的有限引用，避免用后缀线索污染摘要。
        references = self._checkpoint.references if self._checkpoint else ()
        prefix_references = {self._summarized_count: references}
        for index in range(self._summarized_count, len(turns)):
            references = extend_references(
                references,
                turns[index],
                limit=self._config.max_references,
                max_chars=self._config.reference_max_chars,
            )
            prefix_references[index + 1] = references
        reduction_boundary = self._position(reduced.checkpoint.reduced_through)
        suffix_tokens = [0] * (len(turns) + 1)
        for index in range(len(turns) - 1, self._summarized_count - 1, -1):
            suffix_tokens[index] = suffix_tokens[index + 1] + estimate_context_tokens(
                self._turn_messages(index, reduced=index < reduction_boundary)
            )
        candidates = [
            (index, self._reserved_tokens(prefix_references[index]) + suffix_tokens[index])
            for index in range(self._summarized_count + 1, len(turns) + 1)
        ]
        boundary = next((index for index, tokens in candidates if tokens <= target), None)
        if boundary is None:
            if reduced.tokens_after <= self._input_token_limit:
                return reduced
            # 完整目标不可达时，只为恢复可执行请求付出摘要成本。
            boundary = next(
                (index for index, tokens in candidates if tokens <= self._input_token_limit),
                None,
            )
            if boundary is None:
                return replace(
                    reduced,
                    error=InputTooLargeError(
                        "Fixed input and summary reservation exceed input budget."
                    ),
                    reason="input_too_large",
                )
        serialized = [
            {
                "interaction_id": turns[index].interaction_id,
                "agent_id": turns[index].agent_id,
                "outcome": turns[index].payload.outcome,
                "messages": self._turn_messages(index, reduced=True, for_summary=True),
            }
            for index in range(self._summarized_count, boundary)
        ]
        try:
            summary = await summarizer.fold(
                reduced.checkpoint.summary,
                serialized,
                model=model,
                config=self._config,
                language=self._language,
            )
        except SummaryRejectedError as error:
            return self._guard(
                replace(reduced, layer="summary", summary_status="abandoned", reason=error.reason)
            )
        checkpoint = reduced.checkpoint.model_copy(
            update={
                "summary": summary,
                "references": prefix_references[boundary],
                "summarized_through": turns[boundary - 1].interaction_id,
                "reduced_through": turns[max(boundary, reduction_boundary) - 1].interaction_id,
                "created_at": self._clock(),
            }
        )
        return self._guard(
            replace(self._result(checkpoint), layer="summary", summary_status="completed")
        )

    def _guard(self, result: CompactionResult) -> CompactionResult:
        if result.tokens_after > self._input_token_limit:
            return replace(
                result,
                error=CompactionFailedError("Compacted history exceeds input budget."),
                reason=result.reason or "compaction_failed",
            )
        return result

    def _reserved_tokens(self, references: tuple[str, ...]) -> int:
        """预留完整摘要段；中英文两组字符分别取整，拼接最多多出两个 token。"""
        placeholder = "x" * (self._config.summary_max_tokens * 4)
        return estimate_context_tokens(self._fixed(placeholder, references)) + 2

    def _fixed(self, summary: str = "", references: tuple[str, ...] = ()) -> list[dict[str, str]]:
        return self._assembler.build_main_agent_messages(
            self._context,
            session_summary=summary,
            session_references=references,
        )

    def _turn_messages(
        self, index: int, *, reduced: bool, for_summary: bool = False
    ) -> list[dict[str, str]]:
        turn = self._context.session_history[index]
        return self._builder.build_messages(
            [turn],
            current_agent_id=self._context.labels.agent_id,
            language=self._language,
            reduced_through=turn.interaction_id if reduced else None,
            tool_result_excerpt_chars=self._config.tool_result_excerpt_chars,
            tool_result_max_chars=self._config.summary_input_tool_chars if for_summary else None,
        )

    def _render(
        self, reduced_count: int, checkpoint: SessionCompactionCheckpoint | None
    ) -> list[dict[str, str]]:
        """已被摘要覆盖的前缀不重放，其后的减重与原文边界保持稳定。"""
        summarized = self._position(checkpoint.summarized_through) if checkpoint else 0
        fixed = (
            self._fixed(checkpoint.summary, checkpoint.references)
            if checkpoint and checkpoint.summary
            else self._fixed_messages
        )
        history = [
            message
            for index in range(summarized, len(self._context.session_history))
            for message in self._turn_messages(index, reduced=index < reduced_count)
        ]
        return [*fixed[:-1], *history, *fixed[-1:]]

    def _result(self, checkpoint: SessionCompactionCheckpoint) -> CompactionResult:
        boundary = self._position(checkpoint.reduced_through)
        messages = self._render(boundary, checkpoint)
        return CompactionResult(
            messages=messages,
            checkpoint=checkpoint,
            tokens_before=self.estimated_tokens,
            tokens_after=estimate_context_tokens(messages),
            covered_turns=boundary,
            summarized_turns=self._position(checkpoint.summarized_through),
        )
