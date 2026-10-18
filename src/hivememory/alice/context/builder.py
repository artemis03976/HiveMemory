"""按模型预算准备根帧输入，保留可在取消收尾时读取的完整检查点。"""

from time import perf_counter
from typing import Any, Protocol

from hivememory.agent_runtime import AgentRuntime, InputTooLargeError, ResolvedFrameModel
from hivememory.alice.context.summarizer import RelaySummarizer
from hivememory.alice.context.view import CompactionFailedError, SessionContextView
from hivememory.alice.events import BoundAgentRunEvents
from hivememory.config.alice import CompactionConfig
from hivememory.core.errors import ModelNotFoundError
from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.core.protocol.models import AgentRunContext, SessionCompactionCheckpoint
from hivememory.prompts.assembler import AgentPromptAssembler


class ContextProgressOutput(Protocol):
    """上下文准备的交互进度出口，与观测事件发布相互独立。"""

    async def context_compaction(self, *, status: str, data: dict[str, object]) -> None: ...


class RootContextBuilder:
    """每次 run 独占的输入构建器；取消不会把未完成的摘要变成检查点。"""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        prompt_assembler: AgentPromptAssembler,
        config: CompactionConfig,
        summarizer: RelaySummarizer,
    ) -> None:
        self._runtime = runtime
        self._prompt_assembler = prompt_assembler
        self._config = config
        self._summarizer = summarizer
        self.messages: list[dict[str, str]] = []
        self.model: ResolvedFrameModel | None = None
        self.checkpoint: SessionCompactionCheckpoint | None = None
        self.error: Exception | None = None

    async def build(
        self,
        context: AgentRunContext,
        *,
        checkpoint: SessionCompactionCheckpoint | None,
        generation_options: dict[str, Any] | None,
        events: BoundAgentRunEvents,
        output: ContextProgressOutput,
    ) -> None:
        """模型与预算先于视图组装；可预期的输入失败留给运行层形成终态。"""
        try:
            self.model = self._runtime.resolve_model(
                context.agent_profile or OMNI_DOLL_PROFILE, generation_options
            )
            view = SessionContextView(
                context=context,
                prompt_assembler=self._prompt_assembler,
                checkpoint=checkpoint,
                input_token_limit=self.model.input_token_limit,
                config=self._config,
            )
            self.messages = view.messages
            if not view.should_compact():
                return
            progress: dict[str, object] = {
                "layer": "reduction",
                "tokens_before": view.estimated_tokens,
                "tokens_after": view.estimated_tokens,
                "covered_turns": 0,
            }
            events.compaction(status="started", data=progress)
            await output.context_compaction(status="started", data=progress)
            started_at = perf_counter()
            compacted = await view.compact(summarizer=self._summarizer, model=self.model)
            self.messages = compacted.messages
            self.checkpoint = compacted.checkpoint
            progress = {
                "layer": compacted.layer,
                "summary_status": compacted.summary_status,
                "tokens_before": compacted.tokens_before,
                "tokens_after": compacted.tokens_after,
                "covered_turns": compacted.covered_turns,
                "summarized_turns": compacted.summarized_turns,
                "elapsed_ms": (perf_counter() - started_at) * 1000,
            }
            status = (
                "failed"
                if compacted.error
                else ("abandoned" if compacted.summary_status == "abandoned" else "completed")
            )
            events.compaction(status=status, data=progress, reason=compacted.reason)
            await output.context_compaction(status=status, data=progress)
            if compacted.error is not None:
                raise compacted.error
        except (ModelNotFoundError, InputTooLargeError, CompactionFailedError) as error:
            self.error = error
