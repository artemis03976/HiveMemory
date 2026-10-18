"""按模型预算准备根帧输入，保留可在取消收尾时读取的完整检查点。"""

from time import perf_counter
from typing import Any, Protocol

from hivememory.agent_runtime import AgentRuntime, InputTooLargeError, ResolvedFrameModel
from hivememory.alice.context.summarizer import RelaySummarizer
from hivememory.alice.context.view import CompactionFailedError, SessionContextView
from hivememory.alice.events import BoundAgentRunEvents
from hivememory.config.alice import CompactionConfig, KoakumaConfig
from hivememory.core.errors import ModelNotFoundError
from hivememory.core.protocol.models import SessionCompactionCheckpoint
from hivememory.prompts.mtp import MTPPromptBuilder
from hivememory.workspace.contracts import CPUInputManifest


class ContextProgressOutput(Protocol):
    """上下文准备的交互进度出口，与观测事件发布相互独立。"""

    async def context_compaction(self, *, status: str, data: dict[str, object]) -> None: ...


class RootContextBuilder:
    """每次 run 独占的输入构建器；取消不会把未完成的摘要变成检查点。"""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        koakuma_config: KoakumaConfig,
        config: CompactionConfig,
        summarizer: RelaySummarizer,
    ) -> None:
        self._runtime = runtime
        self._koakuma_config = koakuma_config
        self._config = config
        self._summarizer = summarizer
        self.messages: list[dict[str, str]] = []
        self.model: ResolvedFrameModel | None = None
        self.checkpoint: SessionCompactionCheckpoint | None = None
        self.error: Exception | None = None

    async def build(
        self,
        manifest: CPUInputManifest,
        *,
        permitted_verbs: frozenset[str],
        generation_options: dict[str, Any] | None,
        events: BoundAgentRunEvents,
        output: ContextProgressOutput,
    ) -> None:
        """模型与预算先于视图组装；可预期的输入失败留给运行层形成终态。"""
        try:
            self.model = self._runtime.resolve_model(manifest.agent_profile, generation_options)
            # 工具教学属于 Alice；语言沿用进程的解析结果，不能重新读取默认语言。
            prompt_config = self._koakuma_config.mtp_prompt
            mtp_prompt = (
                MTPPromptBuilder(
                    language=manifest.language,
                    include_demo=prompt_config.include_demo,
                    include_error_handling=prompt_config.include_error_handling,
                    allowed_verbs=permitted_verbs,
                ).build()
                if self._koakuma_config.enabled and prompt_config.enabled
                else ""
            )
            view = SessionContextView(
                instructions=manifest.instructions,
                turn_context=manifest.turn_context,
                user_message=manifest.user_message,
                language=manifest.language,
                mtp_prompt=mtp_prompt,
                session_history=manifest.session_history,
                current_agent_id=manifest.labels.agent_id,
                checkpoint=manifest.compaction_checkpoint,
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
