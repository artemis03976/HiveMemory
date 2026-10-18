"""Alice 运行用例：准备输入、驱动根帧与 CALL，并同步交回终态结果。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from hivememory.agent_runtime import (
    AgentRuntime,
    ExecutionFrame,
    FrameExecutionPolicy,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.alice.context.builder import RootContextBuilder
from hivememory.alice.context.summarizer import RelaySummarizer
from hivememory.alice.events import AgentRunEventEmitter, BoundAgentRunEvents
from hivememory.alice.orchestration.call.coordinator import CallCoordinator
from hivememory.alice.orchestration.executor import RunExecutor
from hivememory.alice.orchestration.frames import FrameFactory, FrameSpec
from hivememory.alice.orchestration.session import RunSession
from hivememory.alice.orchestration.verbs import mtp_verbs_for_operations
from hivememory.alice.run.output import RunOutput
from hivememory.alice.run.result import assemble_execution_result, publish_terminal, stats_for
from hivememory.config.alice import CompactionConfig
from hivememory.core.models import OMNI_DOLL_PROFILE, AgentProfile, ExecutionLabels
from hivememory.core.protocol.models import AgentRunContext
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.workspace.contracts import (
    CPUExecutionResult,
    CPUExecutionStatus,
    CPUInputManifest,
    CPUResultSlot,
    OperationSubmitter,
)


class RunExitReason(str, Enum):
    """运行收尾原因；既未形成终态也未发生普通异常时按任务取消收口。"""

    RUNNING = "running"
    TERMINAL = "terminal"
    FAILED = "failed"


def _agent_run_context_from_manifest(manifest: CPUInputManifest) -> AgentRunContext:
    """把 CPU 输入清单转换为提示词组装使用的内部运行上下文。

    只取提示词组装与运行事件需要的字段：记忆以任务进程编译好的文本进入，
    未编译的检索原子不复制。``AgentRunContext`` 不出现在任何公开路由上。
    操作快照在 Alice 一侧翻译为 MTP 动词，提示词与 root frame 共享同一集合；
    此转换只描述执行能力，不参与资源授权。
    """
    return AgentRunContext(
        labels=manifest.labels,
        topic_id=manifest.topic_id,
        user_message=manifest.user_message,
        session_history=manifest.session_history,
        memory_context=manifest.memory_context,
        agent_profile=manifest.agent_profile,
        permitted_mtp_verbs=mtp_verbs_for_operations(manifest.allowed_operations),
        storage_available=manifest.storage_available,
        attachment_context=manifest.attachment_context,
    )


@dataclass(frozen=True, slots=True)
class _RunPreparation:
    """单次运行的内部上下文、帧会话与已发布 started 的观测绑定。"""

    context: AgentRunContext
    session: RunSession
    events: BoundAgentRunEvents


class AgentRunService:
    """运行服务只消费输出端口；队列、任务与交互流由接口层管理。"""

    def __init__(
        self,
        *,
        agent_runtime: AgentRuntime,
        call_coordinator: CallCoordinator,
        frame_factory: FrameFactory,
        prompt_assembler: AgentPromptAssembler,
        agent_run_events: AgentRunEventEmitter,
        compaction_config: CompactionConfig,
        summarizer: RelaySummarizer,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._call_coordinator = call_coordinator
        self._frame_factory = frame_factory
        self._prompt_assembler = prompt_assembler
        self._agent_run_events = agent_run_events
        self._compaction_config = compaction_config
        self._summarizer = summarizer

    async def run_agent(
        self,
        input_manifest: CPUInputManifest,
        generation_options: dict[str, Any] | None = None,
        *,
        submit_operation: OperationSubmitter,
        run_output: RunOutput,
        result_slot: CPUResultSlot | None = None,
    ) -> CPUExecutionResult:
        """执行一次 run 并返回终态；交互输出只经端口发出，取消前同步保存已完成检查点与部分输出。"""
        preparation = self._prepare_run(input_manifest)
        events = preparation.events
        inputs = RootContextBuilder(
            runtime=self._agent_runtime,
            prompt_assembler=self._prompt_assembler,
            config=self._compaction_config,
            summarizer=self._summarizer,
        )
        frame: ExecutionFrame | None = None
        executor: RunExecutor | None = None
        result: CPUExecutionResult | None = None
        exit_reason = RunExitReason.RUNNING
        try:
            run_output.bind_run(preparation.session.agent_run_id)
            executor = RunExecutor(
                agent_runtime=self._agent_runtime,
                session=preparation.session,
                call_coordinator=self._call_coordinator,
                run_events=events,
            )
            await inputs.build(
                preparation.context,
                checkpoint=input_manifest.compaction_checkpoint,
                generation_options=generation_options,
                events=events,
                output=run_output,
            )
            frame = self._create_root_frame(
                messages=inputs.messages,
                submit_operation=submit_operation,
                labels=preparation.context.labels,
                topic_id=preparation.context.topic_id,
                session=preparation.session,
                agent_profile=preparation.context.agent_profile,
                permitted_verbs=preparation.context.permitted_mtp_verbs,
            )
            frame.resolved_model = inputs.model
            if inputs.model is not None:
                frame.progress.model_used = inputs.model.display_name
            if inputs.error is not None:
                terminal = FrameExecutionResult(
                    status=FrameExecutionStatus.FAILED, error=inputs.error
                )
            else:
                terminal = await executor.run(
                    frame, generation_options=generation_options, run_output=run_output
                )
            result = assemble_execution_result(frame, terminal, inputs.checkpoint)
            publish_terminal(events, result, stats_for(frame))
            exit_reason = RunExitReason.TERMINAL
            return result
        except Exception:
            exit_reason = RunExitReason.FAILED
            events.failed(message="Agent run failed.")
            raise
        finally:
            if exit_reason == RunExitReason.RUNNING:
                events.cancelled(
                    message=run_output.cancellation_message, close_reason=run_output.close_reason
                )
            # executor 的递归取消已经收尾；只在这里同步投影结果，不能用 await 吞掉原取消。
            if (
                result is None
                and frame is not None
                and executor is not None
                and executor.terminal_result is not None
            ):
                result = assemble_execution_result(
                    frame, executor.terminal_result, inputs.checkpoint
                )
            elif (
                result is None
                and inputs.checkpoint is not None
                and exit_reason == RunExitReason.RUNNING
            ):
                result = CPUExecutionResult(
                    status=CPUExecutionStatus.CANCELLED, compaction_checkpoint=inputs.checkpoint
                )
            if result_slot is not None and result is not None:
                result_slot.result = result

    def _prepare_run(self, input_manifest: CPUInputManifest) -> _RunPreparation:
        """构造 run 上下文与会话，绑定 ``agent.run.*`` 事件并发布 started。"""
        context = _agent_run_context_from_manifest(input_manifest)
        session = self._create_run_session(process_id=input_manifest.process_id)
        run_events = self._events_for_run(session, context)
        run_events.started()
        return _RunPreparation(context=context, session=session, events=run_events)

    def _create_root_frame(
        self,
        *,
        messages: list[dict[str, str]],
        submit_operation: OperationSubmitter,
        labels: ExecutionLabels,
        topic_id: str,
        agent_profile: AgentProfile | None,
        permitted_verbs: frozenset[str],
        session: RunSession,
    ) -> ExecutionFrame:
        """按提示词使用的同一动词集合，创建并登记当前 run 的唯一 root frame。"""
        profile = agent_profile or OMNI_DOLL_PROFILE
        policy = FrameExecutionPolicy.from_verbs(
            permitted_verbs,
        )
        frame = self._frame_factory.create(
            FrameSpec(
                runtime_scope=self._frame_factory.scope(
                    labels=labels,
                    run_id=session.agent_run_id,
                ),
                profile=profile,
                messages=messages,
                topic_id=topic_id or "",
                execution_policy=policy,
                submit_operation=submit_operation,
            )
        )
        session.register_root_frame(frame)
        return frame

    @staticmethod
    def _create_run_session(
        *,
        process_id: str,
    ) -> RunSession:
        return RunSession(
            agent_run_id=f"agent_run_{uuid.uuid4().hex}",
            process_id=process_id,
        )

    def _events_for_run(
        self,
        session: RunSession,
        agent_run_context: AgentRunContext,
    ) -> BoundAgentRunEvents:
        return self._agent_run_events.for_run(
            agent_run_id=session.agent_run_id,
            process_id=session.process_id,
            topic_id=agent_run_context.topic_id,
            agent_id=agent_run_context.labels.agent_id,
            workspace_id=agent_run_context.labels.workspace_id,
        )


__all__ = ["AgentRunService"]
