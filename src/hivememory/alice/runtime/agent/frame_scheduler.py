"""
帧调度器 (Frame Scheduler)

管理主/子 Agent 的运行时帧栈，负责帧的创建、挂起、恢复和销毁。
Phase 2 多智能体子代理调用的核心调度组件。
"""

import logging
from typing import TYPE_CHECKING, List, Optional
from uuid import uuid4

from hivememory.core.models import AgentProfile, Identity
from hivememory.alice.runtime.models import ExecutionFrame, RuntimeScope

if TYPE_CHECKING:
    from hivememory.prompts.assembler import AgentPromptAssembler

logger = logging.getLogger(__name__)


class FrameScheduler:
    """
    执行帧调度器 - 管理主/子 Agent 的运行时帧栈。

    职责:
        - 帧栈管理 (suspend/resume)
        - 子代理帧构建 (context_refs 注入)
        - 深度跟踪与强制
        - System Prompt 动态裁剪 (剥离 CALL 权限)
    """

    def __init__(
        self,
        prompt_assembler: "AgentPromptAssembler",
    ):
        """
        初始化帧调度器。

        Args:
            prompt_assembler: Agent prompt 组装器
        """
        self._prompt_assembler = prompt_assembler
        self._frame_stack: List[ExecutionFrame] = []
        self._frame_counter = 0

    def create_main_frame(
        self,
        agent_profile: AgentProfile,
        messages: List[dict],
        topic_id: str,
        identity: Identity,
    ) -> ExecutionFrame:
        """
        创建主 Agent 帧 (depth=0)。

        主帧从感知层 TopicBuffer 装载，执行后卸载回 MMU。
        """
        self._frame_counter += 1
        run_id = f"run_{uuid4().hex}"
        frame_id = f"frame_main_{self._frame_counter}"
        frame = ExecutionFrame(
            runtime_scope=RuntimeScope(
                run_id=run_id,
                frame_id=frame_id,
                depth=0,
            ),
            agent_profile=agent_profile,
            working_history=messages,
            topic_id=topic_id,
            identity=identity,
        )
        logger.debug(f"Created main frame: {frame}")
        return frame

    async def fork_sub_frame(
        self,
        parent_frame: ExecutionFrame,
        agent_profile: AgentProfile,
        task: str,
        shared_context: str = "",
    ) -> ExecutionFrame:
        """
        派生子 Agent 帧 (depth=1)。

        流程:
        1. 构建 System Prompt (剥离 CALL 指令教学)
        2. 注入调用方准备好的 shared_context (零开销上下文继承)
        3. 创建瞬态帧 (topic_id=None)
        """
        logger.info(
            f"Forking sub-frame: agent={getattr(agent_profile, 'alias', None) or 'unknown'}, "
            f"task='{task[:50]}...', has_shared_context={bool(shared_context)}"
        )

        working_history = self._prompt_assembler.build_sub_agent_messages(
            profile=agent_profile,
            task=task,
            shared_context=shared_context,
            depth=1,
        )

        self._frame_counter += 1
        frame_id = f"frame_sub_{self._frame_counter}"
        sub_frame = ExecutionFrame(
            runtime_scope=parent_frame.runtime_scope.for_child(frame_id),
            agent_profile=agent_profile,
            working_history=working_history,
            topic_id=None,
            identity=parent_frame.identity,
        )

        logger.debug(f"Created sub-frame: {sub_frame}")
        return sub_frame

    def suspend_frame(self, frame: ExecutionFrame) -> None:
        """挂起当前帧（压栈）。"""
        self._frame_stack.append(frame)
        logger.debug(
            f"Suspended frame: {frame.runtime_scope.frame_id}, stack_depth={len(self._frame_stack)}"
        )

    def resume_frame(self) -> Optional[ExecutionFrame]:
        """恢复父帧（出栈）。"""
        if self._frame_stack:
            frame = self._frame_stack.pop()
            logger.debug(
                f"Resumed frame: {frame.runtime_scope.frame_id}, stack_depth={len(self._frame_stack)}"
            )
            return frame
        logger.warning("Attempted to resume frame but stack is empty")
        return None

    def get_current_depth(self) -> int:
        """获取当前调用栈深度。"""
        return len(self._frame_stack)

    def clear_stack(self) -> None:
        """清空帧栈（用于错误恢复）。"""
        self._frame_stack.clear()
        logger.debug("Cleared frame stack")
