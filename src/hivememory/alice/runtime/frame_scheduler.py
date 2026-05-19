"""
帧调度器 (Frame Scheduler)

管理主/子 Agent 的运行时帧栈，负责帧的创建、挂起、恢复和销毁。
Phase 2 多智能体子代理调用的核心调度组件。
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from hivememory.core.models import AgentProfile, Identity
from hivememory.alice.runtime.execution_frame import ExecutionFrame
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes

if TYPE_CHECKING:
    from hivememory.alice.runtime.runtime import AliceRuntime

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

    def __init__(self, runtime: "AliceRuntime"):
        """
        初始化帧调度器。

        Args:
            runtime: AliceRuntime 实例
        """
        self._runtime = runtime
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
        frame = ExecutionFrame(
            process_id=f"pid_main_{self._frame_counter}",
            agent_profile=agent_profile,
            working_history=messages,
            depth=0,
            topic_id=topic_id,
            identity=identity,
        )
        logger.debug(f"Created main frame: {frame}")
        return frame

    async def fork_sub_frame(
        self,
        parent_frame: ExecutionFrame,
        target_alias: str,
        task: str,
        context_refs: List[str],
    ) -> ExecutionFrame:
        """
        派生子 Agent 帧 (depth=1)。

        流程:
        1. 加载子 Agent 图纸
        2. 构建 System Prompt (剥离 CALL 指令教学)
        3. 注入 context_refs 内容 (零开销上下文继承)
        4. 创建瞬态帧 (topic_id=None)
        """
        sub_profile = await self._runtime.get_agent_profile(target_alias)

        logger.info(
            f"Forking sub-frame: target={target_alias}, "
            f"task='{task[:50]}...', context_refs={context_refs}"
        )

        system_prompt = await self._build_sub_agent_system_prompt(
            profile=sub_profile,
            persona=sub_profile.persona,
            context_refs=context_refs,
            depth=1,
        )

        working_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        self._frame_counter += 1
        sub_frame = ExecutionFrame(
            process_id=f"pid_sub_{self._frame_counter}",
            agent_profile=sub_profile,
            working_history=working_history,
            depth=1,
            topic_id=None,
            parent_frame_id=parent_frame.process_id,
            identity=parent_frame.identity,
        )

        logger.debug(f"Created sub-frame: {sub_frame}")
        return sub_frame

    async def _build_sub_agent_system_prompt(
        self,
        profile: AgentProfile,
        persona: str,
        context_refs: List[str],
        depth: int,
    ) -> str:
        """
        构建子 Agent 的 System Prompt。

        关键点:
            - depth>=1 时动态剥离 CALL 指令教学 (软限制)
            - 不注入 [Available Sub-Agents] 菜单
            - 注入 context_refs 指针内容 (零开销上下文继承)
        """
        from hivememory.prompts.system_prompt import SystemPromptBuilder

        language = (
            self._runtime.config.koakuma.mtp_prompt.language
            if self._runtime.config.koakuma.mtp_prompt
            else "zh"
        )
        builder = SystemPromptBuilder(language=language)

        mtp_prompt = self._runtime.get_mtp_prompt(profile=profile)
        if mtp_prompt and depth >= 1:
            mtp_prompt = self._strip_call_from_prompt(mtp_prompt)
        builder.with_mtp_prompt(mtp_prompt)

        if persona:
            builder.with_persona(persona)

        if context_refs:
            context_content = await self._fetch_context_refs_content(context_refs)
            if context_content:
                builder.with_shared_context(context_content)

        return builder.build()

    def _strip_call_from_prompt(self, prompt: str) -> str:
        """
        从 MTP prompt 中移除 CALL 指令教学 (软限制)。

        通过移除 CALL 相关章节降低子 Agent 产生 CALL 幻觉的概率。
        """
        lines = prompt.split("\n")
        filtered_lines = []
        skip_section = False

        for line in lines:
            if "CALL" in line and ("##" in line or "###" in line or "**CALL**" in line):
                skip_section = True
                continue

            if skip_section and ("##" in line or "###" in line) and "CALL" not in line:
                skip_section = False

            if not skip_section:
                filtered_lines.append(line)

        result = "\n".join(filtered_lines)
        logger.debug(f"Stripped CALL from prompt: {len(prompt)} -> {len(result)} chars")
        return result

    async def _fetch_context_refs_content(self, aliases: List[str]) -> str:
        """
        从存储层批量获取 context_refs 的完整内容。

        零开销上下文继承: 直接注入父 Agent 提供的记忆内容，
        子 Agent 无需再次调用 SEARCH/READ。
        """
        contents = []
        user_id = (
            self._runtime.koakuma._current_identity.user_id
            if hasattr(self._runtime.koakuma, "_current_identity")
            else None
        )

        global_bus = self._runtime.global_bus

        for alias in aliases:
            atom = self._runtime.koakuma.atom_cache.get_atom_by_alias(alias)

            if atom is None and user_id and global_bus is not None:
                try:
                    atom = await global_bus.request(
                        PatchouliRoutes.MEMORY_GET_BY_ALIAS,
                        alias,
                        user_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch context_ref '{alias}': {e}")
                    continue

            if atom:
                title = atom.index.title or alias
                content = atom.payload.content if atom.payload else "(empty)"
                contents.append(f"### [{alias}] {title}\n\n{content}")
            else:
                logger.warning(f"Context ref '{alias}' not found")

        if not contents:
            return ""

        return "[Shared Context from Parent Agent]\n\n" + "\n\n---\n\n".join(contents)

    def suspend_frame(self, frame: ExecutionFrame) -> None:
        """挂起当前帧（压栈）。"""
        self._frame_stack.append(frame)
        logger.debug(
            f"Suspended frame: {frame.process_id}, stack_depth={len(self._frame_stack)}"
        )

    def resume_frame(self) -> Optional[ExecutionFrame]:
        """恢复父帧（出栈）。"""
        if self._frame_stack:
            frame = self._frame_stack.pop()
            logger.debug(
                f"Resumed frame: {frame.process_id}, stack_depth={len(self._frame_stack)}"
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
