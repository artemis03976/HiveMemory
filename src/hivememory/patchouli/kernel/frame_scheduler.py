"""
帧调度器 (Frame Scheduler)

管理主/子 Agent 的运行时帧栈，负责帧的创建、挂起、恢复和销毁。
Phase 2 多智能体子代理调用的核心调度组件。

作者: HiveMemory Team
版本: 3.0 (Phase 2)
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from hivememory.core.models import AgentProfileConfig, Identity
from hivememory.patchouli.kernel.execution_frame import ExecutionFrame

if TYPE_CHECKING:
    from hivememory.patchouli.kernel.core import PatchouliKernel

logger = logging.getLogger(__name__)


class FrameScheduler:
    """
    执行帧调度器 - 管理主/子 Agent 的运行时帧栈

    职责:
        - 帧栈管理 (suspend/resume)
        - 子代理帧构建 (context_refs 注入)
        - 深度跟踪与强制
        - System Prompt 动态裁剪 (剥离 CALL 权限)

    架构:
        - 主 Agent (depth=0): 从 TopicBuffer 装载，执行后卸载回 MMU
        - 子 Agent (depth=1): 内存中直接构造，执行后 GC 销毁（瞬态沙盒）
        - 星型拓扑约束: 强制 depth ≤ 1，防止递归黑洞

    Examples:
        >>> scheduler = FrameScheduler(kernel)
        >>>
        >>> # 创建主帧
        >>> main_frame = scheduler.create_main_frame(
        ...     agent_profile=coder_profile,
        ...     messages=[...],
        ...     topic_id="topic_123",
        ...     identity=identity,
        ... )
        >>>
        >>> # 派生子帧
        >>> sub_frame = await scheduler.fork_sub_frame(
        ...     parent_frame=main_frame,
        ...     target_alias="tester_doll",
        ...     task="为上述代码编写单元测试",
        ...     context_refs=["mem_api_spec"],
        ... )
    """

    def __init__(self, kernel: "PatchouliKernel"):
        """
        初始化帧调度器

        Args:
            kernel: PatchouliKernel 实例（用于访问存储、缓存、配置等）
        """
        self.kernel = kernel
        self._frame_stack: List[ExecutionFrame] = []
        self._frame_counter = 0

    def create_main_frame(
        self,
        agent_profile: AgentProfileConfig,
        messages: List[dict],
        topic_id: str,
        identity: Identity,
    ) -> ExecutionFrame:
        """
        创建主 Agent 帧 (depth=0)

        主帧从感知层 TopicBuffer 装载，执行后卸载回 MMU。

        Args:
            agent_profile: 主 Agent 的图纸配置
            messages: 从感知层组装的初始 messages
            topic_id: 挂载的话题 ID
            identity: 身份标识

        Returns:
            ExecutionFrame: 主 Agent 帧
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
        派生子 Agent 帧 (depth=1)

        子帧在内存中直接构造，执行后 GC 销毁（瞬态沙盒）。

        流程:
        1. 加载子 Agent 图纸
        2. 构建 System Prompt (剥离 CALL 指令教学)
        3. 注入 context_refs 内容 (零开销上下文继承)
        4. 创建瞬态帧 (topic_id=None)

        Args:
            parent_frame: 父 Agent 帧
            target_alias: 子 Agent 别名
            task: 任务描述 (自然语言)
            context_refs: 共享内存指针列表 (记忆别名)

        Returns:
            ExecutionFrame: 子 Agent 帧

        Raises:
            ValueError: 如果子 Agent 图纸不存在
        """
        # 1. 加载子 Agent 图纸
        sub_profile = self.kernel.load_agent_profile(target_alias)
        sub_persona = self.kernel.get_agent_persona(target_alias)

        logger.info(
            f"Forking sub-frame: target={target_alias}, "
            f"task='{task[:50]}...', context_refs={context_refs}"
        )

        # 2. 构建 System Prompt (剥离 CALL 权限)
        system_prompt = await self._build_sub_agent_system_prompt(
            profile=sub_profile,
            persona=sub_persona,
            context_refs=context_refs,
            depth=1,
        )

        # 3. 组装初始 history
        working_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        # 4. 创建瞬态帧
        self._frame_counter += 1
        sub_frame = ExecutionFrame(
            process_id=f"pid_sub_{self._frame_counter}",
            agent_profile=sub_profile,
            working_history=working_history,
            depth=1,
            topic_id=None,  # 瞬态沙盒，无 topic
            parent_frame_id=parent_frame.process_id,
            identity=parent_frame.identity,
        )

        logger.debug(f"Created sub-frame: {sub_frame}")
        return sub_frame

    async def _build_sub_agent_system_prompt(
        self,
        profile: AgentProfileConfig,
        persona: str,
        context_refs: List[str],
        depth: int,
    ) -> str:
        """
        构建子 Agent 的 System Prompt

        关键点:
        - 动态剥离 CALL 指令教学 (软限制)
        - 不注入 [Available Sub-Agents] 菜单
        - 注入 context_refs 指针内容 (零开销上下文继承)

        Args:
            profile: 子 Agent 图纸配置
            persona: 子 Agent 灵魂文本
            context_refs: 共享内存指针列表
            depth: 调用栈深度

        Returns:
            str: 子 Agent 的 System Prompt
        """
        from hivememory.prompts.system_prompt import SystemPromptBuilder

        language = (
            self.kernel.config.koakuma.mtp_prompt.language
            if self.kernel.config.koakuma.mtp_prompt
            else "zh"
        )
        builder = SystemPromptBuilder(language=language)

        # Top: MTP 协议教学 (剥离 CALL)
        mtp_prompt = self.kernel.get_mtp_prompt(profile=profile)
        if mtp_prompt and depth >= 1:
            # 移除 CALL 指令的教学部分 (软限制)
            mtp_prompt = self._strip_call_from_prompt(mtp_prompt)
        builder.with_mtp_prompt(mtp_prompt)

        # Top: 存储降级通知
        if mtp_prompt and not self.kernel.check_storage_health():
            builder.with_storage_offline_notice()

        # Middle: 灵魂注入
        if persona:
            builder.with_persona(persona)

        # Bottom: 注入 context_refs 内容 (零开销上下文继承)
        if context_refs:
            context_content = await self._fetch_context_refs_content(context_refs)
            if context_content:
                builder.with_shared_context(context_content)

        return builder.build()

    def _strip_call_from_prompt(self, prompt: str) -> str:
        """
        从 MTP prompt 中移除 CALL 指令教学 (软限制)

        通过移除 CALL 相关的教学内容，降低子 Agent 产生 CALL 幻觉的概率。

        Args:
            prompt: 原始 MTP prompt

        Returns:
            str: 移除 CALL 教学后的 prompt
        """
        # 简单实现: 移除包含 "CALL" 的段落
        # TODO: 更精确的实现可以使用正则或结构化解析
        lines = prompt.split("\n")
        filtered_lines = []
        skip_section = False

        for line in lines:
            # 检测 CALL 相关章节的开始
            if "CALL" in line and ("##" in line or "###" in line or "**CALL**" in line):
                skip_section = True
                continue

            # 检测下一个章节的开始 (结束 skip)
            if skip_section and ("##" in line or "###" in line) and "CALL" not in line:
                skip_section = False

            if not skip_section:
                filtered_lines.append(line)

        result = "\n".join(filtered_lines)
        logger.debug(f"Stripped CALL from prompt: {len(prompt)} -> {len(result)} chars")
        return result

    async def _fetch_context_refs_content(self, aliases: List[str]) -> str:
        """
        从存储层批量获取 context_refs 的完整内容

        零开销上下文继承: 直接注入父 Agent 提供的记忆内容，
        子 Agent 无需再次调用 SEARCH/READ。

        Args:
            aliases: 记忆别名列表

        Returns:
            str: 格式化的记忆内容 (Markdown)
        """
        contents = []
        user_id = self.kernel.koakuma._current_identity.user_id if hasattr(self.kernel.koakuma, '_current_identity') else None

        for alias in aliases:
            # 1. 尝试从缓存获取
            atom = self.kernel.koakuma.atom_cache.get_atom_by_alias(alias)

            # 2. 缓存未命中，查询存储
            if atom is None and user_id:
                try:
                    atom = self.kernel.storage.get_memory_by_alias(
                        alias=alias,
                        user_id=user_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch context_ref '{alias}': {e}")
                    continue

            # 3. 格式化内容
            if atom:
                title = atom.index.title or alias
                content = atom.payload.content if atom.payload else "(empty)"
                contents.append(f"### [{alias}] {title}\n\n{content}")
            else:
                logger.warning(f"Context ref '{alias}' not found")

        if not contents:
            return ""

        # 组装为 Markdown 格式
        header = "[Shared Context from Parent Agent]"
        return f"{header}\n\n" + "\n\n---\n\n".join(contents)

    def suspend_frame(self, frame: ExecutionFrame) -> None:
        """
        挂起当前帧 (压栈)

        当主 Agent 发出 CALL 指令时，将其帧压入栈中，等待子 Agent 执行完毕后恢复。

        Args:
            frame: 要挂起的帧
        """
        self._frame_stack.append(frame)
        logger.debug(f"Suspended frame: {frame.process_id}, stack_depth={len(self._frame_stack)}")

    def resume_frame(self) -> Optional[ExecutionFrame]:
        """
        恢复父帧 (出栈)

        子 Agent 执行完毕后，从栈中弹出父帧并恢复执行。

        Returns:
            ExecutionFrame: 恢复的父帧，如果栈为空则返回 None
        """
        if self._frame_stack:
            frame = self._frame_stack.pop()
            logger.debug(f"Resumed frame: {frame.process_id}, stack_depth={len(self._frame_stack)}")
            return frame
        logger.warning("Attempted to resume frame but stack is empty")
        return None

    def get_current_depth(self) -> int:
        """
        获取当前调用栈深度

        Returns:
            int: 栈深度 (0 = 主 Agent, 1 = 子 Agent)
        """
        return len(self._frame_stack)

    def clear_stack(self) -> None:
        """清空帧栈 (用于错误恢复)"""
        self._frame_stack.clear()
        logger.debug("Cleared frame stack")
