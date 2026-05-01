"""
Kernel Loop Executor - 帧栈驱动的递归生成循环执行器

职责：
    - 管理主 Agent 和子 Agent 的递归执行循环
    - 处理 CALL 指令的挂起/恢复
    - 自动收割子帧生成的记忆别名
    - 组装 IPC 返回 payload

Phase A→B→C→D 循环：
    A. LLM 生成
    B. 自然停止检测
    C. MTP 执行 (可能触发 CALL → 子帧派生)
    D. 回填 & 继续

作者: HiveMemory Team
版本: 3.0 (Phase 2 重构)
"""

import json
import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from hivememory.patchouli.protocol.models import ChatResult
from hivememory.patchouli.kernel.runtime.execution_frame import ExecutionFrame
from hivememory.patchouli.mtp.models import MTPVerb
from hivememory.engines.perception.models import TraceItem

if TYPE_CHECKING:
    from hivememory.patchouli.kernel import PatchouliKernel
    from hivememory.patchouli.worker_agent import WorkerAgentService

logger = logging.getLogger(__name__)


class KernelLoopExecutor:
    """
    帧栈驱动的递归生成循环执行器

    封装了 Phase 2 多智能体系统的核心执行逻辑，支持：
    - 主 Agent 递归 MTP 执行
    - 子 Agent 调用 (CALL 指令 → 帧挂起/恢复)
    - 自动收割 (WRITE/UPDATE 别名跟踪)
    - 黑盒隔离 (子 Agent 细节不污染主 Agent)
    """

    def __init__(
        self,
        kernel: "PatchouliKernel",
        worker_agent: "WorkerAgentService",
    ):
        """
        初始化执行器

        Args:
            kernel: PatchouliKernel 实例
            worker_agent: WorkerAgentService 实例
        """
        self.kernel = kernel
        self.worker_agent = worker_agent

    async def execute_main_frame(
        self,
        messages: List[Dict[str, str]],
        max_iterations: Optional[int] = None,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        topic_id: Optional[str] = None,
        identity=None,
    ) -> ChatResult:
        """
        执行主帧的递归生成循环

        Args:
            messages: 初始 messages
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项
            agent_profile: 人偶图纸配置
            topic_id: 话题 ID
            identity: 完整身份标识

        Returns:
            ChatResult: 递归生成循环的完整结果
        """
        max_iter = max_iterations or self.kernel.config.koakuma.max_recursion_depth

        main_frame = self.kernel.frame_scheduler.create_main_frame(
            agent_profile=agent_profile or self.kernel.load_agent_profile("omni_doll"),
            messages=messages,
            topic_id=topic_id or "",
            identity=identity,
        )

        return await self.execute_frame(
            frame=main_frame,
            max_iterations=max_iter,
            generation_options=generation_options,
        )

    async def execute_frame(
        self,
        frame: ExecutionFrame,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """
        执行单个帧的递归循环

        这是 Phase 2 的核心方法，同时服务于主 Agent 和子 Agent。
        子 Agent 的 CALL 触发递归调用此方法。

        Phase A→B→C→D:
        A. LLM 生成
        B. 自然停止检测
        C. MTP 执行 (SUSPEND → 子帧派生)
        D. 回填 & 继续

        Args:
            frame: 执行帧
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项

        Returns:
            ChatResult: 执行结果
        """
        text_segments: List[str] = []
        mtp_commands: List[str] = []
        iteration = 0

        self.kernel.koakuma.set_current_identity(frame.identity)
        self.kernel.koakuma.set_active_profile(frame.agent_profile)
        self.kernel.koakuma.set_current_depth(frame.depth)
        self.kernel.koakuma.reset_interaction_state()

        while iteration < max_iterations:
            iteration += 1

            result = await self.worker_agent.generate_async(
                frame.working_history,
                **(generation_options or {}),
            )

            if not result.was_mtp_interrupted:
                text_segments.append(result.text)
                break

            text_segments.append(result.prefix_text)

            mtp_result = await self.kernel.handle_mtp(result.text)

            if mtp_result is None:
                text_segments.append(result.mtp_fragment)
                break

            if mtp_result.response_status == "suspend":
                ipc_response = await self._handle_call_suspend(
                    frame=frame,
                    mtp_result=mtp_result,
                    assistant_text=result.text,
                    max_iterations=max_iterations,
                    generation_options=generation_options,
                )

                frame.working_history.append(
                    {"role": "assistant", "content": result.text + "⟫"}
                )
                frame.working_history.append({
                    "role": "user",
                    "content": f"[System IPC Return]\n{ipc_response}",
                })
                mtp_commands.append("CALL")
                continue

            mtp_commands.append(
                mtp_result.command.verb.value
                if mtp_result.command else "UNKNOWN"
            )

            frame.working_history.append(
                {"role": "assistant", "content": result.text + "⟫"}
            )
            frame.working_history.append({
                "role": "user",
                "content": f"[System MTP Execution Result]\n{mtp_result.formatted_response}",
            })

            if frame.is_sub_frame() and mtp_result.command:
                self._try_harvest_alias(frame, mtp_result)

        return ChatResult(
            final_text="".join(text_segments),
            mtp_iterations=max(0, iteration - 1),
            total_iterations=iteration,
            mtp_commands_executed=mtp_commands,
        )

    async def execute_main_frame_stream(
        self,
        messages: List[Dict[str, str]],
        max_iterations: Optional[int] = None,
        generation_options: Optional[Dict[str, Any]] = None,
        agent_profile=None,
        topic_id: Optional[str] = None,
        identity=None,
    ):
        """
        执行主帧的流式递归生成循环

        与 execute_main_frame 相同，但以流式方式 yield SSE 事件。

        Args:
            messages: 初始 messages
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项
            agent_profile: 人偶图纸配置
            topic_id: 话题 ID
            identity: 完整身份标识

        Yields:
            Dict[str, Any]: SSE 事件 {"event": str, "data": dict}
        """
        max_iter = max_iterations or self.kernel.config.koakuma.max_recursion_depth

        main_frame = self.kernel.frame_scheduler.create_main_frame(
            agent_profile=agent_profile or self.kernel.load_agent_profile("omni_doll"),
            messages=messages,
            topic_id=topic_id or "",
            identity=identity,
        )

        async for event in self.execute_frame_stream(
            frame=main_frame,
            max_iterations=max_iter,
            generation_options=generation_options,
        ):
            yield event

    async def execute_frame_stream(
        self,
        frame: ExecutionFrame,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
    ):
        """
        执行单个帧的流式递归循环

        与 execute_frame 相同的核心逻辑，但逐 token yield LLM 生成内容。

        Yields:
            Dict[str, Any]: SSE 事件
                - {"event": "token", "data": {"content": str}}
                - {"event": "mtp_start", "data": {...}}
                - {"event": "mtp_result", "data": {...}}
                - {"event": "done", "data": ChatResult}
        """
        text_segments: List[str] = []
        mtp_commands: List[str] = []
        iteration = 0

        self.kernel.koakuma.set_current_identity(frame.identity)
        self.kernel.koakuma.set_active_profile(frame.agent_profile)
        self.kernel.koakuma.set_current_depth(frame.depth)
        self.kernel.koakuma.reset_interaction_state()

        while iteration < max_iterations:
            iteration += 1
            gen_result = None

            async for chunk in self.worker_agent.generate_stream(
                frame.working_history,
                **(generation_options or {}),
            ):
                if chunk.is_final:
                    gen_result = chunk.result
                    break
                if not chunk.mtp_detected and chunk.delta:
                    yield {"event": "token", "data": {"content": chunk.delta}}

            if gen_result is None:
                break

            if not gen_result.was_mtp_interrupted:
                text_segments.append(gen_result.text)
                break

            text_segments.append(gen_result.prefix_text)

            verb_hint, target_hint, args_hint, raw_hint = self._extract_mtp_hints(gen_result)
            yield {
                "event": "mtp_start",
                "data": {
                    "verb": verb_hint,
                    "target": target_hint,
                    "args": args_hint,
                    "raw_text": raw_hint,
                    "iteration": iteration,
                },
            }

            mtp_result = await self.kernel.handle_mtp(gen_result.text)

            if mtp_result is None:
                text_segments.append(gen_result.mtp_fragment)
                yield {
                    "event": "mtp_result",
                    "data": {
                        "verb": verb_hint,
                        "target": target_hint,
                        "args": args_hint,
                        "raw_text": raw_hint,
                        "status": "failed",
                        "iteration": iteration,
                    },
                }
                break

            if mtp_result.response_status == "suspend":
                call_params = json.loads(mtp_result.response_content)
                yield {
                    "event": "mtp_result",
                    "data": {
                        "verb": "CALL",
                        "target": call_params["target_alias"],
                        "args": {"task": call_params["task"][:100]},
                        "raw_text": raw_hint,
                        "status": "suspend",
                        "iteration": iteration,
                    },
                }

                ipc_response = await self._handle_call_suspend(
                    frame=frame,
                    mtp_result=mtp_result,
                    assistant_text=gen_result.text,
                    max_iterations=max_iterations,
                    generation_options=generation_options,
                )

                frame.working_history.append(
                    {"role": "assistant", "content": gen_result.text + "⟫"}
                )
                frame.working_history.append({
                    "role": "user",
                    "content": f"[System IPC Return]\n{ipc_response}",
                })
                mtp_commands.append("CALL")

                yield {
                    "event": "mtp_result",
                    "data": {
                        "verb": "CALL",
                        "target": call_params["target_alias"],
                        "args": {"task": call_params["task"][:100]},
                        "raw_text": raw_hint,
                        "status": "success" if '<mtp_response status="success"' in ipc_response else "error",
                        "iteration": iteration,
                    },
                }
                continue

            verb = mtp_result.command.verb.value if mtp_result.command else "UNKNOWN"
            mtp_commands.append(verb)

            if mtp_result.command and mtp_result.command.target:
                target_hint, args_hint, raw_hint = self._extract_command_info(mtp_result.command, raw_hint)

            yield {
                "event": "mtp_result",
                "data": {
                    "verb": verb,
                    "target": target_hint,
                    "args": args_hint,
                    "raw_text": raw_hint,
                    "status": mtp_result.response_status,
                    "iteration": iteration,
                },
            }

            frame.working_history.append({"role": "assistant", "content": gen_result.text + "⟫"})
            frame.working_history.append({
                "role": "user",
                "content": f"[System MTP Execution Result]\n{mtp_result.formatted_response}",
            })

            if frame.is_sub_frame() and mtp_result.command:
                self._try_harvest_alias(frame, mtp_result)

        loop_result = ChatResult(
            final_text="".join(text_segments),
            mtp_iterations=max(0, iteration - 1),
            total_iterations=iteration,
            mtp_commands_executed=mtp_commands,
        )

        yield {"event": "done", "data": loop_result.model_dump()}

    def _extract_mtp_hints(self, gen_result):
        """从生成结果中提取 MTP 提示信息"""
        verb_hint = "UNKNOWN"
        target_hint = ""
        args_hint = {}
        raw_hint = gen_result.mtp_fragment

        try:
            from hivememory.patchouli.mtp.parser import MTPParser
            parsed_hint = MTPParser().complete_and_parse(gen_result.text)
            verb_hint = parsed_hint.verb.value
            if parsed_hint.target.is_wildcard:
                target_hint = "*"
            elif parsed_hint.target.aliases:
                target_hint = ",".join(parsed_hint.target.aliases)
            args_hint = dict(parsed_hint.args)
            raw_hint = parsed_hint.raw_text or raw_hint
        except Exception:
            pass

        return verb_hint, target_hint, args_hint, raw_hint

    def _extract_command_info(self, command, raw_hint):
        """从 MTP 命令中提取信息"""
        if command.target.is_wildcard:
            target_hint = "*"
        elif command.target.aliases:
            target_hint = ",".join(command.target.aliases)
        else:
            target_hint = ""
        args_hint = dict(command.args or {})
        raw_hint = command.raw_text or raw_hint
        return target_hint, args_hint, raw_hint

    async def _handle_call_suspend(
        self,
        frame: ExecutionFrame,
        mtp_result,
        assistant_text: str,
        max_iterations: int,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        处理 CALL 指令的 SUSPEND 状态

        流程:
        1. 解析 CALL 参数
        2. 挂起当前帧
        3. 派生子帧
        4. 递归执行子帧
        5. 恢复父帧
        6. 组装 IPC 返回 payload

        Args:
            frame: 当前帧 (将被挂起)
            mtp_result: MTP 执行结果 (含 CALL 参数)
            assistant_text: LLM 生成的文本
            max_iterations: 最大递归次数
            generation_options: LLM 生成选项

        Returns:
            str: 格式化的 IPC 返回 payload (XML 格式)
        """
        call_params = json.loads(mtp_result.response_content)
        target_alias = call_params["target_alias"]
        task = call_params["task"]
        context_refs = call_params.get("context_refs", [])

        logger.info(
            f"CALL suspend: target={target_alias}, task='{task[:80]}...'"
        )

        self.kernel.frame_scheduler.suspend_frame(frame)

        try:
            sub_frame = await self.kernel.frame_scheduler.fork_sub_frame(
                parent_frame=frame,
                target_alias=target_alias,
                task=task,
                context_refs=context_refs,
            )

            sub_result = await self.execute_frame(
                frame=sub_frame,
                max_iterations=max_iterations,
                generation_options=generation_options,
            )

            self.kernel.frame_scheduler.resume_frame()

            self.kernel.koakuma.set_current_identity(frame.identity)
            self.kernel.koakuma.set_active_profile(frame.agent_profile)
            self.kernel.koakuma.set_current_depth(frame.depth)

            self.kernel.koakuma._current_traces.append(TraceItem(
                action="CALL",
                target=target_alias,
                status="success",
            ))

            return self._assemble_ipc_return(
                sub_result=sub_result,
                harvested_aliases=sub_frame.harvested_aliases,
            )

        except Exception as e:
            logger.error(f"Sub-agent execution failed: {e}", exc_info=True)

            self.kernel.frame_scheduler.resume_frame()
            self.kernel.koakuma.set_current_identity(frame.identity)
            self.kernel.koakuma.set_active_profile(frame.agent_profile)
            self.kernel.koakuma.set_current_depth(frame.depth)

            self.kernel.koakuma._current_traces.append(TraceItem(
                action="CALL",
                target=target_alias,
                status="error",
            ))

            return (
                '<mtp_response status="error" type="ipc_return">\n'
                f'[Sub-Agent Error]: The sub-agent "{target_alias}" encountered '
                f'an error and could not complete the task.\n'
                f'Action: Try a different approach or continue without the sub-agent.\n'
                '</mtp_response>'
            )

    def _assemble_ipc_return(
        self,
        sub_result: ChatResult,
        harvested_aliases: List[str],
    ) -> str:
        """
        组装 IPC 返回 payload (XML 格式)

        将子 Agent 的自然语言回复与自动收割的记忆指针混合打包。

        Args:
            sub_result: 子 Agent 执行结果
            harvested_aliases: 子 Agent 生成的记忆别名列表

        Returns:
            str: 格式化的 IPC 返回 payload
        """
        lines = ['<mtp_response status="success" type="ipc_return">']
        lines.append("[Sub-Agent Reply]:")
        lines.append(sub_result.final_text)

        if harvested_aliases:
            lines.append("")
            lines.append("[Artifacts Generated / Updated]:")
            for alias in harvested_aliases:
                atom = self.kernel.koakuma.atom_cache.get_atom_by_alias(alias)
                if atom and hasattr(atom, 'index') and atom.index.summary:
                    summary = atom.index.summary[:60]
                    lines.append(f"- {alias} ({summary})")
                else:
                    lines.append(f"- {alias}")

        lines.append("</mtp_response>")
        return "\n".join(lines)

    def _try_harvest_alias(self, frame: ExecutionFrame, mtp_result) -> None:
        """
        尝试从 MTP 执行结果中收割别名 (仅子帧)

        当子 Agent 执行 WRITE/UPDATE 时，提取生成的别名并
        添加到帧的 harvested_aliases 列表中。

        Args:
            frame: 当前子帧
            mtp_result: MTP 执行结果
        """
        if not mtp_result.command:
            return

        verb = mtp_result.command.verb
        if verb not in (MTPVerb.WRITE, MTPVerb.UPDATE):
            return

        if verb == MTPVerb.UPDATE:
            alias = mtp_result.command.target.single_alias
            if alias:
                frame.add_harvested_alias(alias)
                logger.debug(f"Harvested UPDATE alias: {alias}")

        elif verb == MTPVerb.WRITE:
            alias = self.kernel.koakuma.get_last_generated_alias()
            if alias:
                frame.add_harvested_alias(alias)
                logger.debug(f"Harvested WRITE alias: {alias}")


__all__ = ["KernelLoopExecutor"]
