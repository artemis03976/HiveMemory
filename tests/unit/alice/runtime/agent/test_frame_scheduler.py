"""
Phase 2 多智能体子代理调用集成测试

测试覆盖:
    1. ExecutionFrame 数据类
    2. FrameScheduler 帧调度器
    3. MTP CALL 指令解析与路由
    4. Koakuma _handle_call() 深度限制
    5. CALL response payload 渲染
    6. 星型拓扑约束 (depth=1)
    7. RAG 菜单渲染 (AGENT_PROFILE 分离)

作者: HiveMemory Team
版本: 1.0
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hivememory.core.models import (
    Identity,
    AgentProfile,
    MemoryAtom,
    MemoryType,
    OMNI_DOLL_PROFILE,
    RuntimeScope,
)
from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import KoakumaConfig
from hivememory.core.mtp import (
    MTPCallResponse,
    MTPVerb,
    MTPResponseStatus,
    MTPParser,
    MTPCommand,
    MTPCallRequest,
    MTPErrorInfo,
    MTPErrorSeverity,
    MTPFormatter,
    MTPResponse,
)


# ========== ExecutionFrame Tests ==========


class TestFrameScheduler:
    """FrameScheduler 帧调度器测试"""

    def _make_kernel_mock(self):
        kernel = MagicMock()
        kernel.config = MagicMock()
        kernel.prompt_assembler = AgentPromptAssembler(KoakumaConfig())
        kernel._global_bus = AsyncMock()
        kernel._global_bus.request = AsyncMock(return_value=None)
        return kernel

    def test_create_main_frame(self):
        """创建主帧"""
        from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler

        kernel = self._make_kernel_mock()
        scheduler = FrameScheduler(kernel.prompt_assembler)

        frame = scheduler.create_main_frame(
            agent_profile=OMNI_DOLL_PROFILE,
            messages=[{"role": "system", "content": "hi"}],
            topic_id="t1",
            identity=Identity(user_id="u1"),
        )

        assert frame.runtime_scope.depth == 0
        assert frame.topic_id == "t1"
        assert frame.is_main_frame()
        assert frame.runtime_scope.run_id.startswith("run_")
        assert frame.runtime_scope.frame_id.startswith("frame_main_")

    def test_suspend_resume(self):
        """帧挂起/恢复"""
        from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler

        kernel = self._make_kernel_mock()
        scheduler = FrameScheduler(kernel.prompt_assembler)

        frame = scheduler.create_main_frame(
            agent_profile=OMNI_DOLL_PROFILE,
            messages=[],
            topic_id="t1",
            identity=Identity(user_id="u1"),
        )

        scheduler.suspend_frame(frame)
        assert scheduler.get_current_depth() == 1

        resumed = scheduler.resume_frame()
        assert resumed is frame
        assert scheduler.get_current_depth() == 0

    def test_resume_empty_stack_returns_none(self):
        """空栈恢复返回 None"""
        from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler

        kernel = self._make_kernel_mock()
        scheduler = FrameScheduler(kernel.prompt_assembler)

        assert scheduler.resume_frame() is None

    @pytest.mark.asyncio
    async def test_fork_sub_frame(self):
        """派生子帧"""
        from hivememory.alice.runtime.agent.frame_scheduler import FrameScheduler

        kernel = self._make_kernel_mock()
        scheduler = FrameScheduler(kernel.prompt_assembler)

        main_frame = scheduler.create_main_frame(
            agent_profile=OMNI_DOLL_PROFILE,
            messages=[],
            topic_id="t1",
            identity=Identity(user_id="u1"),
        )

        sub_frame = await scheduler.fork_sub_frame(
            parent_frame=main_frame,
            agent_profile=OMNI_DOLL_PROFILE,
            task="Write unit tests",
            shared_context="",
        )

        assert sub_frame.runtime_scope.depth == 1
        assert sub_frame.topic_id is None
        assert sub_frame.runtime_scope.frame_id.startswith("frame_sub_")
        assert sub_frame.runtime_scope.parent_frame_id == main_frame.runtime_scope.frame_id
        assert sub_frame.runtime_scope.run_id == main_frame.runtime_scope.run_id
        assert sub_frame.is_sub_frame()
        assert sub_frame.is_transient()
        assert len(sub_frame.working_history) == 2  # system + user

    def test_sub_agent_prompt_disables_call(self):
        """Sub-agent prompt should not render CALL instructions."""
        kernel = self._make_kernel_mock()

        messages = kernel.prompt_assembler.build_sub_agent_messages(
            profile=OMNI_DOLL_PROFILE,
            task="Write unit tests",
            depth=1,
        )

        assert "CALL" not in messages[0]["content"]
        assert "READ" in messages[0]["content"]


# ========== CALL Response Formatting Tests ==========


