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


class TestKoakumaHandleCall:
    """Koakuma _handle_call() 测试"""

    def _make_koakuma(self, depth=0):
        from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
        from hivememory.agent_runtime.models import MTPExecutionContext

        koakuma = MagicMock(spec=KoakumaRuntime)
        koakuma.context = MTPExecutionContext(
            identity=Identity(user_id="u1", agent_id="omni_doll"),
            agent_profile=OMNI_DOLL_PROFILE,
            runtime_scope=RuntimeScope(depth=depth),
        )

        # 绑定真实方法
        import types
        koakuma._handle_call = types.MethodType(
            KoakumaRuntime._handle_call, koakuma
        )
        return koakuma

    @pytest.mark.asyncio
    async def test_call_returns_suspend(self):
        """CALL 返回 SUSPEND 状态"""
        koakuma = self._make_koakuma(depth=0)
        cmd = MagicMock(spec=MTPCommand)
        cmd.target = MagicMock()
        cmd.target.single_alias = "coder_doll"
        cmd.args = {"task": "Write code", "context_refs": '["mem_spec"]'}

        response = await koakuma._handle_call(cmd, context=koakuma.context)

        assert response.status == MTPResponseStatus.SUSPEND
        assert response.content == ""
        assert response.call_request == MTPCallRequest(
            target_alias="coder_doll",
            task="Write code",
            context_refs=["mem_spec"],
        )

    @pytest.mark.asyncio
    async def test_call_depth_check_blocks_sub_agent(self):
        """子 Agent (depth=1) 被禁止调用 CALL"""
        koakuma = self._make_koakuma(depth=1)
        cmd = MagicMock(spec=MTPCommand)
        cmd.target = MagicMock()
        cmd.target.single_alias = "another_doll"
        cmd.args = {"task": "Forbidden task"}

        from hivememory.core.mtp.exceptions import PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            await koakuma._handle_call(cmd, context=koakuma.context)

    @pytest.mark.asyncio
    async def test_call_missing_task(self):
        """CALL 缺少 task 参数返回 ERROR"""
        from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
        from hivememory.agent_runtime.models import MTPExecutionContext

        koakuma = self._make_koakuma(depth=0)
        koakuma._check_verb_permission = MagicMock()
        koakuma._route_and_execute = KoakumaRuntime._route_and_execute.__get__(koakuma)
        response = await koakuma._route_and_execute(
            MTPParser().parse('⟪ CALL | coder_doll | ⟫'),
            MTPExecutionContext(
                identity=Identity(user_id="u1", agent_id="omni_doll"),
                agent_profile=OMNI_DOLL_PROFILE,
                runtime_scope=RuntimeScope(depth=0),
            ),
        )
        assert response.status == MTPResponseStatus.ERROR
        assert response.error is not None
        assert response.error.code == "mtp.argument.invalid"

    @pytest.mark.asyncio
    async def test_call_missing_target(self):
        """CALL 缺少 target 返回 ERROR"""
        from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
        from hivememory.agent_runtime.models import MTPExecutionContext

        koakuma = self._make_koakuma(depth=0)
        koakuma._check_verb_permission = MagicMock()
        koakuma._route_and_execute = KoakumaRuntime._route_and_execute.__get__(koakuma)
        response = await koakuma._route_and_execute(
            MTPParser().parse('⟪ CALL | | task="some task" ⟫'),
            MTPExecutionContext(
                identity=Identity(user_id="u1", agent_id="omni_doll"),
                agent_profile=OMNI_DOLL_PROFILE,
                runtime_scope=RuntimeScope(depth=0),
            ),
        )
        assert response.status == MTPResponseStatus.ERROR
        assert response.error is not None
        assert response.error.code == "mtp.argument.invalid"


# ========== FrameScheduler Tests ==========


