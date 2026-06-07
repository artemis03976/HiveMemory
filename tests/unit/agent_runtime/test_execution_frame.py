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


class TestExecutionFrame:
    """ExecutionFrame 数据类测试"""

    def test_create_main_frame(self):
        """depth=0 的主帧"""
        frame = ExecutionFrame(
            runtime_scope=RuntimeScope(frame_id="frame_main_1", depth=0),
            agent_profile=OMNI_DOLL_PROFILE,
            working_history=[{"role": "system", "content": "test"}],
            topic_id="topic_123",
            identity=Identity(user_id="u1"),
        )
        assert frame.is_main_frame()
        assert not frame.is_sub_frame()
        assert not frame.is_transient()
        assert frame.harvested_aliases == []
        assert frame.runtime_scope.run_id == ""

    def test_create_sub_frame(self):
        """depth=1 的子帧（瞬态沙盒）"""
        frame = ExecutionFrame(
            runtime_scope=RuntimeScope(
                frame_id="frame_sub_1",
                parent_frame_id="frame_main_1",
                depth=1,
            ),
            agent_profile=OMNI_DOLL_PROFILE,
            working_history=[{"role": "user", "content": "write tests"}],
            topic_id=None,
            identity=Identity(user_id="u1"),
        )
        assert not frame.is_main_frame()
        assert frame.is_sub_frame()
        assert frame.is_transient()
        assert frame.runtime_scope.parent_frame_id == "frame_main_1"
        assert frame.runtime_scope.run_id == ""

    def test_harvest_alias(self):
        """别名收割"""
        frame = ExecutionFrame(
            runtime_scope=RuntimeScope(frame_id="frame_sub_1", depth=1),
            agent_profile=OMNI_DOLL_PROFILE,
            working_history=[],
            topic_id=None,
            identity=Identity(user_id="u1"),
        )
        frame.add_harvested_alias("mem_code_1")
        frame.add_harvested_alias("mem_code_2")
        frame.add_harvested_alias("mem_code_1")  # 去重
        assert frame.harvested_aliases == ["mem_code_1", "mem_code_2"]


# ========== MTP CALL Parsing Tests ==========


