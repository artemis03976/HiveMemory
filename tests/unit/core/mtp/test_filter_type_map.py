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


class TestMTPFilterTypeMap:
    """MTP Filter Type Map 测试 (AGENT_PROFILE 支持)"""

    def test_agent_profile_filter(self):
        """type:AGENT_PROFILE 过滤器"""
        from hivememory.core.mtp.parser import _FILTER_TYPE_MAP
        assert "agent_profile" in _FILTER_TYPE_MAP
        assert _FILTER_TYPE_MAP["agent_profile"] == MemoryType.AGENT_PROFILE

    def test_agent_alias_filter(self):
        """type:agent 过滤器 (别名)"""
        from hivememory.core.mtp.parser import _FILTER_TYPE_MAP
        assert "agent" in _FILTER_TYPE_MAP
        assert _FILTER_TYPE_MAP["agent"] == MemoryType.AGENT_PROFILE


# ========== RAG Menu Rendering Tests ==========


