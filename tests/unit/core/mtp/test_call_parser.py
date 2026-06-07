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


class TestMTPCallParsing:
    """MTP CALL 指令解析测试"""

    def test_parse_call_basic(self):
        """基本 CALL 指令解析"""
        parser = MTPParser()
        cmd = parser.parse('⟪ CALL | coder_doll | task="Write unit tests" ⟫')
        assert cmd.verb == MTPVerb.CALL
        assert cmd.target.single_alias == "coder_doll"
        assert cmd.args.get("task") == "Write unit tests"

    def test_parse_call_with_context_refs(self):
        """带 context_refs 的 CALL 指令"""
        parser = MTPParser()
        cmd = parser.parse(
            '⟪ CALL | backend_doll | task="实现接口" context_refs=["mem_api_spec", "mem_db_schema"] ⟫'
        )
        assert cmd.verb == MTPVerb.CALL
        assert cmd.target.single_alias == "backend_doll"
        assert cmd.args.get("task") == "实现接口"
        refs = json.loads(cmd.args["context_refs"])
        assert refs == ["mem_api_spec", "mem_db_schema"]

    def test_parse_call_without_context_refs(self):
        """不带 context_refs 的 CALL 指令"""
        parser = MTPParser()
        cmd = parser.parse('⟪ CALL | tester_doll | task="Run all tests" ⟫')
        assert cmd.verb == MTPVerb.CALL
        assert "context_refs" not in cmd.args

    def test_parse_list_args_single_item(self):
        """列表参数 — 单个元素"""
        parser = MTPParser()
        cmd = parser.parse(
            '⟪ CALL | coder | task="test" context_refs=["mem_spec"] ⟫'
        )
        refs = json.loads(cmd.args["context_refs"])
        assert refs == ["mem_spec"]


# ========== Koakuma CALL Handler Tests ==========


