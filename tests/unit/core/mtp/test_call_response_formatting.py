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


class TestMTPCallResponseFormatting:
    """CALL response payload 渲染测试"""

    def test_assemble_success_no_artifacts(self):
        """成功返回，无 artifacts"""
        payload = MTPFormatter.format_call_response(
            MTPCallResponse(
                status=MTPResponseStatus.SUCCESS,
                agent_alias="coder_doll",
                reply="Task completed successfully.",
            ),
            "en",
        )

        assert payload.startswith("[System MTP Call Response]\n")
        assert '<mtp_response status="success" type="call_response">' in payload
        assert "[Sub-Agent Reply]:" in payload
        assert "Task completed successfully." in payload
        assert "[Artifacts" not in payload
        assert "</mtp_response>" in payload

    def test_assemble_success_with_artifacts(self):
        """成功返回，含 artifacts"""
        payload = MTPFormatter.format_call_response(
            MTPCallResponse(
                status=MTPResponseStatus.SUCCESS,
                agent_alias="coder_doll",
                reply="Code written.",
                artifact_aliases=["mem_code_1", "mem_code_2"],
            ),
            "en",
        )

        assert "[Artifacts Generated / Updated]:" in payload
        assert "- mem_code_1 (pending, readable now)" in payload
        assert "- mem_code_2 (pending, readable now)" in payload

    def test_assemble_error(self):
        """错误返回使用结构化 error 渲染"""
        payload = MTPFormatter.format_call_response(
            MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias="coder_doll",
                error=MTPErrorInfo(
                    code="mtp.call_response.sub_agent_error",
                    message_key="mtp.call_response.sub_agent_error",
                    severity=MTPErrorSeverity.SYSTEM_FAULT,
                    params={"agent_alias": "coder_doll"},
                ),
            ),
            "en",
        )

        assert '<mtp_response status="error" type="call_response">' in payload
        assert '<error code="mtp.call_response.sub_agent_error" severity="system_fault">' in payload
        assert "[Sub-Agent Error]" in payload
        assert "coder_doll" in payload


# ========== MTP Filter Type Map Tests ==========


