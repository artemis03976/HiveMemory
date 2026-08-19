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
from tests.helpers.memory import make_memory_metadata


# ========== ExecutionFrame Tests ==========


class TestRAGMenuRendering:
    """RAG 菜单渲染测试 (AGENT_PROFILE 分离)"""

    def test_render_agent_menu(self):
        """通过 MemoryCompiler envelope 渲染子代理区域"""
        from hivememory.core.models import IndexLayer, MetaData, PayloadLayer
        from hivememory.engines.memory_compiler import (
            MemoryCompiler,
            MemoryEnvelopeTarget,
        )

        agents = [
            MemoryAtom(
                meta=make_memory_metadata(source_agent_id="system", user_id="u1"),
                index=IndexLayer(
                    title="coder_doll",
                    summary="Backend Developer",
                    memory_type=MemoryType.AGENT_PROFILE,
                    alias="coder_doll",
                ),
                payload=PayloadLayer(content="Specializes in Python/FastAPI development"),
            ),
            MemoryAtom(
                meta=make_memory_metadata(source_agent_id="system", user_id="u1"),
                index=IndexLayer(
                    title="translator_doll",
                    summary="EN Translator",
                    memory_type=MemoryType.AGENT_PROFILE,
                    alias="translator_doll",
                ),
                payload=PayloadLayer(content="Chinese-English translation expert"),
            ),
        ]
        envelope = MemoryCompiler().compile(agents, MemoryEnvelopeTarget.RETRIEVAL_CONTEXT)
        section = envelope.text

        assert "可用子代理" in section
        assert "coder_doll" in section
        assert "Backend Developer" in section
        assert "translator_doll" in section
        assert "EN Translator" in section

    def test_render_agent_menu_empty(self):
        """无子代理 section 时不渲染子代理区域"""
        from hivememory.i18n import get_memory_envelope_text, get_default_language
        from hivememory.core.models import IndexLayer, MetaData, PayloadLayer
        from hivememory.engines.memory_compiler import (
            MemoryCompiler,
            MemoryEnvelopeTarget,
        )

        agent_empty_hint = get_memory_envelope_text("retrieval_agent_empty_hint", get_default_language().value)
        atom = MemoryAtom(
            meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
            index=IndexLayer(
                title="regular memory",
                summary="regular memory summary",
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="regular memory content"),
        )
        envelope = MemoryCompiler().compile(
            [atom],
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        )
        assert agent_empty_hint not in envelope.text
        assert "### 可用子代理" not in envelope.text

