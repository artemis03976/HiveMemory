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


class TestRAGMenuRendering:
    """RAG 菜单渲染测试 (AGENT_PROFILE 分离)"""

    def _separate_agent_profiles(self, memories):
        regular, agents = [], []
        for m in memories:
            if hasattr(m, 'index') and hasattr(m.index, 'memory_type') and m.index.memory_type == MemoryType.AGENT_PROFILE:
                agents.append(m)
            else:
                regular.append(m)
        return regular, agents

    def test_separate_agent_profiles(self):
        regular_atom = MagicMock(spec=MemoryAtom)
        regular_atom.index = MagicMock()
        regular_atom.index.memory_type = MemoryType.FACT

        agent_atom = MagicMock(spec=MemoryAtom)
        agent_atom.index = MagicMock()
        agent_atom.index.memory_type = MemoryType.AGENT_PROFILE

        regular, agents = self._separate_agent_profiles([regular_atom, agent_atom])

        assert len(regular) == 1
        assert len(agents) == 1

    def test_render_agent_menu(self):
        """通过 MemoryCompiler envelope 渲染子代理区域"""
        from hivememory.core.models import IndexLayer, MetaData, PayloadLayer
        from hivememory.engines.memory_compiler import (
            MemoryCompiler,
            MemoryEnvelopeTarget,
        )

        agents = [
            MemoryAtom(
                meta=MetaData(source_agent_id="system", user_id="u1"),
                index=IndexLayer(
                    title="coder_doll",
                    summary="Backend Developer",
                    memory_type=MemoryType.AGENT_PROFILE,
                    alias="coder_doll",
                ),
                payload=PayloadLayer(content="Specializes in Python/FastAPI development"),
            ),
            MemoryAtom(
                meta=MetaData(source_agent_id="system", user_id="u1"),
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
            meta=MetaData(source_agent_id="a1", user_id="u1"),
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

    def test_separate_no_agents(self):
        """无 AGENT_PROFILE 时分离正常"""
        atom = MagicMock(spec=MemoryAtom)
        atom.index = MagicMock()
        atom.index.memory_type = MemoryType.FACT

        regular, agents = self._separate_agent_profiles([atom])
        assert len(regular) == 1
        assert len(agents) == 0

