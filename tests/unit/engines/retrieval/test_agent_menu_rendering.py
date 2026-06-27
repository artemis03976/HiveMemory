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
        from hivememory.engines.memory_compiler import (
            CompiledMemoryArtifact,
            MemoryCompiler,
            MemoryCompileTarget,
            MemoryEnvelopeSection,
            MemoryEnvelopeTarget,
        )

        artifacts = [
            CompiledMemoryArtifact(
                target=MemoryCompileTarget.AGENT_PROFILE_MENU,
                text=(
                    '\n<agent_profile alias="coder_doll">\n'
                    "- **角色**: Backend Developer\n"
                    "- **能力特长**: Specializes in Python/FastAPI development\n"
                    "</agent_profile>"
                ),
                source_kind="memory_atom",
                alias="coder_doll",
            ),
            CompiledMemoryArtifact(
                target=MemoryCompileTarget.AGENT_PROFILE_MENU,
                text=(
                    '\n<agent_profile alias="translator_doll">\n'
                    "- **角色**: EN Translator\n"
                    "- **能力特长**: Chinese-English translation expert\n"
                    "</agent_profile>"
                ),
                source_kind="memory_atom",
                alias="translator_doll",
            ),
        ]
        envelope = MemoryCompiler().wrap(
            envelope_target=MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            sections=[
                MemoryEnvelopeSection(
                    kind="agent_profiles",
                    artifacts=artifacts,
                )
            ],
        )
        section = envelope.text

        assert "可用子代理" in section
        assert "coder_doll" in section
        assert "Backend Developer" in section
        assert "translator_doll" in section
        assert "EN Translator" in section

    def test_render_agent_menu_empty(self):
        """无子代理且有记忆时通过 retrieval envelope 渲染占位提示"""
        from hivememory.i18n import get_memory_envelope_text, get_default_language
        from hivememory.engines.memory_compiler import (
            MemoryCompiler,
            MemoryEnvelopeSection,
            MemoryEnvelopeTarget,
        )

        agent_empty_hint = get_memory_envelope_text("retrieval_agent_empty_hint", get_default_language().value)
        envelope = MemoryCompiler().wrap(
            envelope_target=MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            sections=[
                MemoryEnvelopeSection(
                    kind="agent_profiles",
                    empty_text=agent_empty_hint,
                )
            ],
        )
        assert agent_empty_hint in envelope.text

    def test_separate_no_agents(self):
        """无 AGENT_PROFILE 时分离正常"""
        atom = MagicMock(spec=MemoryAtom)
        atom.index = MagicMock()
        atom.index.memory_type = MemoryType.FACT

        regular, agents = self._separate_agent_profiles([atom])
        assert len(regular) == 1
        assert len(agents) == 0

