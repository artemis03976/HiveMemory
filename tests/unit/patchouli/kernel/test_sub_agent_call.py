"""
Phase 2 多智能体子代理调用集成测试

测试覆盖:
    1. ExecutionFrame 数据类
    2. FrameScheduler 帧调度器
    3. MTP CALL 指令解析与路由
    4. Koakuma _handle_call() 深度限制
    5. IPC 返回 payload 组装
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
    MTPVerb,
    MTPResponseStatus,
    MTPParser,
    MTPCommand,
    MTPCallRequest,
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


# ========== IPC Return Assembly Tests ==========


class TestIPCReturnAssembly:
    """IPC 返回 payload 组装测试"""

    def _make_orchestrator(self):
        from hivememory.alice.runtime.orchestrator import AgentOrchestrator
        from unittest.mock import MagicMock
        return AgentOrchestrator(
            agent_runtime=MagicMock(),
            frame_scheduler=MagicMock(),
            agent_profile_resolver=MagicMock(),
            alias_resolver=MagicMock(),
        )

    def test_assemble_success_no_artifacts(self):
        """成功返回，无 artifacts"""
        orchestrator = self._make_orchestrator()
        payload = orchestrator._assemble_ipc_return("Task completed successfully.", [])

        assert '<mtp_response status="success" type="ipc_return">' in payload
        assert "[Sub-Agent Reply]:" in payload
        assert "Task completed successfully." in payload
        assert "[Artifacts" not in payload
        assert "</mtp_response>" in payload

    def test_assemble_success_with_artifacts(self):
        """成功返回，含 artifacts"""
        orchestrator = self._make_orchestrator()
        payload = orchestrator._assemble_ipc_return("Code written.", ["mem_code_1", "mem_code_2"])

        assert "[Artifacts Generated / Updated]:" in payload
        assert "- mem_code_1" in payload
        assert "- mem_code_2" in payload


# ========== MTP Filter Type Map Tests ==========


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


class TestRAGMenuRendering:
    """RAG 菜单渲染测试 (AGENT_PROFILE 分离)"""

    def test_separate_agent_profiles(self):
        """分离 AGENT_PROFILE 和普通记忆"""
        from hivememory.engines.retrieval.renderer import _separate_agent_profiles

        regular_atom = MagicMock(spec=MemoryAtom)
        regular_atom.index = MagicMock()
        regular_atom.index.memory_type = MemoryType.FACT

        agent_atom = MagicMock(spec=MemoryAtom)
        agent_atom.index = MagicMock()
        agent_atom.index.memory_type = MemoryType.AGENT_PROFILE

        regular, agents = _separate_agent_profiles([regular_atom, agent_atom])

        assert len(regular) == 1
        assert len(agents) == 1
        assert regular[0] == regular_atom
        assert agents[0] == agent_atom

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
        from hivememory.engines.retrieval.renderer import _AGENT_EMPTY_HINT
        from hivememory.engines.memory_compiler import (
            MemoryCompiler,
            MemoryEnvelopeSection,
            MemoryEnvelopeTarget,
        )

        envelope = MemoryCompiler().wrap(
            envelope_target=MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            sections=[
                MemoryEnvelopeSection(
                    kind="agent_profiles",
                    empty_text=_AGENT_EMPTY_HINT,
                )
            ],
        )
        assert _AGENT_EMPTY_HINT in envelope.text

    def test_render_agent_section_empty_no_memories(self):
        """无子代理且无记忆时 retrieval renderer 返回整体空结果提示"""
        from hivememory.engines.retrieval.renderer import (
            _AGENT_EMPTY_HINT,
            _EMPTY_CONTEXT_NOTICE,
            FullContextRenderer,
        )
        from hivememory.system.config import FullRendererConfig

        rendered = FullContextRenderer(FullRendererConfig()).render([])
        assert rendered == _EMPTY_CONTEXT_NOTICE
        assert _AGENT_EMPTY_HINT not in rendered

    def test_separate_no_agents(self):
        """无 AGENT_PROFILE 时分离正常"""
        from hivememory.engines.retrieval.renderer import _separate_agent_profiles

        atom = MagicMock(spec=MemoryAtom)
        atom.index = MagicMock()
        atom.index.memory_type = MemoryType.FACT

        regular, agents = _separate_agent_profiles([atom])
        assert len(regular) == 1
        assert len(agents) == 0

