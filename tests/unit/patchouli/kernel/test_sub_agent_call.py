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
    AgentProfileConfig,
    MemoryAtom,
    MemoryType,
    OMNI_DOLL_PROFILE,
)
from hivememory.patchouli.kernel.runtime.execution_frame import ExecutionFrame
from hivememory.patchouli.mtp import (
    MTPVerb,
    MTPResponseStatus,
    MTPParser,
    MTPCommand,
    MTPResponse,
)
from hivememory.patchouli.protocol.models import ChatResult


# ========== ExecutionFrame Tests ==========


class TestExecutionFrame:
    """ExecutionFrame 数据类测试"""

    def test_create_main_frame(self):
        """depth=0 的主帧"""
        frame = ExecutionFrame(
            process_id="pid_main_1",
            agent_profile=OMNI_DOLL_PROFILE,
            working_history=[{"role": "system", "content": "test"}],
            depth=0,
            topic_id="topic_123",
            identity=Identity(user_id="u1"),
        )
        assert frame.is_main_frame()
        assert not frame.is_sub_frame()
        assert not frame.is_transient()
        assert frame.harvested_aliases == []

    def test_create_sub_frame(self):
        """depth=1 的子帧（瞬态沙盒）"""
        frame = ExecutionFrame(
            process_id="pid_sub_1",
            agent_profile=OMNI_DOLL_PROFILE,
            working_history=[{"role": "user", "content": "write tests"}],
            depth=1,
            topic_id=None,
            parent_frame_id="pid_main_1",
            identity=Identity(user_id="u1"),
        )
        assert not frame.is_main_frame()
        assert frame.is_sub_frame()
        assert frame.is_transient()
        assert frame.parent_frame_id == "pid_main_1"

    def test_harvest_alias(self):
        """别名收割"""
        frame = ExecutionFrame(
            process_id="pid_sub_1",
            agent_profile=OMNI_DOLL_PROFILE,
            working_history=[],
            depth=1,
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
        from hivememory.patchouli.kernel.koakuma import KoakumaRuntime

        koakuma = MagicMock(spec=KoakumaRuntime)
        koakuma._current_depth = depth
        koakuma._active_profile = OMNI_DOLL_PROFILE

        # 绑定真实方法
        import types
        koakuma._handle_call = types.MethodType(
            KoakumaRuntime._handle_call, koakuma
        )
        return koakuma

    def test_call_returns_suspend(self):
        """CALL 返回 SUSPEND 状态"""
        koakuma = self._make_koakuma(depth=0)
        cmd = MagicMock(spec=MTPCommand)
        cmd.target = MagicMock()
        cmd.target.single_alias = "coder_doll"
        cmd.args = {"task": "Write code", "context_refs": '["mem_spec"]'}

        response = koakuma._handle_call(cmd)

        assert response.status == MTPResponseStatus.SUSPEND
        payload = json.loads(response.content)
        assert payload["target_alias"] == "coder_doll"
        assert payload["task"] == "Write code"
        assert payload["context_refs"] == ["mem_spec"]

    def test_call_depth_check_blocks_sub_agent(self):
        """子 Agent (depth=1) 被禁止调用 CALL"""
        koakuma = self._make_koakuma(depth=1)
        cmd = MagicMock(spec=MTPCommand)
        cmd.target = MagicMock()
        cmd.target.single_alias = "another_doll"
        cmd.args = {"task": "Forbidden task"}

        from hivememory.patchouli.mtp.exceptions import PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            koakuma._handle_call(cmd)

    def test_call_missing_task(self):
        """CALL 缺少 task 参数返回 ERROR"""
        koakuma = self._make_koakuma(depth=0)
        cmd = MagicMock(spec=MTPCommand)
        cmd.target = MagicMock()
        cmd.target.single_alias = "coder_doll"
        cmd.args = {}  # no task

        response = koakuma._handle_call(cmd)
        assert response.status == MTPResponseStatus.ERROR

    def test_call_missing_target(self):
        """CALL 缺少 target 返回 ERROR"""
        koakuma = self._make_koakuma(depth=0)
        cmd = MagicMock(spec=MTPCommand)
        cmd.target = MagicMock()
        cmd.target.single_alias = None  # no target
        cmd.args = {"task": "some task"}

        response = koakuma._handle_call(cmd)
        assert response.status == MTPResponseStatus.ERROR


# ========== Koakuma Depth Tracking Tests ==========


class TestKoakumaDepthTracking:
    """Koakuma 深度跟踪测试"""

    def test_set_and_get_depth(self):
        from hivememory.patchouli.kernel.koakuma import KoakumaRuntime

        koakuma = MagicMock(spec=KoakumaRuntime)

        import types
        koakuma.set_current_depth = types.MethodType(
            KoakumaRuntime.set_current_depth, koakuma
        )
        koakuma.get_current_depth = types.MethodType(
            KoakumaRuntime.get_current_depth, koakuma
        )

        koakuma.set_current_depth(0)
        assert koakuma.get_current_depth() == 0

        koakuma.set_current_depth(1)
        assert koakuma.get_current_depth() == 1


# ========== FrameScheduler Tests ==========


class TestFrameScheduler:
    """FrameScheduler 帧调度器测试"""

    def _make_kernel_mock(self):
        kernel = MagicMock()
        kernel.load_agent_profile = MagicMock(return_value=OMNI_DOLL_PROFILE)
        kernel.get_agent_persona = MagicMock(return_value="I am a test agent")
        kernel.get_mtp_prompt = MagicMock(return_value="### MTP Instructions\n## CALL\nCALL instructions here\n## READ\nREAD instructions")
        kernel.check_storage_health = MagicMock(return_value=True)
        kernel.config = MagicMock()
        kernel.config.koakuma.mtp_prompt.language = "zh"
        kernel.koakuma = MagicMock()
        kernel.koakuma._current_identity = Identity(user_id="u1")
        kernel.koakuma.atom_cache = MagicMock()
        kernel.koakuma.atom_cache.get_atom_by_alias = MagicMock(return_value=None)
        kernel.storage = MagicMock()
        kernel.storage.get_memory_by_alias = MagicMock(return_value=None)
        return kernel

    def test_create_main_frame(self):
        """创建主帧"""
        from hivememory.patchouli.kernel.runtime.frame_scheduler import FrameScheduler

        kernel = self._make_kernel_mock()
        scheduler = FrameScheduler(kernel)

        frame = scheduler.create_main_frame(
            agent_profile=OMNI_DOLL_PROFILE,
            messages=[{"role": "system", "content": "hi"}],
            topic_id="t1",
            identity=Identity(user_id="u1"),
        )

        assert frame.depth == 0
        assert frame.topic_id == "t1"
        assert frame.is_main_frame()

    def test_suspend_resume(self):
        """帧挂起/恢复"""
        from hivememory.patchouli.kernel.runtime.frame_scheduler import FrameScheduler

        kernel = self._make_kernel_mock()
        scheduler = FrameScheduler(kernel)

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
        from hivememory.patchouli.kernel.runtime.frame_scheduler import FrameScheduler

        kernel = self._make_kernel_mock()
        scheduler = FrameScheduler(kernel)

        assert scheduler.resume_frame() is None

    @pytest.mark.asyncio
    async def test_fork_sub_frame(self):
        """派生子帧"""
        from hivememory.patchouli.kernel.runtime.frame_scheduler import FrameScheduler

        kernel = self._make_kernel_mock()
        scheduler = FrameScheduler(kernel)

        main_frame = scheduler.create_main_frame(
            agent_profile=OMNI_DOLL_PROFILE,
            messages=[],
            topic_id="t1",
            identity=Identity(user_id="u1"),
        )

        sub_frame = await scheduler.fork_sub_frame(
            parent_frame=main_frame,
            target_alias="tester_doll",
            task="Write unit tests",
            context_refs=[],
        )

        assert sub_frame.depth == 1
        assert sub_frame.topic_id is None
        assert sub_frame.parent_frame_id == main_frame.process_id
        assert sub_frame.is_sub_frame()
        assert sub_frame.is_transient()
        assert len(sub_frame.working_history) == 2  # system + user

    def test_strip_call_from_prompt(self):
        """从 MTP prompt 中移除 CALL 教学"""
        from hivememory.patchouli.kernel.runtime.frame_scheduler import FrameScheduler

        kernel = self._make_kernel_mock()
        scheduler = FrameScheduler(kernel)

        prompt = "### MTP Instructions\n## CALL\nCALL instructions here\n## READ\nREAD instructions"
        stripped = scheduler._strip_call_from_prompt(prompt)

        assert "CALL instructions here" not in stripped
        assert "READ instructions" in stripped


# ========== IPC Return Assembly Tests ==========


class TestIPCReturnAssembly:
    """IPC 返回 payload 组装测试"""

    def test_assemble_success_no_artifacts(self):
        """成功返回，无 artifacts"""
        from hivememory.patchouli.system import PatchouliSystem

        sys = MagicMock(spec=PatchouliSystem)
        import types
        sys._assemble_ipc_return = types.MethodType(
            PatchouliSystem._assemble_ipc_return, sys
        )

        result = ChatResult(final_text="Task completed successfully.")
        payload = sys._assemble_ipc_return(result, [])

        assert '<mtp_response status="success" type="ipc_return">' in payload
        assert "[Sub-Agent Reply]:" in payload
        assert "Task completed successfully." in payload
        assert "[Artifacts" not in payload
        assert "</mtp_response>" in payload

    def test_assemble_success_with_artifacts(self):
        """成功返回，含 artifacts"""
        from hivememory.patchouli.system import PatchouliSystem

        sys = MagicMock(spec=PatchouliSystem)
        sys.kernel = MagicMock()
        sys.kernel.koakuma.atom_cache.get_atom_by_alias = MagicMock(return_value=None)

        import types
        sys._assemble_ipc_return = types.MethodType(
            PatchouliSystem._assemble_ipc_return, sys
        )

        result = ChatResult(final_text="Code written.")
        payload = sys._assemble_ipc_return(result, ["mem_code_1", "mem_code_2"])

        assert "[Artifacts Generated / Updated]:" in payload
        assert "- mem_code_1" in payload
        assert "- mem_code_2" in payload


# ========== MTP Filter Type Map Tests ==========


class TestMTPFilterTypeMap:
    """MTP Filter Type Map 测试 (AGENT_PROFILE 支持)"""

    def test_agent_profile_filter(self):
        """type:AGENT_PROFILE 过滤器"""
        from hivememory.patchouli.mtp.parser import _FILTER_TYPE_MAP
        assert "agent_profile" in _FILTER_TYPE_MAP
        assert _FILTER_TYPE_MAP["agent_profile"] == MemoryType.AGENT_PROFILE

    def test_agent_alias_filter(self):
        """type:agent 过滤器 (别名)"""
        from hivememory.patchouli.mtp.parser import _FILTER_TYPE_MAP
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
        """渲染子代理服务菜单"""
        from hivememory.engines.retrieval.renderer import _render_agent_menu

        agent1 = MagicMock(spec=MemoryAtom)
        agent1.index = MagicMock()
        agent1.index.alias = "coder_doll"
        agent1.index.title = "Backend Developer"
        agent1.index.summary = "Specializes in Python/FastAPI development"

        agent2 = MagicMock(spec=MemoryAtom)
        agent2.index = MagicMock()
        agent2.index.alias = "translator_doll"
        agent2.index.title = "EN Translator"
        agent2.index.summary = "Chinese-English translation expert"

        menu = _render_agent_menu([agent1, agent2])

        assert "[Available Sub-Agents (Ready to CALL)]" in menu
        assert '[ID: coder_doll]' in menu
        assert '"Backend Developer"' in menu
        assert '[ID: translator_doll]' in menu
        assert '"EN Translator"' in menu

    def test_render_agent_menu_empty(self):
        """空列表返回空字符串"""
        from hivememory.engines.retrieval.renderer import _render_agent_menu
        assert _render_agent_menu([]) == ""

    def test_separate_no_agents(self):
        """无 AGENT_PROFILE 时分离正常"""
        from hivememory.engines.retrieval.renderer import _separate_agent_profiles

        atom = MagicMock(spec=MemoryAtom)
        atom.index = MagicMock()
        atom.index.memory_type = MemoryType.FACT

        regular, agents = _separate_agent_profiles([atom])
        assert len(regular) == 1
        assert len(agents) == 0
