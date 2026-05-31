"""
RUN 指令执行链路测试

验证 MTP RUN 指令从 Koakuma._handle_run 的完整链路。

测试覆盖:
    1. Target 校验 (通配符/列表/空 target 拒绝)
    2. Level 0 内核工具快速路径 (sys_clock, sys_python_repl)
    3. Level 1 用户态工具路径 (统一原子缓存 + L2 冷检索 + 沙箱执行)

与 test_syscall_chain.py 的区别:
    - test_syscall_chain.py: 聚焦 syscall 函数本身的行为正确性
    - 本文件: 聚焦 RUN 指令的 _handle_run 链路 (target 校验、快速路径分发、用户态路径)

作者: HiveMemory Team
版本: 3.0
"""

import asyncio
import pytest
from uuid import uuid4
from unittest.mock import MagicMock

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
)
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.alice.runtime.models import MTPExecutionContext
from hivememory.alice.runtime.pending_atom_state import PendingAtomResolution
from hivememory.engines.generation.interfaces import DuplicateDecision
from hivememory.engines.generation.models import PendingAtomSettlement
from hivememory.system.config import KoakumaConfig


# ========== Helpers ==========

def _make_code_memory(
    code: str = "print('hello')",
    alias: str = "tool_test",
    mem_id=None,
) -> MemoryAtom:
    """创建 CODE_SNIPPET 类型的记忆原子"""
    return MemoryAtom(
        id=mem_id or uuid4(),
        meta=MetaData(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title="Test Tool",
            summary="A test code snippet tool",
            tags=["test", "tool"],
            memory_type=MemoryType.CODE_SNIPPET,
            alias=alias,
        ),
        payload=PayloadLayer(content=code),
    )


def _make_fact_memory(mem_id=None, alias: str = "fact_not_tool") -> MemoryAtom:
    """创建 FACT 类型的记忆原子 (不可执行)"""
    return MemoryAtom(
        id=mem_id or uuid4(),
        meta=MetaData(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title="Test Fact",
            summary="A test fact memory",
            tags=["test"],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content="This is a fact, not code."),
    )


# ========== Fixtures ==========

@pytest.fixture
def koakuma() -> KoakumaRuntime:
    from .conftest import make_koakuma_runtime, make_mock_bus
    bus = make_mock_bus()
    return make_koakuma_runtime(bus, KoakumaConfig())


def _execute_mtp(koakuma: KoakumaRuntime, text: str, context=None):
    return asyncio.run(koakuma.execute_mtp(text, context=context))


def _intercept_and_execute(koakuma: KoakumaRuntime, assistant_text: str, context=None):
    return asyncio.run(koakuma.intercept_and_execute(assistant_text, context=context))


# ========== Test 1: Target Validation ==========

class TestRunTargetValidation:
    """RUN 指令 target 校验"""

    def test_wildcard_rejected(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | * | ⟫')
        assert not result.success
        assert "single tool alias" in result.response_content.lower() or "requires" in result.response_content.lower()

    def test_list_target_rejected(self, koakuma):
        """列表 target 不支持 (single_alias 返回 None)"""
        result = _execute_mtp(koakuma, '⟪ RUN | [tool_a, tool_b] | ⟫')
        assert not result.success

    def test_empty_target(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | | ⟫')
        assert not result.success


# ========== Test 2: Kernel Fast Path ==========

class TestRunKernelFastPath:
    """Level 0 内核工具快速路径"""

    def test_sys_clock_default(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | sys_clock | ⟫')
        assert result.success
        assert "UTC" in result.response_content
        assert koakuma._bus._memory_citations == []

    def test_sys_clock_iso(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | sys_clock | format="iso" ⟫')
        assert result.success
        assert "T" in result.response_content

    def test_sys_clock_date(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | sys_clock | format="date" ⟫')
        assert result.success
        # YYYY-MM-DD format
        assert "-" in result.response_content
        assert len(result.response_content.strip()) == 10 or "20" in result.response_content

    def test_sys_python_repl_calculation(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | sys_python_repl | code="print(6 * 7)" ⟫')
        assert result.success
        assert "42" in result.response_content

    def test_sys_python_repl_import_blocked(self, koakuma):
        result = _execute_mtp(koakuma, '⟪ RUN | sys_python_repl | code="import os" ⟫')
        assert result.success  # handler 返回 error string, 不是异常
        assert "Error" in result.response_content
        assert "import" in result.response_content.lower()

    def test_sys_python_repl_multiline(self, koakuma):
        # 使用反引号语法支持多行代码 (Section 2.1)
        result = _execute_mtp(koakuma, '⟪ RUN | sys_python_repl | code=`x = 10\ny = 20\nprint(x + y)` ⟫')
        assert result.success
        assert "30" in result.response_content


# ========== Test 3: User Tool Path (Level 1) ==========

class TestRunUserToolPath:
    """Level 1 用户态工具慢速路径"""

    def test_unknown_tool_not_found(self, koakuma):
        """L1+L2 均未命中，返回 not found"""
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = None
        result = _execute_mtp(koakuma, '⟪ RUN | nonexistent_tool | ⟫')
        assert not result.success
        assert "not found" in result.response_content.lower()

    def test_unknown_tool_suggests_search(self, koakuma):
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = None
        result = _execute_mtp(koakuma, '⟪ RUN | my_custom_tool | ⟫')
        assert not result.success
        assert "SEARCH" in result.response_content

    def test_l2_route_failure_returns_infra_error(self, koakuma):
        koakuma._bus._mock_storage.get_memory_by_alias.side_effect = KeyError(
            "AsyncSystemBus: route 'memory.retrieve_by_aliases' not registered"
        )

        result = _execute_mtp(koakuma, '⟪ RUN | my_custom_tool | ⟫')

        assert not result.success
        assert "Service Unavailable" in result.response_content

    def test_l2_hit_executes_code_snippet(self, koakuma):
        """L2 命中 CODE_SNIPPET，沙箱执行成功"""
        mem = _make_code_memory(code="print('tool output')", alias="tool_greet")
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        result = _execute_mtp(koakuma, '⟪ RUN | tool_greet | ⟫')

        assert result.success
        assert "tool output" in result.response_content
        assert koakuma._bus._memory_citations == [
            {"memory_id": mem.id, "source": "mtp.run"}
        ]

    def test_l1_alias_hit_executes(self, koakuma):
        """L1 别名命中 → 加载 → 执行"""
        mem = _make_code_memory(code="print('from l1')", alias="tool_l1")
        koakuma.atom_cache.ingest_atom(mem)

        result = _execute_mtp(koakuma, '⟪ RUN | tool_l1 | ⟫')

        assert result.success
        assert "from l1" in result.response_content
        assert koakuma._bus._memory_citations == [
            {"memory_id": mem.id, "source": "mtp.run"}
        ]

    def test_cache_hit_skips_qdrant(self, koakuma):
        """第二次调用走 LRU 缓存，不查 Qdrant"""
        mem = _make_code_memory(code="print('cached')", alias="tool_cached")
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        # 第一次调用: L2 命中 → 加载 → 缓存
        result1 = _execute_mtp(koakuma, '⟪ RUN | tool_cached | ⟫')
        assert result1.success

        # 重置 mock 调用计数
        koakuma._bus._mock_storage.get_memory.reset_mock()
        koakuma._bus._mock_storage.get_memory_by_alias.reset_mock()

        # 第二次调用: 应走 LRU 缓存
        result2 = _execute_mtp(koakuma, '⟪ RUN | tool_cached | ⟫')
        assert result2.success
        assert "cached" in result2.response_content

        # 验证没有再查 Qdrant
        koakuma._bus._mock_storage.get_memory.assert_not_called()
        koakuma._bus._mock_storage.get_memory_by_alias.assert_not_called()

    def test_non_code_snippet_rejected(self, koakuma):
        """类型不是 CODE_SNIPPET 时拒绝执行"""
        fact_mem = _make_fact_memory()
        koakuma.atom_cache.ingest_atom(fact_mem)

        result = _execute_mtp(koakuma, '⟪ RUN | fact_not_tool | ⟫')

        assert not result.success
        assert "CODE_SNIPPET" in result.response_content
        assert koakuma._bus._memory_citations == []

    def test_sandbox_timeout(self, koakuma):
        """死循环代码触发超时"""
        mem = _make_code_memory(
            code="while True: pass",
            alias="tool_infinite",
        )
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem
        # 使用极短超时加速测试
        koakuma._config.python_repl_timeout_seconds = 1

        result = _execute_mtp(koakuma, '⟪ RUN | tool_infinite | ⟫')

        assert not result.success
        assert "timed out" in result.response_content.lower()

    def test_sandbox_import_blocked(self, koakuma):
        """import 语句被拦截"""
        mem = _make_code_memory(
            code="import os\nprint(os.getcwd())",
            alias="tool_bad_import",
        )
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        result = _execute_mtp(koakuma, '⟪ RUN | tool_bad_import | ⟫')

        assert not result.success
        assert "import" in result.response_content.lower()

    def test_params_injection(self, koakuma):
        """验证 params 字典正确传入用户态工具"""
        mem = _make_code_memory(
            code='print(f"x={params[\'x\']}, y={params[\'y\']}")',
            alias="tool_params",
        )
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        result = _execute_mtp(koakuma, '⟪ RUN | tool_params | x="42" y="hello" ⟫')

        assert result.success
        assert "x=42" in result.response_content
        assert "y=hello" in result.response_content

    def test_cache_hit_after_ingest(self, koakuma):
        """缓存命中后直接执行，不查 Qdrant"""
        mem = _make_code_memory(code="print('cached')", alias="tool_cached_ingest")
        koakuma.atom_cache.ingest_atom(mem)

        result = _execute_mtp(koakuma, '⟪ RUN | tool_cached_ingest | ⟫')

        assert result.success
        assert "cached" in result.response_content
        # 验证没有查 Qdrant
        koakuma._bus._mock_storage.get_memory.assert_not_called()
        koakuma._bus._mock_storage.get_memory_by_alias.assert_not_called()

    def test_redirected_pending_alias_executes_canonical_tool(self, koakuma):
        pending = koakuma.pending_cache.register_write(
            content="pending tool",
            title="Pending Tool",
            reason=None,
            identity=MTPExecutionContext().identity,
        )
        canonical = _make_code_memory(
            code="print('redirected tool output')",
            alias="tool_canonical",
        )
        koakuma.atom_cache.ingest_atom(canonical)
        koakuma.pending_cache.apply_settlement(
            PendingAtomSettlement(
                pending_alias=pending.pending_alias,
                intent_id=pending.intent_id,
                resolution=PendingAtomResolution.CREATED,
                duplicate_decision=DuplicateDecision.CREATE,
                canonical_alias="tool_canonical",
                canonical_uuid=str(canonical.id),
            )
        )

        result = _execute_mtp(koakuma, f'⟪ RUN | {pending.pending_alias} | ⟫')

        assert result.success
        assert "[Alias Redirected]" in result.response_content
        assert f"Requested alias: {pending.pending_alias}" in result.response_content
        assert "Canonical alias: tool_canonical" in result.response_content
        assert "redirected tool output" in result.response_content
        assert koakuma._bus._memory_citations == [
            {"memory_id": canonical.id, "source": "mtp.run"}
        ]

    def test_user_tool_success_returns_execution_result(self, koakuma):
        """成功执行后返回工具输出，trace 由 TurnEvent reducer 负责生成。"""
        mem = _make_code_memory(code="print('traced')", alias="tool_trace")
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        result = _execute_mtp(koakuma, '⟪ RUN | tool_trace | ⟫')

        assert result.success
        assert "traced" in result.response_content

    def test_user_tool_error_returns_execution_failure(self, koakuma):
        """执行失败时返回错误响应，trace 由 TurnEvent reducer 负责生成。"""
        mem = _make_code_memory(code="raise ValueError('boom')", alias="tool_err")
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        result = _execute_mtp(koakuma, '⟪ RUN | tool_err | ⟫')

        assert not result.success
        assert "Error" in result.response_content
        assert koakuma._bus._memory_citations == []

    def test_citation_failure_keeps_user_tool_success_response(self, koakuma):
        mem = _make_code_memory(code="print('still ok')", alias="tool_cite_fail")
        koakuma.atom_cache.ingest_atom(mem)
        koakuma._bus.unregister("patchouli.public.record_memory_citation")

        result = _execute_mtp(koakuma, '⟪ RUN | tool_cite_fail | ⟫')

        assert result.success
        assert "still ok" in result.response_content
