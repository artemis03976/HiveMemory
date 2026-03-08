"""
RUN 指令执行链路测试

验证 MTP RUN 指令从 Koakuma._handle_run 的完整链路。

测试覆盖:
    1. Target 校验 (通配符/列表/空 target 拒绝)
    2. Level 0 内核工具快速路径 (sys_clock, sys_python_repl)
    3. Level 1 用户态工具慢速路径 (LRU 缓存 + L2 冷检索 + 沙箱执行)
    4. _LRUCache 单元测试

与 test_syscall_chain.py 的区别:
    - test_syscall_chain.py: 聚焦 syscall 函数本身的行为正确性
    - 本文件: 聚焦 RUN 指令的 _handle_run 链路 (target 校验、快速路径分发、慢速路径)

作者: HiveMemory Team
版本: 2.0
"""

import pytest
from uuid import uuid4
from unittest.mock import MagicMock

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
)
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime, _LRUCache
from hivememory.patchouli.config import KoakumaConfig


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


def _make_fact_memory(mem_id=None) -> MemoryAtom:
    """创建 FACT 类型的记忆原子 (不可执行)"""
    return MemoryAtom(
        id=mem_id or uuid4(),
        meta=MetaData(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title="Test Fact",
            summary="A test fact memory",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="This is a fact, not code."),
    )


# ========== Fixtures ==========

@pytest.fixture
def koakuma() -> KoakumaRuntime:
    from tests.unit.engines.mtp.conftest import make_mock_bus
    bus = make_mock_bus()
    return KoakumaRuntime(bus=bus, config=KoakumaConfig())


# ========== Test 1: Target Validation ==========

class TestRunTargetValidation:
    """RUN 指令 target 校验"""

    def test_wildcard_rejected(self, koakuma):
        result = koakuma.execute_mtp('⟪ RUN | * | ⟫')
        assert not result.success
        assert "single tool alias" in result.response_content.lower() or "requires" in result.response_content.lower()

    def test_list_target_rejected(self, koakuma):
        """列表 target 不支持 (single_alias 返回 None)"""
        result = koakuma.execute_mtp('⟪ RUN | [tool_a, tool_b] | ⟫')
        assert not result.success

    def test_empty_target(self, koakuma):
        result = koakuma.execute_mtp('⟪ RUN | | ⟫')
        assert not result.success


# ========== Test 2: Kernel Fast Path ==========

class TestRunKernelFastPath:
    """Level 0 内核工具快速路径"""

    def test_sys_clock_default(self, koakuma):
        result = koakuma.execute_mtp('⟪ RUN | sys_clock | ⟫')
        assert result.success
        assert "UTC" in result.response_content

    def test_sys_clock_iso(self, koakuma):
        result = koakuma.execute_mtp('⟪ RUN | sys_clock | format="iso" ⟫')
        assert result.success
        assert "T" in result.response_content

    def test_sys_clock_date(self, koakuma):
        result = koakuma.execute_mtp('⟪ RUN | sys_clock | format="date" ⟫')
        assert result.success
        # YYYY-MM-DD format
        assert "-" in result.response_content
        assert len(result.response_content.strip()) == 10 or "20" in result.response_content

    def test_sys_python_repl_calculation(self, koakuma):
        result = koakuma.execute_mtp('⟪ RUN | sys_python_repl | code="print(6 * 7)" ⟫')
        assert result.success
        assert "42" in result.response_content

    def test_sys_python_repl_import_blocked(self, koakuma):
        result = koakuma.execute_mtp('⟪ RUN | sys_python_repl | code="import os" ⟫')
        assert result.success  # handler 返回 error string, 不是异常
        assert "Error" in result.response_content
        assert "import" in result.response_content.lower()

    def test_sys_python_repl_multiline(self, koakuma):
        # 使用反引号语法支持多行代码 (Section 2.1)
        result = koakuma.execute_mtp('⟪ RUN | sys_python_repl | code=`x = 10\ny = 20\nprint(x + y)` ⟫')
        assert result.success
        assert "30" in result.response_content


# ========== Test 3: User Tool Path (Level 1) ==========

class TestRunUserToolPath:
    """Level 1 用户态工具慢速路径"""

    def test_unknown_tool_not_found(self, koakuma):
        """L1+L2 均未命中，返回 not found"""
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = None
        result = koakuma.execute_mtp('⟪ RUN | nonexistent_tool | ⟫')
        assert not result.success
        assert "not found" in result.response_content.lower()

    def test_unknown_tool_suggests_search(self, koakuma):
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = None
        result = koakuma.execute_mtp('⟪ RUN | my_custom_tool | ⟫')
        assert not result.success
        assert "SEARCH" in result.response_content

    def test_l2_route_failure_returns_infra_error(self, koakuma):
        koakuma._bus._mock_storage.get_memory_by_alias.side_effect = KeyError(
            "SystemBus: 路由 'storage.get_memory_by_alias' 未注册"
        )

        result = koakuma.execute_mtp('⟪ RUN | my_custom_tool | ⟫')

        assert not result.success
        assert "L2 alias lookup failed" in result.response_content
        assert "storage route is unavailable" in result.response_content

    def test_l2_hit_executes_code_snippet(self, koakuma):
        """L2 命中 CODE_SNIPPET，沙箱执行成功"""
        mem = _make_code_memory(code="print('tool output')", alias="tool_greet")
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        result = koakuma.execute_mtp('⟪ RUN | tool_greet | ⟫')

        assert result.success
        assert "tool output" in result.response_content

    def test_l1_alias_hit_executes(self, koakuma):
        """L1 别名命中 → 加载 → 执行"""
        mem = _make_code_memory(code="print('from l1')", alias="tool_l1")
        uid = str(mem.id)
        koakuma._alias_resolver.register_context_alias("tool_l1", uid)
        koakuma._bus._mock_storage.get_memory.return_value = mem

        result = koakuma.execute_mtp('⟪ RUN | tool_l1 | ⟫')

        assert result.success
        assert "from l1" in result.response_content

    def test_cache_hit_skips_qdrant(self, koakuma):
        """第二次调用走 LRU 缓存，不查 Qdrant"""
        mem = _make_code_memory(code="print('cached')", alias="tool_cached")
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        # 第一次调用: L2 命中 → 加载 → 缓存
        result1 = koakuma.execute_mtp('⟪ RUN | tool_cached | ⟫')
        assert result1.success

        # 重置 mock 调用计数
        koakuma._bus._mock_storage.get_memory.reset_mock()
        koakuma._bus._mock_storage.get_memory_by_alias.reset_mock()

        # 第二次调用: 应走 LRU 缓存
        result2 = koakuma.execute_mtp('⟪ RUN | tool_cached | ⟫')
        assert result2.success
        assert "cached" in result2.response_content

        # 验证没有再查 Qdrant
        koakuma._bus._mock_storage.get_memory.assert_not_called()
        koakuma._bus._mock_storage.get_memory_by_alias.assert_not_called()

    def test_non_code_snippet_rejected(self, koakuma):
        """类型不是 CODE_SNIPPET 时拒绝执行"""
        fact_mem = _make_fact_memory()
        uid = str(fact_mem.id)
        koakuma._alias_resolver.register_context_alias("fact_not_tool", uid)
        koakuma._bus._mock_storage.get_memory.return_value = fact_mem

        result = koakuma.execute_mtp('⟪ RUN | fact_not_tool | ⟫')

        assert not result.success
        assert "CODE_SNIPPET" in result.response_content

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

        result = koakuma.execute_mtp('⟪ RUN | tool_infinite | ⟫')

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

        result = koakuma.execute_mtp('⟪ RUN | tool_bad_import | ⟫')

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

        result = koakuma.execute_mtp('⟪ RUN | tool_params | x="42" y="hello" ⟫')

        assert result.success
        assert "x=42" in result.response_content
        assert "y=hello" in result.response_content

    def test_memory_deleted_after_alias_resolve(self, koakuma):
        """别名解析成功但记忆已被删除"""
        uid = str(uuid4())
        koakuma._alias_resolver.register_context_alias("tool_gone", uid)
        koakuma._bus._mock_storage.get_memory.return_value = None

        result = koakuma.execute_mtp('⟪ RUN | tool_gone | ⟫')

        assert not result.success
        assert "not found" in result.response_content.lower() or "archived" in result.response_content.lower()

    def test_trace_recorded_on_success(self, koakuma):
        """成功执行后记录 TraceItem"""
        mem = _make_code_memory(code="print('traced')", alias="tool_trace")
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        koakuma.execute_mtp('⟪ RUN | tool_trace | ⟫')

        traces = koakuma.get_interaction_traces()
        run_traces = [t for t in traces if t.action == "RUN"]
        assert len(run_traces) == 1
        assert run_traces[0].tool == "tool_trace"
        assert run_traces[0].status == "success"

    def test_trace_recorded_on_error(self, koakuma):
        """执行失败也记录 TraceItem"""
        mem = _make_code_memory(code="raise ValueError('boom')", alias="tool_err")
        koakuma._bus._mock_storage.get_memory_by_alias.return_value = mem
        koakuma._bus._mock_storage.get_memory.return_value = mem

        koakuma.execute_mtp('⟪ RUN | tool_err | ⟫')

        traces = koakuma.get_interaction_traces()
        run_traces = [t for t in traces if t.action == "RUN"]
        assert len(run_traces) == 1
        assert run_traces[0].status == "error"


# ========== Test 4: _LRUCache ==========

class TestLRUCache:
    """_LRUCache 单元测试"""

    def test_basic_put_get(self):
        cache = _LRUCache(maxsize=4)
        cache.put("a", "value_a")
        assert cache.get("a") == "value_a"

    def test_get_miss_returns_none(self):
        cache = _LRUCache(maxsize=4)
        assert cache.get("nonexistent") is None

    def test_eviction(self):
        """超出容量淘汰最久未使用"""
        cache = _LRUCache(maxsize=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")  # 淘汰 "a"

        assert cache.get("a") is None
        assert cache.get("b") == "2"
        assert cache.get("c") == "3"

    def test_access_refreshes_order(self):
        """访问刷新顺序，避免被淘汰"""
        cache = _LRUCache(maxsize=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.get("a")       # 刷新 "a"
        cache.put("c", "3")  # 淘汰 "b" (最久未用)

        assert cache.get("a") == "1"
        assert cache.get("b") is None
        assert cache.get("c") == "3"

    def test_update_existing_key(self):
        cache = _LRUCache(maxsize=4)
        cache.put("a", "old")
        cache.put("a", "new")
        assert cache.get("a") == "new"
        assert len(cache) == 1

    def test_contains(self):
        cache = _LRUCache(maxsize=4)
        cache.put("a", "1")
        assert "a" in cache
        assert "b" not in cache

    def test_len(self):
        cache = _LRUCache(maxsize=4)
        assert len(cache) == 0
        cache.put("a", "1")
        cache.put("b", "2")
        assert len(cache) == 2
