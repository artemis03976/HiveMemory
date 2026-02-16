"""
RUN 指令执行链路测试

验证 MTP RUN 指令从 Koakuma._handle_run 的完整链路。

测试覆盖:
    1. Target 校验 (通配符/列表/空 target 拒绝)
    2. Level 0 内核工具快速路径 (sys_clock, sys_python_repl)
    3. Level 1 用户态工具 (未实现, 返回 not found)

与 test_syscall_chain.py 的区别:
    - test_syscall_chain.py: 聚焦 syscall 函数本身的行为正确性
    - 本文件: 聚焦 RUN 指令的 _handle_run 链路 (target 校验、快速路径分发)

作者: HiveMemory Team
版本: 1.0
"""

import pytest
from unittest.mock import MagicMock

from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.config import KoakumaConfig


# ========== Fixtures ==========

@pytest.fixture
def koakuma() -> KoakumaRuntime:
    return KoakumaRuntime(
        retrieval_familiar=MagicMock(),
        librarian_core=MagicMock(),
        storage=MagicMock(),
        config=KoakumaConfig(),
    )


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


# ========== Test 3: User Tool Path ==========

class TestRunUserToolPath:
    """Level 1 用户态工具 (未实现)"""

    def test_unknown_tool_not_found(self, koakuma):
        result = koakuma.execute_mtp('⟪ RUN | nonexistent_tool | ⟫')
        assert not result.success
        assert "not found" in result.response_content.lower()

    def test_unknown_tool_suggests_search(self, koakuma):
        result = koakuma.execute_mtp('⟪ RUN | my_custom_tool | ⟫')
        assert not result.success
        assert "SEARCH" in result.response_content
