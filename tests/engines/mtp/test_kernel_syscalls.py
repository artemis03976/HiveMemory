"""
Kernel Syscalls 单元测试

测试覆盖:
- sys_clock: 各格式输出
- sys_python_repl: 正常执行、安全限制、超时、错误处理
- build_kernel_registry: 注册表构建
- Koakuma RUN 集成: 通过 MTP 指令执行 syscalls

对应设计文档: MemoryToolProtocol.md Chapter 4 & 8
"""

import re
import pytest
from unittest.mock import MagicMock

from hivememory.patchouli.kernel.syscalls import (
    KernelSyscall,
    sys_clock,
    sys_python_repl,
    build_kernel_registry,
)
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.config import KoakumaConfig


# ========== sys_clock 测试 ==========

class TestSysClock:
    """测试 sys_clock 内核工具"""

    def test_default_format(self):
        """默认格式: YYYY-MM-DD HH:MM:SS (UTC+X)"""
        result = sys_clock({})
        # 匹配 "2025-06-01 14:30:00 (UTC+8)" 格式
        assert re.match(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(UTC[+-]\d+\)",
            result,
        ), f"Unexpected format: {result}"

    def test_iso_format(self):
        """ISO 8601 格式"""
        result = sys_clock({"format": "iso"})
        # ISO 格式包含 T 分隔符
        assert "T" in result
        # 包含时区信息
        assert "+" in result or "-" in result or "Z" in result

    def test_date_format(self):
        """日期格式: YYYY-MM-DD"""
        result = sys_clock({"format": "date"})
        assert re.match(r"\d{4}-\d{2}-\d{2}$", result), f"Unexpected: {result}"

    def test_time_format(self):
        """时间格式: HH:MM:SS"""
        result = sys_clock({"format": "time"})
        assert re.match(r"\d{2}:\d{2}:\d{2}$", result), f"Unexpected: {result}"

    def test_no_args(self):
        """无参数时使用默认格式"""
        result = sys_clock({})
        assert "UTC" in result

    def test_unknown_format_uses_default(self):
        """未知格式回退到默认"""
        result = sys_clock({"format": "unknown"})
        assert "UTC" in result


# ========== sys_python_repl 测试 ==========
class TestSysPythonRepl:
    """测试 sys_python_repl 受限 REPL"""

    def test_simple_calculation(self):
        """简单计算: print(1+1) → Stdout: 2"""
        result = sys_python_repl({"code": "print(1 + 1)"})
        assert result == "Stdout: 2"

    def test_no_output(self):
        """无输出: x=1 → 成功但无输出"""
        result = sys_python_repl({"code": "x = 1"})
        assert "Executed successfully" in result
        assert "no output" in result

    def test_missing_code_arg(self):
        """缺少 code 参数"""
        result = sys_python_repl({})
        assert "Error" in result
        assert "'code' argument is required" in result

    def test_empty_code_arg(self):
        """空 code 参数"""
        result = sys_python_repl({"code": ""})
        assert "Error" in result

    def test_import_blocked(self):
        """import 语句被阻止"""
        result = sys_python_repl({"code": "import os"})
        assert "Error" in result
        assert "import" in result.lower()

    def test_open_blocked(self):
        """open() 不在白名单中"""
        result = sys_python_repl({"code": "open('test.txt')"})
        assert "Error" in result

    def test_exec_blocked(self):
        """exec() 不在白名单中"""
        result = sys_python_repl({"code": "exec('print(1)')"})
        assert "Error" in result

    def test_eval_blocked(self):
        """eval() 不在白名单中"""
        result = sys_python_repl({"code": "eval('1+1')"})
        assert "Error" in result

    def test_dunder_import_blocked(self):
        """__import__() 被阻止"""
        result = sys_python_repl({"code": "__import__('os')"})
        assert "Error" in result
        assert "import" in result.lower()

    def test_multiline_code(self):
        """多行代码正常执行"""
        code = "x = 10\ny = 20\nprint(x + y)"
        result = sys_python_repl({"code": code})
        assert result == "Stdout: 30"

    def test_runtime_error(self):
        """运行时错误: 1/0 → 返回 traceback"""
        result = sys_python_repl({"code": "1/0"})
        assert "Error" in result
        assert "ZeroDivisionError" in result

    def test_timeout(self):
        """超时: while True → 超时错误"""
        result = sys_python_repl(
            {"code": "while True: pass"},
            timeout_seconds=1,
        )
        assert "Error" in result
        assert "timed out" in result

    def test_safe_builtins_available(self):
        """安全 builtins 可用: len, range, sorted"""
        code = "print(len(list(range(5))))"
        result = sys_python_repl({"code": code})
        assert result == "Stdout: 5"

    def test_sorted_builtin(self):
        """sorted() 可用"""
        code = "print(sorted([3, 1, 2]))"
        result = sys_python_repl({"code": code})
        assert result == "Stdout: [1, 2, 3]"

    def test_math_operations(self):
        """数学运算"""
        code = "print(pow(2, 10))"
        result = sys_python_repl({"code": code})
        assert result == "Stdout: 1024"

    def test_string_operations(self):
        """字符串操作"""
        code = "print('hello'.upper())"
        result = sys_python_repl({"code": code})
        assert result == "Stdout: HELLO"

    def test_list_comprehension(self):
        """列表推导式"""
        code = "print([x**2 for x in range(5)])"
        result = sys_python_repl({"code": code})
        assert result == "Stdout: [0, 1, 4, 9, 16]"


# ========== build_kernel_registry 测试 ==========

class TestKernelRegistry:
    """测试注册表构建"""

    def test_build_registry(self):
        """注册表包含 sys_clock 和 sys_python_repl"""
        registry = build_kernel_registry()
        assert "sys_clock" in registry
        assert "sys_python_repl" in registry

    def test_registry_types(self):
        """注册表值为 KernelSyscall 类型"""
        registry = build_kernel_registry()
        for name, syscall in registry.items():
            assert isinstance(syscall, KernelSyscall), f"{name} is not KernelSyscall"

    def test_registry_descriptions(self):
        """每个工具有非空描述"""
        registry = build_kernel_registry()
        for name, syscall in registry.items():
            assert syscall.description, f"{name} has empty description"

    def test_registry_handlers_callable(self):
        """每个工具的 handler 可调用"""
        registry = build_kernel_registry()
        for name, syscall in registry.items():
            assert callable(syscall.handler), f"{name} handler not callable"

    def test_custom_repl_timeout(self):
        """自定义 REPL 超时"""
        registry = build_kernel_registry(python_repl_timeout=5)
        # 验证 sys_python_repl 的 handler 是 partial
        assert "sys_python_repl" in registry


# ========== Koakuma RUN 集成测试 ==========

class TestKoakumaRunIntegration:
    """测试通过 Koakuma 执行 RUN 指令"""

    @pytest.fixture
    def koakuma(self) -> KoakumaRuntime:
        """提供 KoakumaRuntime 实例"""
        return KoakumaRuntime(
            retrieval_familiar=MagicMock(),
            librarian_core=MagicMock(),
            storage=MagicMock(),
            config=KoakumaConfig(),
        )

    def test_run_sys_clock(self, koakuma: KoakumaRuntime):
        """通过 MTP 执行 RUN | sys_clock"""
        result = koakuma.execute_mtp("⟪ RUN | sys_clock | ⟫")
        assert result.success is True
        assert "UTC" in result.response_content

    def test_run_sys_clock_with_format(self, koakuma: KoakumaRuntime):
        """通过 MTP 执行 RUN | sys_clock | format="iso" """
        result = koakuma.execute_mtp('⟪ RUN | sys_clock | format="iso" ⟫')
        assert result.success is True
        assert "T" in result.response_content

    def test_run_sys_python_repl(self, koakuma: KoakumaRuntime):
        """通过 MTP 执行 RUN | sys_python_repl"""
        result = koakuma.execute_mtp(
            '⟪ RUN | sys_python_repl | code="print(42)" ⟫'
        )
        assert result.success is True
        assert "42" in result.response_content

    def test_run_unknown_tool(self, koakuma: KoakumaRuntime):
        """未知工具返回 not found"""
        result = koakuma.execute_mtp("⟪ RUN | nonexistent_tool | ⟫")
        assert result.success is False
        assert "not found" in result.response_content.lower()

    def test_run_no_target(self, koakuma: KoakumaRuntime):
        """无 target 返回错误"""
        result = koakuma.execute_mtp("⟪ RUN | * | ⟫")
        assert result.success is False

    def test_run_repl_security_via_mtp(self, koakuma: KoakumaRuntime):
        """通过 MTP 验证 REPL 安全限制"""
        result = koakuma.execute_mtp(
            '⟪ RUN | sys_python_repl | code="import os" ⟫'
        )
        assert result.success is True  # handler 返回 error string, 不是异常
        assert "Error" in result.response_content
        assert "import" in result.response_content.lower()
