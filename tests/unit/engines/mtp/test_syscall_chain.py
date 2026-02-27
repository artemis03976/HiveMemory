"""
Syscall 执行链路闭环测试

验证所有内核 syscall 的代码正确性，包括:
    1. 直接函数调用 (sys_clock, sys_python_repl, sys_web_search, sys_read_file, sys_write_file)
    2. 通过 MTP RUN 指令的集成路径 (intercept_and_execute)
    3. build_kernel_registry 注册表
    4. 错误恢复与多轮递归
    5. MTP Prompt 教学内容验证

与 test_run_chain.py 的区别:
    - test_run_chain.py: 聚焦 RUN 指令的 _handle_run 链路 (target 校验、快速路径分发、用户态工具)
    - 本文件: 聚焦 syscall 函数本身的行为正确性 + 通过 MTP 的集成验证

作者: HiveMemory Team
版本: 1.0
"""

import re
import pytest
from unittest.mock import MagicMock, patch

from hivememory.patchouli.kernel.syscalls import (
    KernelSyscall,
    sys_clock,
    sys_python_repl,
    sys_web_search,
    sys_read_file,
    sys_write_file,
    build_kernel_registry,
)
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.config import KoakumaConfig
from hivememory.patchouli.protocol.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTPVerb,
)
from hivememory.patchouli.protocol.models import MTPExecutionResult
from hivememory.patchouli.prompts.mtp_prompt import (
    MTPPromptBuilder,
    AgentRole,
)


# ========== Fixtures ==========

@pytest.fixture
def koakuma() -> KoakumaRuntime:
    """标准 KoakumaRuntime (真实 syscalls, mock 兄弟服务)"""
    from tests.unit.engines.mtp.conftest import make_mock_bus
    bus = make_mock_bus()
    return KoakumaRuntime(bus=bus, config=KoakumaConfig())


@pytest.fixture
def mtp_prompt_en() -> str:
    """英文 MTP System Prompt"""
    return MTPPromptBuilder(language="en", role=AgentRole.DEFAULT).build()


@pytest.fixture
def mtp_prompt_zh() -> str:
    """中文 MTP System Prompt"""
    return MTPPromptBuilder(language="zh", role=AgentRole.DEFAULT).build()


# ========== Helpers ==========

def simulate_kernel_loop_single(
    koakuma: KoakumaRuntime,
    agent_text: str,
) -> MTPExecutionResult:
    """模拟单次 Kernel Recursive Loop"""
    result = koakuma.intercept_and_execute(agent_text)
    assert result is not None, (
        f"Kernel Loop 未检测到 MTP 指令。Agent 文本: {agent_text!r}"
    )
    return result


def build_resumed_history(
    agent_prefix: str,
    mtp_result: MTPExecutionResult,
) -> str:
    """构建 Fake Assistant History"""
    return agent_prefix + mtp_result.formatted_response


# ========== Test 1: sys_clock 直接调用 ==========

class TestSysClock:
    """sys_clock 函数直接测试"""

    def test_default_format(self):
        result = sys_clock({})
        assert re.match(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(UTC[+-]\d+\)",
            result,
        ), f"Unexpected format: {result}"

    def test_iso_format(self):
        result = sys_clock({"format": "iso"})
        assert "T" in result
        assert "+" in result or "-" in result or "Z" in result

    def test_date_format(self):
        result = sys_clock({"format": "date"})
        assert re.match(r"\d{4}-\d{2}-\d{2}$", result)

    def test_time_format(self):
        result = sys_clock({"format": "time"})
        assert re.match(r"\d{2}:\d{2}:\d{2}$", result)

    def test_no_args_uses_default(self):
        result = sys_clock({})
        assert "UTC" in result

    def test_unknown_format_uses_default(self):
        result = sys_clock({"format": "unknown"})
        assert "UTC" in result


# ========== Test 2: sys_python_repl 直接调用 ==========

class TestSysPythonRepl:
    """sys_python_repl 函数直接测试"""

    def test_simple_calculation(self):
        result = sys_python_repl({"code": "print(1 + 1)"})
        assert result == "Stdout: 2"

    def test_no_output(self):
        result = sys_python_repl({"code": "x = 1"})
        assert "no output" in result.lower()

    def test_missing_code_arg(self):
        result = sys_python_repl({})
        assert "Error" in result
        assert "'code' argument is required" in result

    def test_empty_code_arg(self):
        result = sys_python_repl({"code": ""})
        assert "Error" in result

    def test_import_blocked(self):
        result = sys_python_repl({"code": "import os"})
        assert "Error" in result
        assert "import" in result.lower()

    def test_open_blocked(self):
        result = sys_python_repl({"code": "open('test.txt')"})
        assert "Error" in result

    def test_exec_blocked(self):
        result = sys_python_repl({"code": "exec('print(1)')"})
        assert "Error" in result

    def test_eval_blocked(self):
        result = sys_python_repl({"code": "eval('1+1')"})
        assert "Error" in result

    def test_dunder_import_blocked(self):
        result = sys_python_repl({"code": "__import__('os')"})
        assert "Error" in result
        assert "import" in result.lower()

    def test_multiline_code(self):
        code = "x = 10\ny = 20\nprint(x + y)"
        result = sys_python_repl({"code": code})
        assert result == "Stdout: 30"

    def test_runtime_error(self):
        result = sys_python_repl({"code": "1/0"})
        assert "Error" in result
        assert "ZeroDivisionError" in result

    def test_timeout(self):
        result = sys_python_repl(
            {"code": "while True: pass"},
            timeout_seconds=1,
        )
        assert "Error" in result
        assert "timed out" in result

    def test_safe_builtins_available(self):
        result = sys_python_repl({"code": "print(len(list(range(5))))"})
        assert result == "Stdout: 5"

    def test_sorted_builtin(self):
        result = sys_python_repl({"code": "print(sorted([3, 1, 2]))"})
        assert result == "Stdout: [1, 2, 3]"

    def test_math_operations(self):
        result = sys_python_repl({"code": "print(pow(2, 10))"})
        assert result == "Stdout: 1024"

    def test_string_operations(self):
        result = sys_python_repl({"code": "print('hello'.upper())"})
        assert result == "Stdout: HELLO"

    def test_list_comprehension(self):
        result = sys_python_repl({"code": "print([x**2 for x in range(5)])"})
        assert result == "Stdout: [0, 1, 4, 9, 16]"

# ========== Test 3: sys_web_search 直接调用 ==========

class TestSysWebSearch:
    """sys_web_search 函数直接测试"""

    def test_missing_query(self):
        result = sys_web_search({})
        assert "Error" in result
        assert "query" in result.lower()

    def test_empty_query(self):
        result = sys_web_search({"query": ""})
        assert "Error" in result

    @patch("hivememory.patchouli.kernel.syscalls.DDGS", create=True)
    def test_normal_search(self, mock_ddgs_cls):
        """正常搜索 (mock DDGS)"""
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = [
            {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
        ]
        mock_ddgs_cls.return_value = mock_instance

        try:
            from duckduckgo_search import DDGS
            with patch("hivememory.patchouli.kernel.syscalls.DDGS", mock_ddgs_cls):
                result = sys_web_search({"query": "python async"})
                assert "Result 1" in result
        except ImportError:
            result = sys_web_search({"query": "test"})
            assert "not installed" in result

    def test_num_parameter_non_numeric(self):
        """num 非数字默认为 3"""
        try:
            from duckduckgo_search import DDGS
            pytest.skip("Skipping to avoid real network call")
        except ImportError:
            result = sys_web_search({"query": "test", "num": "abc"})
            assert "not installed" in result


# ========== Test 4: sys_read_file 直接调用 ==========

class TestSysReadFile:
    """sys_read_file 函数直接测试"""

    def test_normal_read(self, tmp_path):
        (tmp_path / "hello.txt").write_text("Hello World", encoding="utf-8")
        result = sys_read_file({"path": "hello.txt"}, workspace=str(tmp_path))
        assert "Hello World" in result
        assert "<content>" in result

    def test_file_not_found(self, tmp_path):
        result = sys_read_file({"path": "missing.txt"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "not found" in result.lower()

    def test_path_traversal_blocked(self, tmp_path):
        result = sys_read_file({"path": "../../etc/passwd"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "denied" in result.lower() or "escape" in result.lower()

    def test_large_file_truncated(self, tmp_path):
        (tmp_path / "large.txt").write_text("A" * 200, encoding="utf-8")
        result = sys_read_file({"path": "large.txt"}, workspace=str(tmp_path), max_bytes=50)
        assert "Truncated" in result

    def test_missing_path_arg(self):
        result = sys_read_file({})
        assert "Error" in result
        assert "path" in result.lower()

    def test_binary_file_rejected(self, tmp_path):
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\x03" * 200)
        result = sys_read_file({"path": "binary.bin"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "binary" in result.lower()

    def test_subdirectory_read(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("print('hello')", encoding="utf-8")
        result = sys_read_file({"path": "src/main.py"}, workspace=str(tmp_path))
        assert "print('hello')" in result

# ========== Test 5: sys_write_file 直接调用 ==========

class TestSysWriteFile:
    """sys_write_file 函数直接测试"""

    def test_normal_write(self, tmp_path):
        result = sys_write_file(
            {"path": "output.txt", "content": "Hello"},
            workspace=str(tmp_path),
        )
        assert "Success" in result
        assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "Hello"

    def test_path_traversal_blocked(self, tmp_path):
        result = sys_write_file(
            {"path": "../../evil.txt", "content": "bad"},
            workspace=str(tmp_path),
        )
        assert "Error" in result
        assert "denied" in result.lower() or "escape" in result.lower()

    def test_content_too_large(self, tmp_path):
        result = sys_write_file(
            {"path": "big.txt", "content": "A" * 200},
            workspace=str(tmp_path),
            max_bytes=50,
        )
        assert "Error" in result
        assert "too large" in result.lower()

    def test_missing_content(self, tmp_path):
        result = sys_write_file({"path": "empty.txt"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "content" in result.lower()

    def test_missing_path(self, tmp_path):
        result = sys_write_file({"content": "hello"}, workspace=str(tmp_path))
        assert "Error" in result
        assert "path" in result.lower()

    def test_append_mode(self, tmp_path):
        (tmp_path / "append.txt").write_text("line1\n", encoding="utf-8")
        result = sys_write_file(
            {"path": "append.txt", "content": "line2\n", "mode": "append"},
            workspace=str(tmp_path),
        )
        assert "Success" in result
        assert (tmp_path / "append.txt").read_text(encoding="utf-8") == "line1\nline2\n"

    def test_invalid_mode(self, tmp_path):
        result = sys_write_file(
            {"path": "test.txt", "content": "data", "mode": "delete"},
            workspace=str(tmp_path),
        )
        assert "Error" in result
        assert "Invalid mode" in result

    def test_auto_create_parent_dirs(self, tmp_path):
        result = sys_write_file(
            {"path": "sub/dir/file.txt", "content": "nested"},
            workspace=str(tmp_path),
        )
        assert "Success" in result
        assert (tmp_path / "sub" / "dir" / "file.txt").exists()

    def test_overwrite_existing(self, tmp_path):
        (tmp_path / "exist.txt").write_text("old content", encoding="utf-8")
        result = sys_write_file(
            {"path": "exist.txt", "content": "new content"},
            workspace=str(tmp_path),
        )
        assert "Success" in result
        assert (tmp_path / "exist.txt").read_text(encoding="utf-8") == "new content"

# ========== Test 6: build_kernel_registry ==========

class TestKernelRegistry:
    """注册表构建测试"""

    def test_contains_all_syscalls(self):
        registry = build_kernel_registry()
        assert "sys_clock" in registry
        assert "sys_python_repl" in registry
        assert "sys_web_search" in registry
        assert "sys_read_file" in registry
        assert "sys_write_file" in registry

    def test_registry_types(self):
        registry = build_kernel_registry()
        for name, syscall in registry.items():
            assert isinstance(syscall, KernelSyscall), f"{name} is not KernelSyscall"

    def test_registry_descriptions(self):
        registry = build_kernel_registry()
        for name, syscall in registry.items():
            assert syscall.description, f"{name} has empty description"

    def test_registry_handlers_callable(self):
        registry = build_kernel_registry()
        for name, syscall in registry.items():
            assert callable(syscall.handler), f"{name} handler not callable"

    def test_custom_repl_timeout(self):
        registry = build_kernel_registry(python_repl_timeout=5)
        assert "sys_python_repl" in registry


# ========== Test 7: Syscall 通过 MTP 集成 ==========

class TestSyscallViaMTP:
    """通过 MTP intercept_and_execute 验证 syscall 集成"""

    def test_clock_intercept(self, koakuma):
        agent_text = "让我查看一下当前时间。\n⟪ RUN | sys_clock |"
        result = simulate_kernel_loop_single(koakuma, agent_text)
        assert result.success is True
        assert result.command.verb == MTPVerb.RUN
        assert re.search(r"\d{4}-\d{2}-\d{2}", result.response_content)
        assert "UTC" in result.response_content

    def test_clock_response_format(self, koakuma):
        result = simulate_kernel_loop_single(koakuma, "⟪ RUN | sys_clock |")
        assert "⟪" in result.formatted_response
        assert '<mtp_response status="success"' in result.formatted_response
        assert "</mtp_response>" in result.formatted_response

    def test_clock_iso_via_mtp(self, koakuma):
        result = simulate_kernel_loop_single(
            koakuma, '⟪ RUN | sys_clock | format="iso"'
        )
        assert result.success is True
        assert "T" in result.response_content

    def test_clock_date_via_mtp(self, koakuma):
        result = simulate_kernel_loop_single(
            koakuma, '⟪ RUN | sys_clock | format="date"'
        )
        assert result.success is True
        assert re.match(r"\d{4}-\d{2}-\d{2}$", result.response_content)

    def test_repl_arithmetic_via_mtp(self, koakuma):
        result = simulate_kernel_loop_single(
            koakuma, '⟪ RUN | sys_python_repl | code="print(12345 * 6789)"'
        )
        assert result.success is True
        assert "83810205" in result.response_content

    def test_repl_backtick_multiline_via_mtp(self, koakuma):
        agent_text = (
            "⟪ RUN | sys_python_repl | code=`\n"
            "total = sum(range(1, 101))\n"
            "print(f'Sum 1-100: {total}')\n"
            "` ⟫"
        )
        result = koakuma.execute_mtp(agent_text)
        assert result.success is True
        assert "5050" in result.response_content

    def test_repl_security_via_mtp(self, koakuma):
        result = simulate_kernel_loop_single(
            koakuma, '⟪ RUN | sys_python_repl | code="import os"'
        )
        assert result.success is True  # handler 返回 error string
        assert "Error" in result.response_content

    def test_repl_runtime_error_via_mtp(self, koakuma):
        result = simulate_kernel_loop_single(
            koakuma, '⟪ RUN | sys_python_repl | code="1/0"'
        )
        assert result.success is True
        assert "ZeroDivisionError" in result.response_content

    def test_repl_no_output_via_mtp(self, koakuma):
        result = simulate_kernel_loop_single(
            koakuma, '⟪ RUN | sys_python_repl | code="x = 42"'
        )
        assert result.success is True
        assert "no output" in result.response_content.lower()

    def test_web_search_missing_query_via_mtp(self, koakuma):
        result = simulate_kernel_loop_single(koakuma, "⟪ RUN | sys_web_search |")
        assert result.success is True
        assert "query" in result.response_content.lower()

    def test_clock_natural_language_response(self, koakuma):
        """响应是人类可读的时间字符串，不是 JSON"""
        result = simulate_kernel_loop_single(koakuma, "⟪ RUN | sys_clock |")
        assert "{" not in result.response_content
        assert "}" not in result.response_content

# ========== Test 8: File I/O 通过 MTP 集成 ==========

class TestFileIOViaMTP:
    """sys_read_file / sys_write_file 通过 MTP 集成测试"""

    @pytest.fixture
    def workspace(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        return ws

    @pytest.fixture
    def file_koakuma(self, workspace):
        from tests.unit.engines.mtp.conftest import make_mock_bus
        bus = make_mock_bus()
        return KoakumaRuntime(
            bus=bus,
            config=KoakumaConfig(workspace_path=str(workspace)),
        )

    def test_read_file_via_mtp(self, file_koakuma, workspace):
        (workspace / "hello.txt").write_text("Hello, World!", encoding="utf-8")
        result = simulate_kernel_loop_single(
            file_koakuma, '⟪ RUN | sys_read_file | path="hello.txt"'
        )
        assert result.success is True
        assert "Hello, World!" in result.response_content
        assert "<content>" in result.response_content

    def test_read_file_not_found_via_mtp(self, file_koakuma):
        result = simulate_kernel_loop_single(
            file_koakuma, '⟪ RUN | sys_read_file | path="nonexistent.txt"'
        )
        assert "not found" in result.response_content.lower()

    def test_read_file_path_traversal_via_mtp(self, file_koakuma):
        result = simulate_kernel_loop_single(
            file_koakuma, '⟪ RUN | sys_read_file | path="../../etc/passwd"'
        )
        assert "denied" in result.response_content.lower() or "escape" in result.response_content.lower()

    def test_read_file_binary_rejected_via_mtp(self, file_koakuma, workspace):
        (workspace / "binary.dat").write_bytes(b"\x00\x01\x02\x03" * 128)
        result = simulate_kernel_loop_single(
            file_koakuma, '⟪ RUN | sys_read_file | path="binary.dat"'
        )
        assert "binary" in result.response_content.lower()

    def test_read_file_truncation_via_mtp(self, file_koakuma, workspace):
        (workspace / "large.txt").write_text("x" * 200000, encoding="utf-8")
        result = simulate_kernel_loop_single(
            file_koakuma, '⟪ RUN | sys_read_file | path="large.txt"'
        )
        assert "truncated" in result.response_content.lower()

    def test_write_file_via_mtp(self, file_koakuma, workspace):
        result = simulate_kernel_loop_single(
            file_koakuma,
            '⟪ RUN | sys_write_file | path="output.txt" content="Hello, World!"',
        )
        assert result.success is True
        assert "success" in result.response_content.lower()
        assert (workspace / "output.txt").read_text(encoding="utf-8") == "Hello, World!"

    def test_write_file_append_via_mtp(self, file_koakuma, workspace):
        (workspace / "log.txt").write_text("line1\n", encoding="utf-8")
        result = simulate_kernel_loop_single(
            file_koakuma,
            '⟪ RUN | sys_write_file | path="log.txt" content="line2\n" mode="append"',
        )
        assert result.success is True
        content = (workspace / "log.txt").read_text(encoding="utf-8")
        assert "line1\n" in content
        assert "line2\n" in content

    def test_write_file_path_traversal_via_mtp(self, file_koakuma):
        result = simulate_kernel_loop_single(
            file_koakuma,
            '⟪ RUN | sys_write_file | path="../../evil.txt" content="pwned"',
        )
        assert "denied" in result.response_content.lower() or "escape" in result.response_content.lower()

    def test_write_file_auto_create_dirs_via_mtp(self, file_koakuma, workspace):
        result = simulate_kernel_loop_single(
            file_koakuma,
            '⟪ RUN | sys_write_file | path="deep/nested/file.txt" content="nested!"',
        )
        assert result.success is True
        assert (workspace / "deep" / "nested" / "file.txt").exists()


# ========== Test 9: 错误恢复 ==========

class TestSyscallErrorRecovery:
    """错误恢复与多轮递归"""

    def test_unknown_tool_guides_search(self, koakuma):
        result = simulate_kernel_loop_single(koakuma, "⟪ RUN | sys_nonexistent_tool |")
        assert result.success is False
        assert "not found" in result.response_content.lower()
        assert "SEARCH" in result.response_content

    def test_invalid_verb_error(self, koakuma):
        result = koakuma.execute_mtp("⟪ DELETE | * | ⟫")
        assert result.success is False
        assert result.command is None
        assert "syntax error" in result.response_content.lower()

    def test_missing_code_arg(self, koakuma):
        result = simulate_kernel_loop_single(koakuma, "⟪ RUN | sys_python_repl |")
        assert result.success is True
        assert "code" in result.response_content.lower()

    def test_error_response_xml_format(self, koakuma):
        result = koakuma.execute_mtp("⟪ RUN | fake_tool | ⟫")
        assert '<mtp_response status="error"' in result.formatted_response
        assert "</mtp_response>" in result.formatted_response

    def test_error_recovery_retry(self, koakuma):
        """Round 1: 错误工具 → Round 2: 纠正"""
        r1 = koakuma.execute_mtp("⟪ RUN | sys_clok | ⟫")
        assert r1.success is False

        r2 = koakuma.execute_mtp("⟪ RUN | sys_clock | ⟫")
        assert r2.success is True
        assert "UTC" in r2.response_content

    def test_no_mtp_in_normal_text(self, koakuma):
        result = koakuma.intercept_and_execute("普通回答，没有 MTP 指令。")
        assert result is None

# ========== Test 10: 多轮递归 ==========

class TestSyscallRecursiveLoop:
    """多轮递归循环验证"""

    def test_two_round_recursive(self, koakuma):
        """Round 1: 获取时间 → Round 2: 计算"""
        r1 = simulate_kernel_loop_single(
            koakuma, "用户问了一个关于时间的问题。\n⟪ RUN | sys_clock |"
        )
        assert r1.success is True

        history_r1 = build_resumed_history("用户问了一个关于时间的问题。\n", r1)
        assert "<mtp_response" in history_r1

        r2 = simulate_kernel_loop_single(
            koakuma, '⟪ RUN | sys_python_repl | code="print(42)"'
        )
        assert r2.success is True
        assert "42" in r2.response_content

    def test_mixed_syscall_sequence(self, koakuma):
        """clock → repl 混合序列"""
        r1 = simulate_kernel_loop_single(koakuma, "⟪ RUN | sys_clock |")
        assert r1.success is True
        assert "UTC" in r1.response_content

        r2 = simulate_kernel_loop_single(
            koakuma, '⟪ RUN | sys_python_repl | code="print(42)"'
        )
        assert r2.success is True
        assert "42" in r2.response_content

    def test_full_loop_history_assembly(self, koakuma):
        """完整闭环: Agent 前缀 + 回填 = 可续写的 assistant 历史"""
        prefix = "让我查看一下当前时间。\n"
        result = simulate_kernel_loop_single(koakuma, prefix + "⟪ RUN | sys_clock |")
        history = build_resumed_history(prefix, result)

        assert history.startswith("让我查看一下当前时间。\n")
        assert "<mtp_response" in history
        assert "</mtp_response>" in history

    def test_repl_history_assembly(self, koakuma):
        """计算结果正确回填到 assistant 历史"""
        prefix = "这个计算需要精确结果。\n"
        result = simulate_kernel_loop_single(
            koakuma, prefix + '⟪ RUN | sys_python_repl | code="print(2**10)"'
        )
        history = build_resumed_history(prefix, result)

        assert "1024" in history
        assert '<mtp_response status="success"' in history


# ========== Test 11: Prompt 教学验证 ==========

class TestSyscallPromptTeaching:
    """MTP System Prompt 教学内容验证"""

    def test_prompt_contains_mtp_syntax(self, mtp_prompt_en):
        assert MTP_LEFT_DELIMITER in mtp_prompt_en
        assert MTP_RIGHT_DELIMITER in mtp_prompt_en
        assert "VERB" in mtp_prompt_en
        assert "TARGET" in mtp_prompt_en
        assert "ARGS" in mtp_prompt_en

    def test_prompt_lists_mvp_syscalls(self, mtp_prompt_en):
        assert "sys_clock" in mtp_prompt_en
        assert "sys_python_repl" in mtp_prompt_en

    def test_prompt_contains_run_verb(self, mtp_prompt_en):
        assert "RUN" in mtp_prompt_en
        assert "Execute" in mtp_prompt_en or "execute" in mtp_prompt_en

    def test_prompt_contains_demo(self, mtp_prompt_en):
        assert "<mtp_response" in mtp_prompt_en
        assert "</mtp_response>" in mtp_prompt_en

    def test_prompt_contains_error_recovery(self, mtp_prompt_en):
        assert "ERROR RECOVERY" in mtp_prompt_en or "error" in mtp_prompt_en.lower()
        assert "retry" in mtp_prompt_en.lower()

    def test_prompt_forbids_json(self, mtp_prompt_en):
        assert "JSON" in mtp_prompt_en
        assert "NEVER" in mtp_prompt_en or "NOT" in mtp_prompt_en

    def test_prompt_zh_structure(self, mtp_prompt_zh):
        assert MTP_LEFT_DELIMITER in mtp_prompt_zh
        assert "sys_clock" in mtp_prompt_zh
        assert "sys_python_repl" in mtp_prompt_zh
        assert "<mtp_response" in mtp_prompt_zh

    def test_prompt_teaches_inline_flow(self, mtp_prompt_en):
        lower = mtp_prompt_en.lower()
        assert "inline" in lower or "thought process" in lower

    def test_prompt_demo_parseable(self, mtp_prompt_en):
        """Prompt 演示中的 MTP 指令可被解析器正确解析"""
        from hivememory.patchouli.protocol.mtp import MTPParser
        parser = MTPParser()

        demo_marker = "ONE-SHOT DEMONSTRATION"
        demo_start = mtp_prompt_en.find(demo_marker)
        assert demo_start != -1, "Prompt 中未找到演示部分"

        demo_text = mtp_prompt_en[demo_start:]
        left = demo_text.find(MTP_LEFT_DELIMITER)
        right = demo_text.find(MTP_RIGHT_DELIMITER, left)
        assert left != -1 and right != -1

        demo_cmd = demo_text[left:right + 1]
        cmd = parser.parse(demo_cmd)
        assert cmd.verb in (
            MTPVerb.SEARCH, MTPVerb.READ,
            MTPVerb.RUN, MTPVerb.WRITE, MTPVerb.UPDATE,
        )
