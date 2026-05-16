import re

import pytest

from hivememory.system.config import KoakumaConfig
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.mtp.models import MTPVerb

from .conftest import build_resumed_history, simulate_kernel_loop_single


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
        assert "runtime errors" in result.response_content.lower()

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
        from .conftest import make_mock_bus
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
