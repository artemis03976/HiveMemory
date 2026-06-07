import pytest

from hivememory.agent_runtime.mtp.syscalls.repl import sys_python_repl
from hivememory.core.mtp.exceptions import (
    SyscallExecutionError,
    SyscallInvalidArgumentError,
    SyscallPermissionDeniedError,
    SyscallTimeoutError,
)


class TestSysPythonRepl:
    """sys_python_repl 函数直接测试。"""

    def test_simple_calculation(self):
        result = sys_python_repl({"code": "print(1 + 1)"})
        assert result.content == "Stdout: 2"

    def test_no_output(self):
        result = sys_python_repl({"code": "x = 1"})
        assert "no output" in result.content.lower() or "无输出" in result.content

    def test_missing_code_arg(self):
        with pytest.raises(SyscallInvalidArgumentError) as exc_info:
            sys_python_repl({})

        assert exc_info.value.message_key == "syscall.repl.missing_code"

    def test_empty_code_arg(self):
        with pytest.raises(SyscallInvalidArgumentError):
            sys_python_repl({"code": ""})

    def test_import_blocked(self):
        with pytest.raises(SyscallPermissionDeniedError) as exc_info:
            sys_python_repl({"code": "import os"})

        assert exc_info.value.message_key == "syscall.repl.import_blocked"

    def test_open_blocked(self):
        with pytest.raises(SyscallExecutionError):
            sys_python_repl({"code": "open('test.txt')"})

    def test_exec_blocked(self):
        with pytest.raises(SyscallExecutionError):
            sys_python_repl({"code": "exec('print(1)')"})

    def test_eval_blocked(self):
        with pytest.raises(SyscallExecutionError):
            sys_python_repl({"code": "eval('1+1')"})

    def test_dunder_import_blocked(self):
        with pytest.raises(SyscallPermissionDeniedError) as exc_info:
            sys_python_repl({"code": "__import__('os')"})

        assert exc_info.value.message_key == "syscall.repl.import_blocked"

    def test_multiline_code(self):
        code = "x = 10\ny = 20\nprint(x + y)"
        result = sys_python_repl({"code": code})
        assert result.content == "Stdout: 30"

    def test_runtime_error(self):
        with pytest.raises(SyscallExecutionError) as exc_info:
            sys_python_repl({"code": "1/0"})

        assert exc_info.value.message_key == "syscall.repl.execution_failed"

    def test_timeout(self):
        with pytest.raises(SyscallTimeoutError) as exc_info:
            sys_python_repl(
                {"code": "while True: pass"},
                timeout_seconds=1,
            )

        assert exc_info.value.message_key == "syscall.repl.timeout"

    def test_safe_builtins_available(self):
        result = sys_python_repl({"code": "print(len(list(range(5))))"})
        assert result.content == "Stdout: 5"

    def test_sorted_builtin(self):
        result = sys_python_repl({"code": "print(sorted([3, 1, 2]))"})
        assert result.content == "Stdout: [1, 2, 3]"

    def test_math_operations(self):
        result = sys_python_repl({"code": "print(pow(2, 10))"})
        assert result.content == "Stdout: 1024"

    def test_string_operations(self):
        result = sys_python_repl({"code": "print('hello'.upper())"})
        assert result.content == "Stdout: HELLO"

    def test_list_comprehension(self):
        result = sys_python_repl({"code": "print([x**2 for x in range(5)])"})
        assert result.content == "Stdout: [0, 1, 4, 9, 16]"
