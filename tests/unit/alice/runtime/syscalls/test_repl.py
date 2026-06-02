from hivememory.agent_runtime.mtp.syscalls.repl import sys_python_repl


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
        assert "runtime errors" in result.lower()

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
