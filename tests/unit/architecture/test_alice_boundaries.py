"""Alice 子系统内部的静态结构约束（AST 扫描）。

包分层测试把 ``alice`` 与 ``agent_runtime`` 视为同一子系统，不约束两者之间的方向；
本模块守护子系统内部的两条方向：执行内核不反向依赖编排层，且只有 RunExecutor
推进 frame。
"""

from __future__ import annotations

import ast

from tests.unit.architecture.test_package_layers import SRC_ROOT, _internal_imports


def test_agent_runtime_package_does_not_import_alice() -> None:
    """agent_runtime 是 Alice 编排层之下的执行内核，不能反向导入 alice。"""
    violations = [
        str(path.relative_to(SRC_ROOT))
        for path in sorted((SRC_ROOT / "agent_runtime").rglob("*.py"))
        if any(
            module == "hivememory.alice" or module.startswith("hivememory.alice.")
            for module in _internal_imports(path)
        )
    ]

    assert violations == []


def test_run_executor_is_the_only_alice_run_frame_caller() -> None:
    """Alice 编排层仅递归 RunExecutor 可以推进 frame。"""
    callers = {
        path.name
        for path in (SRC_ROOT / "alice").rglob("*.py")
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_frame"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    }

    assert callers == {"run_executor.py"}
