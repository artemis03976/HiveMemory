"""
RunExecutor 架构哨兵测试（unit 保留集）

CallCoordinator 错误路径的真实协作测试已迁移至
tests/integration/alice/orchestration/test_call_coordinator_error_paths.py。
本文件保留架构守护测试：仅 RunExecutor 可以推进 frame。
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_run_executor_is_the_only_alice_run_frame_caller():
    """Alice 编排层仅递归 RunExecutor 可以推进 frame。"""
    repo_root = Path(__file__).resolve().parents[4]
    alice_root = repo_root / "src" / "hivememory" / "alice"
    callers: set[str] = set()

    for source_path in alice_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_frame"
            for node in ast.walk(tree)
        ):
            callers.add(source_path.name)

    assert callers == {"run_executor.py"}
