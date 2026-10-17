"""Alice 子系统内部的静态结构约束（AST 扫描）。

包分层测试把 ``alice`` 与 ``agent_runtime`` 视为同一子系统，不约束两者之间的方向；
本模块守护执行内核与 Alice 各层的依赖方向，且只有 RunExecutor 推进 frame。
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.unit.architecture.test_package_layers import (
    SRC_ROOT,
    _internal_imports,
    _resolve_relative,
)

ALICE_ROOT = SRC_ROOT / "alice"
ALLOWED_ALICE_IMPORTS = {
    "__init__": {"system"},
    "system": {"interface", "run", "context", "orchestration", "events", "contracts"},
    "interface": {"interface", "run", "orchestration.output", "events", "contracts"},
    "run": {"run", "context", "orchestration", "events"},
    "context": {"context", "events"},
    "orchestration": {"orchestration", "events"},
    "events": {"events"},
    "contracts": {"contracts"},
}
# 运行服务还负责创建流式适配器；只登记这一条具体依赖，不能放宽整个层。
KNOWN_REVERSE_IMPORTS = {("run/service.py", "interface.stream")}


def _package_imports(path: Path, package: str) -> set[str]:
    """解析指定包的模块依赖，包含相对导入、包内模块导入与类型检查分支。"""
    modules = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        if not isinstance(node, ast.ImportFrom):
            continue
        base = _resolve_relative(path, node) if node.level else node.module
        if base is None:
            continue
        for alias in node.names:
            module = f"{base}.{alias.name}"
            module_path = SRC_ROOT.parent.joinpath(*module.split("."))
            if module_path.with_suffix(".py").is_file() or module_path.is_dir():
                modules.add(module)
            else:
                modules.add(base)
    return {
        "__init__" if module == package else module.removeprefix(f"{package}.")
        for module in modules
        if module == package or module.startswith(f"{package}.")
    }


def test_alice_layers_only_import_allowed_dependencies() -> None:
    """上下文准备与帧编排互不依赖；各层不能经导入把控制权交回上层。"""
    violations = set()
    for path in sorted(ALICE_ROOT.rglob("*.py")):
        relative = path.relative_to(ALICE_ROOT)
        layer = relative.parts[0] if len(relative.parts) > 1 else path.stem
        assert layer in ALLOWED_ALICE_IMPORTS, f"未登记 Alice 层: {relative}"
        allowed = ALLOWED_ALICE_IMPORTS[layer]
        for target in _package_imports(path, "hivememory.alice"):
            if not any(target == prefix or target.startswith(f"{prefix}.") for prefix in allowed):
                violations.add((relative.as_posix(), target))

    # 精确比对使新增逆向依赖和已经消失的例外都需要显式审查。
    assert violations == KNOWN_REVERSE_IMPORTS


def test_alice_and_system_only_import_runtime_public_api() -> None:
    """内核使用方只依赖包根显式公开的符号，不能穿透到实现模块。"""
    package = "hivememory.agent_runtime"
    public_tree = ast.parse((SRC_ROOT / "agent_runtime/__init__.py").read_text(encoding="utf-8"))
    exports = next(
        set(ast.literal_eval(node.value))
        for node in public_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    paths = [
        *ALICE_ROOT.rglob("*.py"),
        *(SRC_ROOT / "system").rglob("*.py"),
        SRC_ROOT.parents[1] / "scripts/calibrate_session_compaction.py",
    ]
    violations = []
    for path in paths:
        for target in _package_imports(path, package) - {"__init__"}:
            violations.append(f"{path.name}: {package}.{target}")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            base = _resolve_relative(path, node) if node.level else node.module
            if base == package:
                violations.extend(
                    f"{path.name}: 未公开的符号 {alias.name}"
                    for alias in node.names
                    if alias.name not in exports
                )

    assert violations == []


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
    """Alice 编排层只有递归执行器可以推进 frame。"""
    callers = {
        path.relative_to(ALICE_ROOT).as_posix()
        for path in (SRC_ROOT / "alice").rglob("*.py")
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_frame"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    }

    assert callers == {"orchestration/executor.py"}
