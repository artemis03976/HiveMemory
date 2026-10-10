"""workspace 内部依赖边界的单元测试。

包级分层规则（各层之间、子系统之间的依赖方向）由
``tests/unit/architecture/test_package_layers.py`` 统一守护；本文件只保护
workspace 包内更严格的约束：

1. 认证、准入、读取能力（cache / resolution / runtime）与配置模块只依赖
   core 与 workspace 自身——resolver 经 backing 协议冷读，不导入总线、
   路由常量或任何子系统实现；运行时装配与失效订阅者仅另依赖总线机制；
2. ``intents`` 作为共享登记只可另依赖 ``components``；
   ``capability`` / ``assets`` / ``process`` 作为能力层、资产设施与进程
   编排，可依赖更低层的机制（``components``），不受上一条约束；
3. ``process`` 以外的 workspace 模块不得导入 ``hivememory.workspace.process``：
   能力层与共享设施不得依赖进程表与任务进程编排；
4. ``contracts`` 是跨子系统公共契约子包，只依赖 ``core``——不导入
   workspace 的其他模块，也不导入任何子系统实现。

以 AST 静态扫描直接 import（含相对导入）判定。
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "hivememory"
WORKSPACE_PACKAGE = SRC_ROOT / "workspace"
UNRESTRICTED_SUBPACKAGES = (
    WORKSPACE_PACKAGE / "capability",
    WORKSPACE_PACKAGE / "assets",
    WORKSPACE_PACKAGE / "process",
)
SHARED_EVENT_SUBSCRIBERS = {
    WORKSPACE_PACKAGE / "runtime.py",
    WORKSPACE_PACKAGE / "cache" / "invalidation.py",
}
ALLOWED_INTERNAL = ("hivememory.core", "hivememory.workspace")
PROCESS_PACKAGE_MODULE = "hivememory.workspace.process"


def _resolve_relative(path: Path, node: ast.ImportFrom) -> str:
    """把相对导入解析为 hivememory 内的绝对模块名。"""
    parts = list(path.parent.relative_to(SRC_ROOT).parts)
    keep = len(parts) - (node.level - 1)
    base = parts[: max(keep, 0)]
    if node.module:
        base = [*base, *node.module.split(".")]
    return ".".join(["hivememory", *base])


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.append(node.module)
            elif node.level > 0:
                modules.append(_resolve_relative(path, node))
    return modules


def _sources_excluding_unrestricted() -> list[Path]:
    files = [
        path
        for path in sorted(WORKSPACE_PACKAGE.rglob("*.py"))
        if "__pycache__" not in path.parts
        and not any(sub in path.parents for sub in UNRESTRICTED_SUBPACKAGES)
    ]
    # 防空转：目录缺失或路径漂移时让测试显式失败，而不是零扫描静默通过
    assert files, f"未扫描到 workspace 源文件，请检查路径: {WORKSPACE_PACKAGE}"
    return files


def test_workspace_access_and_read_modules_depend_only_on_core():
    """读取本体只依赖 core / workspace，共享事件装配仅另依赖 components。"""
    violations = [
        f"{path}: {module}"
        for path in _sources_excluding_unrestricted()
        for module in _imports_of(path)
        if module.startswith("hivememory")
        and not module.startswith(ALLOWED_INTERNAL)
        and not (
            (path in SHARED_EVENT_SUBSCRIBERS or WORKSPACE_PACKAGE / "intents" in path.parents)
            and module.startswith("hivememory.components")
        )
    ]
    assert violations == []


def _non_process_sources() -> list[Path]:
    """workspace 中 process 子包以外的全部源文件（含 capability 与 assets）。"""
    process_package = WORKSPACE_PACKAGE / "process"
    files = [
        path
        for path in sorted(WORKSPACE_PACKAGE.rglob("*.py"))
        if "__pycache__" not in path.parts and process_package not in path.parents
    ]
    assert files, f"未扫描到 workspace 源文件，请检查路径: {WORKSPACE_PACKAGE}"
    return files


def _import_targets_of(path: Path) -> list[str]:
    """直接 import 的模块，以及 ``from X import Y`` 形式可能指向的子模块 ``X.Y``。"""
    targets = _imports_of(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                base = node.module
            elif node.level > 0:
                base = _resolve_relative(path, node)
            else:
                continue
            targets.extend(f"{base}.{alias.name}" for alias in node.names)
    return targets


def test_workspace_non_process_modules_do_not_import_process():
    """process 以外的 workspace 模块（含能力层与资产设施）不得导入进程表与任务进程编排。"""
    violations = [
        f"{path}: {target}"
        for path in _non_process_sources()
        for target in _import_targets_of(path)
        if target == PROCESS_PACKAGE_MODULE or target.startswith(f"{PROCESS_PACKAGE_MODULE}.")
    ]
    assert violations == []


def test_workspace_contracts_only_depend_on_core():
    """contracts 是跨子系统公共契约子包：只允许 import core 与自身子模块。"""
    contracts_package = WORKSPACE_PACKAGE / "contracts"
    contracts_module = "hivememory.workspace.contracts"
    files = [
        path for path in sorted(contracts_package.rglob("*.py")) if "__pycache__" not in path.parts
    ]
    # 防空转：目录缺失或路径漂移时让测试显式失败，而不是零扫描静默通过
    assert files, f"未扫描到 workspace contracts 源文件，请检查路径: {contracts_package}"
    violations = [
        f"{path}: {module}"
        for path in files
        for module in _imports_of(path)
        if module.startswith("hivememory")
        and not module.startswith("hivememory.core")
        and not module.startswith(contracts_module)
    ]
    assert violations == []


def test_workspace_shared_facilities_do_not_import_capability():
    """意图与派生读取设施不得反向依赖能力入口，避免产生第二个编排所有者。"""
    packages = [WORKSPACE_PACKAGE / name for name in ("intents", "cache", "resolution")]
    violations = [
        f"{path}: {target}"
        for package in packages
        for path in package.rglob("*.py")
        for target in _import_targets_of(path)
        if target == "hivememory.workspace.capability"
        or target.startswith("hivememory.workspace.capability.")
    ]
    assert violations == []
