"""workspace 内部依赖边界的单元测试（读取能力与访问模块只依赖 core）。

包级分层规则（各层之间、子系统之间的依赖方向）由
``tests/unit/architecture/test_package_layers.py`` 统一守护；本文件只保护
workspace 包内更严格的约束：认证、准入、读取能力（cache / resolution /
runtime）与配置模块只依赖 core 与 workspace 自身——resolver 经 backing 协议
冷读，不导入总线、路由常量或任何子系统实现。``capability`` 与 ``assets``
作为能力层与资产设施，可依赖更低层的机制与适配器，不在此约束内。

以 AST 静态扫描直接 import（含相对导入）判定。
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "hivememory"
WORKSPACE_PACKAGE = SRC_ROOT / "workspace"
UNRESTRICTED_SUBPACKAGES = (WORKSPACE_PACKAGE / "capability", WORKSPACE_PACKAGE / "assets")
ALLOWED_INTERNAL = ("hivememory.core", "hivememory.workspace")


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


def _restricted_sources() -> list[Path]:
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
    """认证、准入、读取能力与配置模块只允许 import core / workspace 自身。"""
    violations = [
        f"{path}: {module}"
        for path in _restricted_sources()
        for module in _imports_of(path)
        if module.startswith("hivememory") and not module.startswith(ALLOWED_INTERNAL)
    ]
    assert violations == []
