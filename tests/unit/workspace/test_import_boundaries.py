"""workspace 包静态依赖边界的单元测试。

保护父计划 4.2 节的依赖方向：``workspace`` 只能依赖资源端口、scope/DTO、
Policy 与明确的 Patchouli provider；不得导入 AliceRuntime、AgentRuntime
或 MTP handler。以 AST 静态扫描直接 import（含相对导入）判定；动态
``importlib`` 间接导入与传递依赖（workspace→patchouli→…）不在本检查
范围内，由 code review 与集成测试覆盖。
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "hivememory"
WORKSPACE_PACKAGE = SRC_ROOT / "workspace"

FORBIDDEN_PREFIXES = (
    "hivememory.alice",
    "hivememory.agent_runtime",
)

ALLOWED_TOP_LEVEL = (
    "hivememory.core",
    "hivememory.patchouli",
    "hivememory.engines.retrieval.policy",
    "hivememory.workspace",
)


def _package_parts(path: Path) -> list[str]:
    """计算模块在 hivememory 包内的层级（不含模块名）。"""
    return list(path.parent.relative_to(SRC_ROOT).parts)


def _resolve_relative(path: Path, node: ast.ImportFrom) -> str:
    """把相对导入解析为 hivememory 内的绝对模块名（逃逸包时返回空串）。"""
    parts = _package_parts(path)
    # level=1 表示当前包，每多一级向上一层
    keep = len(parts) - (node.level - 1)
    if keep <= 0:
        return ""
    base = parts[:keep]
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
            else:
                resolved = _resolve_relative(path, node)
                if resolved:
                    modules.append(resolved)
    return modules


def _workspace_sources() -> list[Path]:
    files = sorted(WORKSPACE_PACKAGE.rglob("*.py"))
    # 防空转：目录缺失或路径漂移时让测试显式失败，而不是零扫描静默通过
    assert files, f"未扫描到 workspace 源文件，请检查路径: {WORKSPACE_PACKAGE}"
    return files


def test_workspace_package_never_imports_alice_or_agent_runtime():
    """workspace 包的任何模块不得 import alice / agent_runtime。"""
    violations: list[str] = []
    for path in _workspace_sources():
        for module in _imports_of(path):
            if module.startswith(FORBIDDEN_PREFIXES):
                violations.append(f"{path}: {module}")
    assert violations == [], f"workspace 包出现反向依赖: {violations}"


def test_workspace_package_imports_stay_within_allowed_boundaries():
    """workspace 包的 hivememory 内部 import 只允许 core / patchouli / 自身。"""
    violations: list[str] = []
    for path in _workspace_sources():
        for module in _imports_of(path):
            if not module.startswith("hivememory"):
                continue
            if not module.startswith(ALLOWED_TOP_LEVEL):
                violations.append(f"{path}: {module}")
    assert violations == [], f"workspace 包出现边界外依赖: {violations}"
