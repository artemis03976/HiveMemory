"""workspace 包静态依赖边界的单元测试。

保护父计划 4.2 节的依赖方向：``workspace`` 只承担访问边界、cache、失效
和快照基础设施，只能依赖 core；不得导入 AliceRuntime、AgentRuntime、
MTP handler 或任何 Patchouli 实现（业务统一由 application/GlobalSystemBus
承接，Patchouli 反向消费本包基础设施）。以 AST 静态扫描直接 import（含
相对导入）判定；动态 ``importlib`` 间接导入与传递依赖不在本检查范围内，
由 code review 与集成测试覆盖。
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "hivememory"
WORKSPACE_PACKAGE = SRC_ROOT / "workspace"

FORBIDDEN_PREFIXES = (
    "hivememory.alice",
    "hivememory.agent_runtime",
    "hivememory.patchouli",
)

ALLOWED_INTERNAL = ("hivememory.core", "hivememory.workspace")


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
    """workspace 包的 hivememory 内部 import 只允许 core / 自身基础设施。"""
    violations: list[str] = []
    for path in _workspace_sources():
        for module in _imports_of(path):
            if not module.startswith("hivememory"):
                continue
            if not module.startswith(ALLOWED_INTERNAL):
                violations.append(f"{path}: {module}")
    assert violations == [], f"workspace 包出现边界外依赖: {violations}"


def test_patchouli_application_does_not_import_system_authentication():
    """包含 TYPE_CHECKING 注解在内，Patchouli 只消费 Workspace 的访问契约。"""
    files = sorted((SRC_ROOT / "patchouli" / "application").glob("*.py"))
    assert files, "未扫描到 Patchouli application 源文件"
    violations = [
        f"{path}: {module}"
        for path in files
        for module in _imports_of(path)
        if module.startswith("hivememory.system.access")
    ]
    assert violations == [], f"Patchouli application 依赖了 System 认证实现: {violations}"
