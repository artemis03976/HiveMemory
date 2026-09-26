"""workspace 包静态依赖边界的单元测试（A2 §8 D-2 分层导入白名单）。

保护父计划 4.2 节与 A2 §8 D-2 的依赖方向：

- ``access`` / ``registry`` / ``cache`` / ``resolution`` / ``runtime``：只依赖
  core 与 workspace 自身——读取能力经 backing 协议冷读，不导入总线路由常量
  或 Patchouli 实现；
- ``capability``（自 ``system/application`` 迁入的能力层）：过渡期额外允许
  ``system.contracts`` / ``system.runtime`` / ``system.services`` /
  ``system.config``、``patchouli.contracts`` 与 ``utils``；
- 任何子包都不得导入 Alice、AgentRuntime、Gateway、engines 或 Patchouli
  非 contracts 包。

TODO(A5/A6)：能力层对 system 基础设施的依赖方向在 A5/A6 收口时重新整理，
届时收紧 ``CAPABILITY_EXTRA_ALLOWED``。

以 AST 静态扫描直接 import（含相对导入）判定；动态 ``importlib`` 间接导入与
传递依赖不在本检查范围内，由 code review 与集成测试覆盖。
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "hivememory"
WORKSPACE_PACKAGE = SRC_ROOT / "workspace"

CAPABILITY_PACKAGE = WORKSPACE_PACKAGE / "capability"

FORBIDDEN_PREFIXES = (
    "hivememory.alice",
    "hivememory.agent_runtime",
    "hivememory.gateway",
    "hivememory.engines",
)

ALLOWED_INTERNAL = ("hivememory.core", "hivememory.workspace")

# TODO(A5/A6)：过渡期分层白名单，仅对 workspace/capability 生效（A2 §8 D-2）。
CAPABILITY_EXTRA_ALLOWED = (
    "hivememory.system.contracts",
    "hivememory.system.runtime",
    "hivememory.system.services",
    "hivememory.system.config",
    "hivememory.patchouli.contracts",
    "hivememory.utils",
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


def _is_capability_module(path: Path) -> bool:
    return CAPABILITY_PACKAGE in path.parents


def test_workspace_package_never_imports_actor_or_library_internals():
    """workspace 任何子包都不得 import Alice/AgentRuntime/Gateway/engines 或 Patchouli 非 contracts 包。"""
    violations: list[str] = []
    for path in _workspace_sources():
        for module in _imports_of(path):
            library_internal = module.startswith("hivememory.patchouli") and not module.startswith(
                "hivememory.patchouli.contracts"
            )
            if module.startswith(FORBIDDEN_PREFIXES) or library_internal:
                violations.append(f"{path}: {module}")
    assert violations == [], f"workspace 包出现反向依赖: {violations}"


def test_workspace_core_subpackages_depend_only_on_core():
    """能力层以外的 workspace 模块只允许 import core / workspace 自身。"""
    violations: list[str] = []
    for path in _workspace_sources():
        if _is_capability_module(path):
            continue
        for module in _imports_of(path):
            if not module.startswith("hivememory"):
                continue
            if not module.startswith(ALLOWED_INTERNAL):
                violations.append(f"{path}: {module}")
    assert violations == [], f"workspace 包出现边界外依赖: {violations}"


def test_capability_layer_imports_stay_within_transitional_allowlist():
    """能力层只允许在 core / workspace 之外依赖过渡期白名单中的 system 与契约包。"""
    allowed = ALLOWED_INTERNAL + CAPABILITY_EXTRA_ALLOWED
    capability_sources = [path for path in _workspace_sources() if _is_capability_module(path)]
    assert capability_sources, f"未扫描到能力层源文件，请检查路径: {CAPABILITY_PACKAGE}"
    violations = [
        f"{path}: {module}"
        for path in capability_sources
        for module in _imports_of(path)
        if module.startswith("hivememory") and not module.startswith(allowed)
    ]
    assert violations == [], f"能力层出现白名单外依赖: {violations}"


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
