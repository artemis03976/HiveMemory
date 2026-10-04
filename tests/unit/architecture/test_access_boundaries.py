"""A1 访问边界的架构级静态约束（AST 扫描，补充包分层泛化规则）。

访问 context 是授权点之间的进程内凭据：只有 workspace 的认证入口、guard
与授权点（能力层、任务进程、注册入口、server 的 HTTP 入口）接触它；
Gateway、Patchouli、Alice 等资源 owner 与引擎位于授权点以下，只流动授权
点组装的 ``IdentityScope``（A1 访问边界返工第 4.1 节，身份与访问体系
不变量 2/4）。包分层测试只能证明"子系统之间只导入 contracts"；本模块把
"谁允许导入 guard 与认证入口"固化为精确清单，防止授权点边界被绕开。
context 类型本身经 ``core.access`` 是依赖中立的值类型，其导入不受本约束
（兑现只能经 guard，拿到类型无法取出身份）。
"""

from __future__ import annotations

from pathlib import Path

from tests.unit.architecture.test_package_layers import SRC_ROOT, _source_files, _top_package

# 允许导入 guard/认证入口的顶层包：workspace 自身、组合根（装配注入）与
# HTTP 入口（声明认证、请求级 context）。
ACCESS_AWARE_PACKAGES = {"workspace", "system", "server"}

# guard 与认证入口的宿主模块：其余包一律不得导入。
GUARD_MODULES = (
    "hivememory.workspace.access",
    "hivememory.workspace.authentication",
)


def _imports_guard(path: Path) -> bool:
    """源文件是否导入 guard/guard 宿主模块（含 TYPE_CHECKING 与函数内导入）。"""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith(GUARD_MODULES) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.startswith(GUARD_MODULES):
                return True
            if node.level > 0 and _top_package(path) == "workspace":
                # workspace 包内的相对导入：解析模块名后按前缀判断。
                parts = list(path.parent.relative_to(SRC_ROOT).parts)
                keep = len(parts) - (node.level - 1)
                base = parts[: max(keep, 0)]
                module = ".".join(
                    ["hivememory", *base, *(node.module.split(".") if node.module else [])]
                )
                if module.startswith(GUARD_MODULES):
                    return True
    return False


def test_only_access_aware_packages_import_the_guard():
    """访问凭据与 guard 只出现在 workspace、组合根与 HTTP 入口。"""
    violations = [
        path.relative_to(SRC_ROOT).as_posix()
        for path in _source_files()
        if _top_package(path) not in ACCESS_AWARE_PACKAGES and _imports_guard(path)
    ]
    assert violations == []


def test_workspace_access_layer_stays_below_process_and_capability():
    """workspace 内部依赖方向：process/capability 依赖 access，access 不反向依赖。

    guard 是授权点的共享检查设施，位于进程编排与能力层之下；它不导入
    任务进程或能力层，避免授权设施反向持有业务编排（plan 第 4.1 节）。
    """
    access_files = [
        path
        for path in _source_files()
        if _top_package(path) == "workspace"
        and path.relative_to(SRC_ROOT).as_posix()
        in {"workspace/access.py", "workspace/registry.py"}
    ]
    assert {p.name for p in access_files} == {"access.py", "registry.py"}
    for path in access_files:
        text = path.read_text(encoding="utf-8")
        assert "hivememory.workspace.process" not in text
        assert "hivememory.workspace.capability" not in text
