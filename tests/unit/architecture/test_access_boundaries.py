"""A1 访问边界的架构级静态约束（AST 扫描，补充包分层泛化规则）。

访问 context 是授权点之间的进程内凭据：只有 workspace 的认证入口、认证
一侧、操作授权者与授权点（能力层、任务进程、注册入口、server 的 HTTP
入口）接触它；Gateway、Patchouli、Alice 等资源 owner 与引擎位于授权点
以下，只流动授权点组装的 ``IdentityScope``（A1 访问边界返工第 4.1 节，
身份与访问体系不变量 2/4）。包分层测试只能证明"子系统之间只导入
contracts"；本模块把认证与授权两侧的依赖方向固化为精确清单：

- 认证与授权模块（``workspace.authentication`` / ``workspace.authorization``）
  只出现在 workspace、组合根与 HTTP 入口；
- 能力层、``TaskProcess`` 与 CPU 分配是操作授权者（``authorization``）的
  调用方，不导入认证网关与 ``WorkspaceAuthenticator``；
- 操作授权者不导入任务进程与能力层。

context 类型本身经 ``core.access`` 是依赖中立的值类型，其导入不受本约束
（兑现只能经授权者与认证一侧，拿到类型无法取出身份）。
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.unit.architecture.test_package_layers import SRC_ROOT, _internal_imports

# 允许导入认证与授权模块的顶层包：workspace 自身、组合根（装配注入）与
# HTTP 入口（声明认证、请求级 context、进程句柄）。
ACCESS_AWARE_PACKAGES = {"workspace", "system", "server"}

AUTH_MODULES = (
    "hivememory.workspace.authentication",
    "hivememory.workspace.authorization",
)

# 只经操作授权者授权的授权点文件（不含注册入口：它另持认证网关）。
AUTHORIZER_ONLY_FILES = frozenset(
    {
        "workspace/capability/memory.py",
        "workspace/capability/agent_profiles.py",
        "workspace/capability/topic.py",
        "workspace/capability/memory_tasks.py",
        "workspace/capability/assets.py",
        "workspace/process/task_process.py",
        "workspace/process/allocation.py",
    }
)


def _top_package(path: Path) -> str:
    return path.relative_to(SRC_ROOT).parts[0]


def _source_files() -> list[Path]:
    files = [p for p in sorted(SRC_ROOT.rglob("*.py")) if "__pycache__" not in p.parts]
    assert files, f"未扫描到源文件，请检查路径: {SRC_ROOT}"
    return files


def _imports_any(path: Path, prefixes: tuple[str, ...]) -> bool:
    """源文件是否导入任一模块前缀（含 TYPE_CHECKING 与函数内导入）。"""
    return any(module.startswith(prefixes) for module in _internal_imports(path))


def test_auth_and_authz_imported_only_by_access_aware_packages():
    """认证与授权模块只出现在 workspace、组合根与 HTTP 入口。"""
    violations = [
        path.relative_to(SRC_ROOT).as_posix()
        for path in _source_files()
        if _top_package(path) not in ACCESS_AWARE_PACKAGES and _imports_any(path, AUTH_MODULES)
    ]
    assert violations == []


def test_capability_and_process_do_not_import_the_auth_side():
    """能力层、TaskProcess 与 CPU 分配只经操作授权者授权，不导入认证一侧。

    授权点不接触签发、失效与授予记录：认证网关与 ``WorkspaceAuthenticator``
    只属于运行持有者（server、注册入口）与组合根。
    """
    scoped_files = [
        path
        for path in _source_files()
        if path.relative_to(SRC_ROOT).as_posix() in AUTHORIZER_ONLY_FILES
    ]
    # 防空转：清单漂移时显式失败，而不是零扫描静默通过。
    assert {p.relative_to(SRC_ROOT).as_posix() for p in scoped_files} == AUTHORIZER_ONLY_FILES
    violations = [
        path.relative_to(SRC_ROOT).as_posix()
        for path in scoped_files
        if _imports_any(path, ("hivememory.workspace.authentication",))
    ]
    assert violations == []


def test_authorizer_imports_neither_process_nor_capability():
    """操作授权者不导入任务进程与能力层（依赖方向：process → capability → 授权者）。"""
    imports = _internal_imports(SRC_ROOT / "workspace" / "authorization.py")
    offenders = [
        module
        for module in imports
        if module.startswith(
            (
                "hivememory.workspace.process",
                "hivememory.workspace.capability",
            )
        )
    ]
    assert offenders == []


def test_cpu_execution_identity_is_only_called_from_cpu_allocation():
    """CPU 执行身份（过渡）只能由任务进程的 CPU 分配调用（A1 返工第 9 节）。

    ``cpu_execution_identity`` 不检查 operation，调用面必须收敛：AST 扫描
    全部生产代码，方法调用只允许出现在 ``workspace/process/allocation.py``
    （CPU 分配）与 ``workspace/authorization.py``（定义处）。
    """
    offenders: list[str] = []
    for path in _source_files():
        relative = path.relative_to(SRC_ROOT).as_posix()
        if relative in {"workspace/process/allocation.py", "workspace/authorization.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "cpu_execution_identity":
                offenders.append(relative)
    assert offenders == []
