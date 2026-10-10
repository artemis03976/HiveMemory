"""A1 访问边界的架构级静态约束（AST 扫描，补充包分层泛化规则）。

访问 context 是授权点之间的进程内凭据：只有 workspace 的认证入口、认证
一侧、操作授权者与授权点（能力层、任务进程、注册入口、server 的 HTTP
入口）接触它；Gateway、Patchouli、Alice 等资源 owner 与引擎位于授权点
以下，资源调用只流动授权点组装的 ``IdentityScope``；Alice 与执行适配器
只持操作提交函数和观测标签（A1 访问边界返工第 4.1 节，
身份与访问体系不变量 2/4）。包分层测试只能证明"子系统之间只导入
contracts"；本模块把认证与授权两侧的依赖方向固化为精确清单：

- 认证与授权模块（``workspace.authentication`` / ``workspace.authorization``）
  只出现在 workspace、组合根与 HTTP 入口；
- 能力层、任务进程执行器（``TaskProcessRunner``）与 CPU 分配是操作授权者
  （``authorization``）的调用方，不导入认证网关与 ``WorkspaceAuthenticator``；
- 操作授权者不导入任务进程与能力层；认证一侧与操作授权者互不导入，只经
  context 这个类型发生联系（I-10 的 2026-10-04 补充）；
- context 是密封凭据：签发（``_seal``）与撤销（``_revoke``）只在认证一侧
  调用，读取授予内容（``_unseal``）只在操作授权者与认证一侧的诊断查询
  调用。

context 类型本身经 ``core.access`` 是依赖中立的值类型，其导入不受本约束
（持有它只能把它交给授权点，读取内容的私有接口受上述清单约束）。
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
        "workspace/capability/operations.py",
        "workspace/capability/agent_profiles.py",
        "workspace/capability/topic.py",
        "workspace/capability/memory_tasks.py",
        "workspace/capability/assets.py",
        "workspace/process/runner.py",
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
    """能力层、任务进程执行器与 CPU 分配只经操作授权者授权，不导入认证一侧。

    授权点不接触签发与撤销：认证网关与 ``WorkspaceAuthenticator``
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
    """操作授权者不导入任务进程与能力层，只被编排与能力授权点消费。"""
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


def test_cpu_execution_identity_has_no_definition_or_call_site():
    """过渡执行身份已删除，生产代码不能重新定义或调用其无 operation 路径。"""
    offenders: list[str] = []
    for path in _source_files():
        relative = path.relative_to(SRC_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "cpu_execution_identity"
                or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "cpu_execution_identity"
                or isinstance(node, ast.Name)
                and node.id == "cpu_execution_identity"
            ):
                offenders.append(f"{relative}:{node.lineno}")
    assert offenders == []


def _private_call_sites(attribute: str) -> set[str]:
    """生产代码中访问某个属性名的源文件（相对路径），不含定义处 core/access.py。"""
    sites: set[str] = set()
    for path in _source_files():
        relative = path.relative_to(SRC_ROOT).as_posix()
        if relative == "core/access.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Attribute) and node.attr == attribute for node in ast.walk(tree)
        ):
            sites.add(relative)
    return sites


def test_authentication_and_authorization_do_not_import_each_other():
    """认证一侧与操作授权者互不依赖：授权者读取密封 context，不经认证一侧。"""
    authorization_imports = _internal_imports(SRC_ROOT / "workspace" / "authorization.py")
    authentication_imports = _internal_imports(SRC_ROOT / "workspace" / "authentication.py")

    assert not any(
        module.startswith("hivememory.workspace.authentication") for module in authorization_imports
    )
    assert not any(
        module.startswith("hivememory.workspace.authorization") for module in authentication_imports
    )


def test_context_issue_and_revoke_are_only_called_by_the_authentication_side():
    """密封 context 的签发与撤销只由认证一侧调用：授权点与入口不能签发凭据。"""
    assert _private_call_sites("_seal") == {"workspace/authentication.py"}
    assert _private_call_sites("_revoke") == {"workspace/authentication.py"}


def test_context_content_is_only_read_by_authorizer_and_diagnostics():
    """授予内容只在操作授权者与认证一侧的诊断查询读取：其余代码只传递 context。"""
    assert _private_call_sites("_unseal") == {
        "workspace/authorization.py",
        "workspace/authentication.py",
    }


def test_alice_resource_operations_do_not_reference_patchouli_routes():
    """Alice 与 MTP 适配器只提交操作请求，不保留资源 owner 的 RPC 路径。"""
    files = [path for path in _source_files() if _top_package(path) in {"alice", "agent_runtime"}]
    assert files, "未扫描到 Alice 与 agent_runtime 源文件"
    offenders: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("PATCHOULI_")
                or isinstance(node, ast.Name)
                and node.id.startswith("PATCHOULI_")
                or isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("patchouli.public.")
            ):
                offenders.append(f"{path.relative_to(SRC_ROOT).as_posix()}:{node.lineno}")
    assert offenders == []


def test_alice_execution_does_not_import_or_reference_identity_scope():
    """Alice 与 MTP 执行适配器不能从观测标签重建资源访问身份。"""
    files = [path for path in _source_files() if _top_package(path) in {"alice", "agent_runtime"}]
    assert files, "未扫描到 Alice 与 agent_runtime 源文件"
    offenders: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and any(alias.name == "IdentityScope" for alias in node.names)
                or isinstance(node, ast.Name)
                and node.id == "IdentityScope"
                or isinstance(node, ast.Attribute)
                and node.attr == "IdentityScope"
            ):
                offenders.append(f"{path.relative_to(SRC_ROOT).as_posix()}:{node.lineno}")
    assert offenders == []
