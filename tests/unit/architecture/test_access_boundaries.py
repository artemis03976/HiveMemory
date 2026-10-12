"""A1 访问边界的架构级静态约束（AST 扫描，补充包分层泛化规则）。

访问 context 是密封的进程内凭据：workspace 的认证一侧、授权点与运行持有者
接触它；CPU 只持派生的线程 context，不持进程 context。Gateway、Patchouli
等资源 owner 与引擎位于授权点以下，资源调用只流动授权点组装的
``IdentityScope``；AliceCPU 将线程 context 绑定为操作提交函数，Agent loop
与执行适配器只持提交函数和观测标签（A1 访问边界返工第 4.1 节，
身份与访问体系不变量 2/4）。包分层测试只能证明"子系统之间只导入
contracts"；本模块把认证与授权两侧的依赖方向固化为精确清单：

- 认证与授权模块（``workspace.authentication`` / ``workspace.authorization``）
  只出现在 workspace、组合根与 HTTP 入口；
- 能力层、任务进程执行器（``TaskProcessRunner``）与 CPU 分配是操作授权者
  （``authorization``）的调用方，不导入认证网关与 ``WorkspaceAuthenticator``；
- 操作授权者不导入任务进程与能力层；认证一侧与操作授权者互不导入，只经
  context 这个类型发生联系（I-10 的 2026-10-04 补充）；
- context 是密封凭据：签发（``_seal``）与撤销（``_revoke``）只在认证一侧
  调用，读取授予内容（``_unseal``）只在操作授权者与认证一侧的派生、
  撤销校验和诊断查询调用。

context 类型本身经 ``core.access`` 是依赖中立的值类型，其导入不受本约束
（持有它只能把它交给授权点，读取内容的私有接口受上述清单约束）。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args

from pydantic import BaseModel

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
        "workspace/capability/actor/memory.py",
        "workspace/capability/operations.py",
        "workspace/capability/actor/agent_profiles.py",
        "workspace/capability/root/memory.py",
        "workspace/capability/root/agent_profiles.py",
        "workspace/capability/root/topic.py",
        "workspace/capability/root/memory_tasks.py",
        "workspace/capability/root/assets.py",
        "workspace/process/runner.py",
        "workspace/process/allocation.py",
        "workspace/process/threads.py",
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

    执行器只经窄化生命周期端口派生与撤销线程，认证网关与
    ``WorkspaceAuthenticator`` 仍只由注册入口、server 与组合根导入。
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


def test_stage_authorization_is_only_called_by_task_processes():
    """阶段授权只供任务进程的系统编排使用，不能成为 actor 操作的旁路。"""
    call_sites: set[str] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Attribute) and node.attr == "authorize_stage"
            for node in ast.walk(tree)
        ):
            call_sites.add(path.relative_to(SRC_ROOT).as_posix())

    # 精确清单同时防止调用面扩散和扫描空转；方法引用也属于调用面。
    assert call_sites == {
        "workspace/process/runner.py",
        "workspace/process/allocation.py",
    }


def test_stage_operation_snapshot_is_only_read_by_task_process():
    """操作快照只供进程分配与派生读取，不能成为 actor API 或 CPU 的授权旁路。"""
    assert _private_call_sites("get_stage_operations") == {
        "workspace/process/allocation.py",
        "workspace/process/threads.py",
    }


def test_thread_process_binding_queries_stay_inside_capability():
    """线程的进程关联只供意图登记与入口补偿，不能扩散到 CPU 或资源 owner。"""
    assert _private_call_sites("get_thread_process_id") == {
        "workspace/capability/operations.py",
        "workspace/capability/actor/memory.py",
    }


def test_thread_lifecycle_is_only_consumed_by_process_threads():
    """派生与撤销集中在进程线程管理，CPU 不取得签发能力。"""
    for method in ("derive_thread_context", "derive_child_thread_context", "revoke_thread_context"):
        assert _private_call_sites(method) == {
            "workspace/authentication.py",
            "workspace/process/threads.py",
        }


def test_thread_authorization_is_only_used_by_actor_api():
    """只有 actor 能力从线程 context 组装资源 scope，CPU 只提交请求。"""
    assert _private_call_sites("authorize_thread_operation") == {
        "workspace/capability/actor/memory.py",
        "workspace/capability/actor/agent_profiles.py",
    }
    assert _private_call_sites("validate_thread_context") == {
        "workspace/capability/operations.py",
        "workspace/process/threads.py",
    }


def _model_field_types(model: type[BaseModel]) -> list[tuple[str, object]]:
    """展开模型字段标注中的联合、泛型参数与嵌套模型，返回（字段路径，类型）。"""
    found: list[tuple[str, object]] = []
    pending: list[type[BaseModel]] = [model]
    visited: set[type[BaseModel]] = set()
    while pending:
        current = pending.pop()
        visited.add(current)
        for name, field in current.model_fields.items():
            annotations = [field.annotation]
            while annotations:
                annotation = annotations.pop()
                found.append((f"{current.__name__}.{name}", annotation))
                annotations.extend(get_args(annotation))
                nested = isinstance(annotation, type) and issubclass(annotation, BaseModel)
                pending.extend([annotation] if nested and annotation not in visited else [])
    return found


def test_cpu_manifest_cannot_carry_access_context():
    """输入清单（含嵌套模型）的身份只有观测标签，线程 context 必须走 CPU 的独立参数。"""
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import ExecutionLabels
    from hivememory.workspace.contracts import CPUInputManifest

    field_types = _model_field_types(CPUInputManifest)

    assert ("CPUInputManifest.labels", ExecutionLabels) in field_types
    assert [path for path, kind in field_types if kind is WorkspaceAccessContext] == []


def test_removed_execution_credentials_have_no_production_references():
    """线程 context 是唯一执行令牌，不能重新引入凭据表或第二份绑定。"""
    removed = {"ExecutionCredential", "ExecutionCredentialRegistry", "ExecutionBinding"}
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in removed:
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in removed:
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and any(a.name in removed for a in node.names):
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
    assert offenders == []


def test_actor_dispatch_does_not_unpack_requests_into_handlers():
    """操作入口仅保留统一 execute，六项 actor 请求直接分派到能力方法。"""
    path = SRC_ROOT / "workspace/capability/operations.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    entry = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert {
        n.name for n in entry.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    } == {"__init__", "execute"}
    dispatch = next(
        n.value
        for n in ast.walk(entry)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Attribute)
        and n.target.attr == "_handlers"
    )
    assert isinstance(dispatch, ast.Dict)
    assert {
        ast.unparse(key): ast.unparse(value) for key, value in zip(dispatch.keys, dispatch.values)
    } == {
        "SubmitWriteIntentRequest": "memory.submit_write_intent",
        "SubmitUpdateIntentRequest": "memory.submit_update_intent",
        "ResolveReferencesRequest": "memory.resolve_references",
        "ResolveRunTargetRequest": "memory.resolve_run_target",
        "RetrieveRequest": "memory.retrieve",
        "ResolveCallTargetRequest": "agent.resolve_call_target",
    }


def test_profile_permission_symbols_have_no_production_references():
    """生产代码不能重新从 Profile 取得权限，也不能重新注入两份旧白名单。"""
    removed_symbols = {
        "allowed_mtp_verbs",
        "allowed_sys_tools",
        "is_verb_allowed",
        "is_tool_allowed",
        "get_verb_set",
        "get_tool_set",
        "OMNI_DOLL_ALLOWED_MTP_VERBS",
        "OMNI_DOLL_ALLOWED_SYS_TOOLS",
    }
    root = SRC_ROOT.parent.parent
    paths = [*_source_files(), *(root / "scripts").rglob("*.py")]
    offenders = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            match node:
                case ast.Name(id=name) | ast.Attribute(attr=name):
                    symbol = name
                case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                    symbol = name
                case ast.Constant(value=str() as value):
                    symbol = value
                case _:
                    continue
            if symbol in removed_symbols:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}:{symbol}")
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


def test_context_content_is_only_read_by_authorizer_and_authentication():
    """授予内容只在操作授权者与认证侧读取：CPU 与其他代码只能传递 context。"""
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


def test_root_authorization_is_only_used_by_root_api():
    """root 授权只能供管理 API 使用，不能被 actor 操作或阶段编排借用。"""
    call_sites: set[str] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Attribute) and node.attr == "authorize_root"
            for node in ast.walk(tree)
        ):
            call_sites.add(path.relative_to(SRC_ROOT).as_posix())
    assert call_sites == {
        "workspace/capability/root/memory.py",
        "workspace/capability/root/agent_profiles.py",
        "workspace/capability/root/topic.py",
        "workspace/capability/root/memory_tasks.py",
        "workspace/capability/root/assets.py",
    }
