"""访问边界的架构级静态约束（AST 扫描，补充包分层泛化规则）。

认证得到工作 actor（``core.access.WorkingActor``）：执行者与系统的 root 工作
actor 留在任务进程记录中，子工作 actor 留在子线程记录中，入口为一次请求认证
得到的工作 actor 只留在入口调用链内。CPU 只持不透明的线程句柄，经提交函数
提交操作；Gateway、Patchouli 等资源 owner 与引擎位于授权点以下，资源调用只
流动授权点组装的 ``IdentityScope``。工作 actor 是普通值，进程内的防误用由
本模块限定的构造、持有与调用位置承担。包分层测试只能证明"子系统之间只导入
contracts"；本模块把认证与授权两侧的依赖方向固化为精确清单：

- 认证与授权模块（``workspace.authentication`` / ``workspace.authorization``）
  只出现在 workspace、组合根与 HTTP 入口；
- 能力层、任务进程执行器与 CPU 分配是操作授权者的调用方，不导入认证一侧；
- 操作授权者不导入任务进程与能力层；认证一侧与操作授权者互不导入，只经
  工作 actor 这个类型发生联系；
- 工作 actor 只由认证一侧构造，``IdentityScope`` 只由授权者（与登记的资产
  桥接例外）构造；两个授权方法与系统自身的 principal 各自限定使用位置。

工作 actor 类型本身经 ``core.access`` 是依赖中立的值类型，其导入不受上述
模块清单约束，持有位置由本模块的引用扫描约束。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args, get_type_hints

from pydantic import BaseModel

from tests.unit.architecture.test_package_layers import SRC_ROOT, _internal_imports

# 允许导入认证与授权模块的顶层包：workspace 自身、组合根（装配注入）与
# HTTP 入口（声明认证、请求的工作 actor、进程句柄）。
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
        "workspace/capability/root/interaction.py",
        "workspace/capability/root/memory.py",
        "workspace/capability/root/agent_profiles.py",
        "workspace/capability/root/topic.py",
        "workspace/capability/root/session.py",
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
    """能力层、任务进程执行器、CPU 分配与线程管理只经操作授权者授权，不导入认证一侧。

    线程管理只经 ``core.access`` 的子工作 actor 准入端口为被调用方做准入，认证
    网关与 ``WorkspaceAuthenticator`` 只由注册入口、Import Bus 服务、server 与
    组合根导入。
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


def test_agent_authorization_is_only_called_by_actor_api_and_task_process():
    """actor 操作授权只在 actor API 与任务进程 prepare 的检索中调用，其他代码不能借用白名单。"""
    assert _attribute_sites("authorize_agent") == {
        "workspace/capability/actor/memory.py",
        "workspace/capability/actor/agent_profiles.py",
        "workspace/process/runner.py",
    }


def test_root_authorization_is_only_called_by_root_api_task_process_and_import_bus():
    """root 授权只供 root API、任务进程的系统阶段与取消，以及 Import Bus 服务使用。"""
    assert _attribute_sites("authorize_root") == {
        "workspace/capability/root/agent_profiles.py",
        "workspace/capability/root/assets.py",
        "workspace/capability/root/interaction.py",
        "workspace/capability/root/memory.py",
        "workspace/capability/root/memory_tasks.py",
        "workspace/capability/root/session.py",
        "workspace/capability/root/topic.py",
        "workspace/process/allocation.py",
        "workspace/process/runner.py",
        "workspace/process/service.py",
        "system/application/import_bus_service.py",
    }


def test_allowed_operations_query_is_only_called_by_task_process():
    """操作集合的查询只在任务进程中调用（线程描述的快照、prepare 是否检索），不能成为授权旁路。

    同名的线程描述字段与访问记录字段是普通读取，不在此列。
    """
    assert _method_call_sites("allowed_operations") == {
        "workspace/process/threads.py",
        "workspace/process/runner.py",
    }


def test_sub_actor_admission_is_only_consumed_by_process_threads():
    """子工作 actor 的准入集中在线程管理：CPU 与能力层不能为任意 Agent 取得工作 actor。"""
    assert _attribute_sites("admit_sub_actor") == {
        "workspace/authentication.py",
        "workspace/process/threads.py",
    }


def test_system_principal_is_only_injected_by_the_composition_root():
    """系统自身的 principal 只由组合根注入任务进程的注册入口与 Import Bus 服务。

    它能声明 system：在别处使用就能绕过入口的认证取得 root 工作 actor。
    """
    sites = _name_sites("SYSTEM_PRINCIPAL_ID") - {
        "system/access/__init__.py",
        "system/access/registry.py",
    }
    assert sites == {"system/assembler.py"}


def test_profile_management_read_is_only_requested_by_cpu_allocation():
    """跳过 Agent 可见性的 Profile 管理读取只由 CPU 分配以 root scope 发起。

    CALL 等 actor 操作必须按调用方的可见性解析图纸；资源 owner（Patchouli）
    内部的管理读取不在扫描范围。
    """
    sites: set[str] = set()
    for path in _source_files():
        if _top_package(path) in {"patchouli", "engines"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.keyword)
            and node.arg == "enforce_actor_visibility"
            and not (isinstance(node.value, ast.Constant) and node.value.value is True)
            and not (
                isinstance(node.value, ast.Name) and node.value.id == "enforce_actor_visibility"
            )
            for node in ast.walk(tree)
        ):
            sites.add(path.relative_to(SRC_ROOT).as_posix())
    assert sites == {"workspace/process/allocation.py"}


def test_working_actor_is_only_constructed_by_the_authentication_side():
    """工作 actor 只由认证一侧构造：授权点、入口与 CPU 都只能接收认证得到的值。"""
    assert _constructor_sites("WorkingActor") == {"workspace/authentication.py"}


def test_identity_scope_is_only_constructed_by_the_authorizer_and_asset_bridge():
    """``IdentityScope`` 只由授权者组装；资产桥接在一次租借内组装 scope 是登记的例外。"""
    assert _constructor_sites("IdentityScope") == {
        "workspace/authorization.py",
        "system/services/asset_materialization_reader.py",
    }


def test_resource_owners_and_cpu_do_not_reference_working_actor():
    """Alice、agent_runtime、Patchouli、Gateway 与引擎不引用工作 actor：身份只以 scope 下行。"""
    packages = {"alice", "agent_runtime", "patchouli", "gateway", "engines"}
    files = [path for path in _source_files() if _top_package(path) in packages]
    assert files, "未扫描到资源 owner 与 CPU 的源文件"
    offenders = [
        f"{path.relative_to(SRC_ROOT).as_posix()}:{line}"
        for path in files
        for line in _symbol_lines(path, "WorkingActor")
    ]
    assert offenders == []


def test_actor_api_is_reached_only_through_the_operation_entry():
    """actor API 只经操作入口调用：组合根装配它，门面与各入口都不导入它。"""
    actor_package = "workspace/capability/actor/"
    importers = {
        relative
        for path in _source_files()
        if not (relative := path.relative_to(SRC_ROOT).as_posix()).startswith(actor_package)
        and _imports_any(path, ("hivememory.workspace.capability.actor",))
    }
    assert importers == {"workspace/capability/operations.py", "system/assembler.py"}


def test_request_level_submission_is_only_used_by_mcp_entry():
    """请求级提交不做线程存活判定，只有不建进程的 MCP 入口以入口认证的工作 actor 调用它。"""
    assert _attribute_sites("execute_request") == {"mcp_server/server.py"}


def test_mcp_entry_reaches_capabilities_only_through_the_facade():
    """MCP 入口不导入 actor API、root API、任务进程与 Import Bus 的服务和编排器。

    操作入口与 Import Bus 的落位都经门面取得；Import Bus 只导入在途轮次的
    输入模型，用来接收连接器补上的轮次信息。
    """
    forbidden = (
        "hivememory.workspace.capability.actor",
        "hivememory.workspace.capability.root",
        "hivememory.workspace.process",
        "hivememory.system.application.import_bus_service",
        "hivememory.system.services.import_bus",
    )
    allowed = {"hivememory.system.services.import_bus.models"}
    files = sorted((SRC_ROOT / "mcp_server").rglob("*.py"))
    assert files, "未扫描到 MCP 入口的源文件"
    offenders = [
        f"{path.relative_to(SRC_ROOT).as_posix()}: {module}"
        for path in files
        for module in _internal_imports(path)
        if module.startswith(forbidden) and module not in allowed
    ]
    assert offenders == []


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


def test_cpu_visible_objects_cannot_carry_working_actor_or_scope():
    """CPU 可见的对象不含工作 actor 与 ``IdentityScope``：输入清单（含嵌套模型）、线程描述与提交函数交出的描述。

    线程句柄只在执行线程中交给 CPU 驱动，由驱动绑定进提交函数内部。
    """
    from hivememory.core.access import WorkingActor
    from hivememory.core.models import ExecutionLabels, IdentityScope
    from hivememory.workspace.contracts import (
        CPUInputManifest,
        OperationSubmitter,
        ThreadDescriptor,
    )

    identity_types = (WorkingActor, IdentityScope)
    field_types = _model_field_types(CPUInputManifest)
    assert ("CPUInputManifest.labels", ExecutionLabels) in field_types
    assert [path for path, kind in field_types if kind in identity_types] == []

    descriptor_types = get_type_hints(ThreadDescriptor)
    assert descriptor_types["labels"] is ExecutionLabels
    assert not set(descriptor_types.values()) & set(identity_types)
    thread_property = OperationSubmitter.__dict__["thread"]
    assert get_type_hints(thread_property.fget)["return"] is ThreadDescriptor


def test_actor_dispatch_does_not_unpack_requests_into_handlers():
    """入口只把记忆请求直接分派到能力方法；请求级提交共用分派表，派生与结束是生命周期方法。"""
    path = SRC_ROOT / "workspace/capability/operations.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    entry = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert {
        n.name for n in entry.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    } == {"__init__", "execute", "execute_request", "derive_child_thread", "end_thread"}
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
    }


def _attribute_sites(attribute: str) -> set[str]:
    """生产代码中访问某个属性名的源文件（相对路径）；方法引用也属于调用面。"""
    sites: set[str] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Attribute) and node.attr == attribute for node in ast.walk(tree)
        ):
            sites.add(path.relative_to(SRC_ROOT).as_posix())
    return sites


def _method_call_sites(method: str) -> set[str]:
    """生产代码中以 ``obj.method(...)`` 调用某个方法的源文件（相对路径）。"""
    sites: set[str] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            for node in ast.walk(tree)
        ):
            sites.add(path.relative_to(SRC_ROOT).as_posix())
    return sites


def _symbol_lines(path: Path, symbol: str) -> list[int]:
    """源文件中引用某个名称的行：名称、属性访问与导入别名（含 TYPE_CHECKING 导入）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == symbol
            or isinstance(node, ast.Attribute)
            and node.attr == symbol
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and any(alias.name == symbol for alias in node.names):
            lines.append(node.lineno)
    return lines


def _name_sites(symbol: str) -> set[str]:
    """生产代码中引用某个名称的源文件（相对路径）。"""
    return {
        path.relative_to(SRC_ROOT).as_posix()
        for path in _source_files()
        if _symbol_lines(path, symbol)
    }


def _constructor_sites(class_name: str) -> set[str]:
    """生产代码中以 ``Name(...)`` 或 ``module.Name(...)`` 直接构造某个类的源文件。"""
    sites: set[str] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Name) and func.id == class_name) or (
                isinstance(func, ast.Attribute) and func.attr == class_name
            ):
                sites.add(path.relative_to(SRC_ROOT).as_posix())
    return sites


def test_authentication_and_authorization_do_not_import_each_other():
    """认证一侧与操作授权者互不依赖：授权者读取工作 actor，不经认证一侧。"""
    authorization_imports = _internal_imports(SRC_ROOT / "workspace" / "authorization.py")
    authentication_imports = _internal_imports(SRC_ROOT / "workspace" / "authentication.py")

    assert not any(
        module.startswith("hivememory.workspace.authentication") for module in authorization_imports
    )
    assert not any(
        module.startswith("hivememory.workspace.authorization") for module in authentication_imports
    )


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
    """Alice 与 MTP 操作适配器不能从观测标签重建资源访问身份。"""
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


def test_alice_runtime_cannot_import_thread_handle_or_raw_derivation_results():
    """只有 CPU 驱动接触线程句柄与派生原始结果，frame、CALL 和 MTP 只持提交函数。"""
    forbidden = {"ThreadHandle", "ExecutionThread", "OperationEntry"}
    # CPU 驱动把执行线程绑定为提交函数，Alice 组合入口只把操作入口交给驱动。
    driver_files = {SRC_ROOT / "alice/interface/cpu.py", SRC_ROOT / "alice/system.py"}
    offenders = []
    files = [*(SRC_ROOT / "alice").rglob("*.py"), *(SRC_ROOT / "agent_runtime").rglob("*.py")]
    for path in sorted(set(files) - driver_files):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}:{node.id}")
            elif isinstance(node, ast.ImportFrom) and any(
                alias.name in forbidden for alias in node.names
            ):
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
    assert offenders == []


def test_intent_dispatch_uses_only_public_submission_path():
    """主动生成只从意图提交服务进入。"""
    calls = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "request"
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Attribute)
                    and node.args[0].attr == "GENERATION_SUBMIT_ACTIVE"
                ):
                    calls.append(path.relative_to(SRC_ROOT).as_posix())
    assert calls == ["patchouli/application/memory_intent_submission_service.py"]


def test_capability_and_process_entries_take_identity_only_as_working_actor():
    """能力层、任务进程与 Import Bus 的公共方法不接收调用方组装的 IdentityScope。

    身份以工作 actor 交给它们，由它们在入口授权；显式目标 workspace 必须与
    工作 actor 配对出现。
    """
    import inspect

    from hivememory.system.application.import_bus_service import ImportBusService
    from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
    from hivememory.workspace.capability.actor.memory import ActorMemoryService
    from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService
    from hivememory.workspace.capability.root.assets import RootWorkspaceAssetService
    from hivememory.workspace.capability.root.interaction import RootInteractionService
    from hivememory.workspace.capability.root.memory import RootMemoryService
    from hivememory.workspace.capability.root.memory_tasks import RootMemoryTaskService
    from hivememory.workspace.capability.root.session import RootSessionService
    from hivememory.workspace.capability.root.topic import RootTopicService
    from hivememory.workspace.process.service import TaskProcessService

    violations = []
    for service in (
        ImportBusService,
        ActorAgentService,
        ActorMemoryService,
        RootAgentProfileService,
        RootInteractionService,
        RootMemoryService,
        RootMemoryTaskService,
        RootSessionService,
        RootTopicService,
        RootWorkspaceAssetService,
        TaskProcessService,
    ):
        for name, method in inspect.getmembers(service, inspect.isfunction):
            if name.startswith("_"):
                continue
            parameters = inspect.signature(method).parameters
            if "identity_scope" in parameters:
                violations.append(f"{service.__name__}.{name} 接收 identity_scope")
            if "target_workspace" in parameters and "actor" not in parameters:
                violations.append(f"{service.__name__}.{name} 接收目标 workspace 却没有 actor")

    assert violations == []
