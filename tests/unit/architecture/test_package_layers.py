"""hivememory 顶层包分层依赖规则的单元测试（AST 静态扫描 + 导入隔离检查）。

分层（低 → 高）：

- L0 ``core`` / ``config`` / ``utils`` / ``i18n`` / ``_version``：模型、契约常量、错误类型、
  端口协议与配置段模型；
- L1 ``components``：进程内运行时机制（总线、调度器、work queue、运行时事件等）；
- L2 ``engines`` / ``infrastructure`` / ``prompts``：算法与外部技术适配器；
- L3 ``workspace`` / ``patchouli`` / ``gateway`` / ``alice`` + ``agent_runtime``：各子系统；
- L4 ``system``：组合根、门面与系统级能力；
- L5 ``server``：入口（传输层 adapter）。

规则：只能导入同层或更低层；L3 子系统之间只能导入对方公开的 ``contracts``
子包（``alice`` 与 ``agent_runtime`` 视为同一子系统）；根包只导入 ``_version``；
根配置与加载模块 ``hivememory.config.app`` 只供组合根（system）与入口（server）使用。
静态扫描包含函数内与 ``TYPE_CHECKING`` 导入；动态 ``importlib`` 不在范围内。
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "hivememory"

LAYERS: dict[str, int] = {
    "_version": 0,
    "core": 0,
    "config": 0,
    "utils": 0,
    "i18n": 0,
    "components": 1,
    "engines": 2,
    "infrastructure": 2,
    "prompts": 2,
    "workspace": 3,
    "patchouli": 3,
    "gateway": 3,
    "alice": 3,
    "agent_runtime": 3,
    "system": 4,
    "server": 5,
}
SUBSYSTEM_LAYER = 3
# 同一子系统的不同包：互相导入不受 contracts 限制。
SUBSYSTEM_GROUP = {"agent_runtime": "alice"}

ROOT_CONFIG_MODULE = "hivememory.config.app"
ROOT_CONFIG_CONSUMERS = {"system", "server"}

# 已知的向上依赖（engines 直接使用 Patchouli 存储层等），
# 属于分层重构之前就存在的耦合，另行处理。
# 断言为精确相等：新增违规或修复后未同步本清单都会失败。
KNOWN_UPWARD_IMPORTS = {
    ("engines/artifacts/document.py", "hivememory.patchouli.memory_library"),
    ("engines/artifacts/engine.py", "hivememory.patchouli.memory_library.stores"),
    ("engines/artifacts/interaction.py", "hivememory.patchouli.memory_library"),
    ("engines/artifacts/memory.py", "hivememory.patchouli.memory_library"),
    ("engines/gateway/interceptors.py", "hivememory.gateway.commands"),
    ("engines/generation/engine.py", "hivememory.patchouli.memory_library.stores"),
    ("engines/lifecycle/engine.py", "hivememory.patchouli.memory_library.stores"),
    ("engines/lifecycle/garbage_collector.py", "hivememory.patchouli.memory_library.library"),
    ("engines/lifecycle/reinforcement.py", "hivememory.patchouli.memory_library.stores"),
    ("engines/retrieval/retriever.py", "hivememory.patchouli.memory_library.stores"),
}


def _source_files() -> list[Path]:
    files = [p for p in sorted(SRC_ROOT.rglob("*.py")) if "__pycache__" not in p.parts]
    # 防空转：路径漂移时显式失败，而不是零扫描静默通过
    assert files, f"未扫描到源文件，请检查路径: {SRC_ROOT}"
    return files


def _top_package(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


def _resolve_relative(path: Path, node: ast.ImportFrom) -> str:
    """把相对导入解析为 hivememory 内的绝对模块名。"""
    parts = list(path.parent.relative_to(SRC_ROOT).parts)
    keep = len(parts) - (node.level - 1)
    base = parts[: max(keep, 0)]
    if node.module:
        base = [*base, *node.module.split(".")]
    return ".".join(["hivememory", *base])


def _internal_imports(path: Path) -> list[str]:
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
    return [m for m in modules if m.startswith("hivememory.")]


def _cross_package_imports() -> list[tuple[str, str, str, str]]:
    """返回 (相对路径, 源包, 目标包, 导入模块)，只含跨顶层包的导入。"""
    edges = []
    for path in _source_files():
        source = _top_package(path)
        for module in _internal_imports(path):
            target = module.split(".")[1]
            if target != source:
                edges.append((path.relative_to(SRC_ROOT).as_posix(), source, target, module))
    return edges


def test_every_top_level_package_is_assigned_a_layer():
    """新增顶层包必须先在分层表中登记，不能游离于依赖规则之外。"""
    packages = {_top_package(path) for path in _source_files()} - {"__init__"}
    assert packages - LAYERS.keys() == set()


def test_no_package_imports_a_higher_layer_beyond_known_debt():
    """任何包都不得导入更高层；向上依赖只允许已登记的存量耦合。"""
    upward = {
        (relative, module)
        for relative, source, target, module in _cross_package_imports()
        if source != "__init__" and LAYERS[target] > LAYERS[source]
    }
    assert upward == KNOWN_UPWARD_IMPORTS


def test_subsystems_only_import_each_others_contracts():
    """L3 子系统之间只能依赖对方公开的 contracts 子包，不得持有对方内部实现。"""
    violations = [
        f"{relative}: {module}"
        for relative, source, target, module in _cross_package_imports()
        if source != "__init__"
        and LAYERS[source] == LAYERS[target] == SUBSYSTEM_LAYER
        and SUBSYSTEM_GROUP.get(source, source) != SUBSYSTEM_GROUP.get(target, target)
        and not module.startswith(f"hivememory.{target}.contracts")
    ]
    assert violations == []


def _imports_root_config(path: Path) -> bool:
    """是否导入根配置模块（含 ``from hivememory.config import app`` 形式）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == ROOT_CONFIG_MODULE for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == ROOT_CONFIG_MODULE:
                return True
            if node.module == "hivememory.config" and any(
                alias.name == "app" for alias in node.names
            ):
                return True
    return False


def test_root_config_is_only_used_by_composition_root_and_entries():
    """根配置与加载函数只供 system / server 使用，其他包只能导入各自的配置段。"""
    violations = [
        path.relative_to(SRC_ROOT).as_posix()
        for path in _source_files()
        if _top_package(path) not in ROOT_CONFIG_CONSUMERS | {"config"}
        and _imports_root_config(path)
    ]
    assert violations == []


def test_root_package_only_imports_version():
    """根包初始化不得加载任何子包，否则任意导入都会连带加载整个系统。"""
    root_imports = _internal_imports(SRC_ROOT / "__init__.py")
    assert root_imports == ["hivememory._version"]


def test_cpu_input_compilation_does_not_depend_on_alice_or_its_configuration():
    """进程提示词编译不能导入具体 CPU 或其配置，任意执行者应共用同一产物。"""
    forbidden = ("hivememory.alice", "hivememory.agent_runtime", "hivememory.config.alice")
    paths = (SRC_ROOT / "workspace/process/allocation.py", SRC_ROOT / "prompts/compilation.py")
    violations = []
    for path in paths:
        modules = _internal_imports(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = _resolve_relative(path, node) if node.level else node.module
                modules.extend(f"{base}.{alias.name}" for alias in node.names)
        violations.extend(
            f"{path.relative_to(SRC_ROOT)}: {module}"
            for module in modules
            if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden)
        )
    assert violations == []


def test_importing_core_layer_loads_no_higher_layer_package():
    """导入 core 模型时，运行时加载的 hivememory 包都应位于 L0。"""
    script = (
        "import sys, hivememory.core.models, hivememory.core.protocol, hivememory.core.access\n"
        "print(' '.join(sorted({m.split('.')[1] for m in sys.modules "
        "if m.startswith('hivememory.')})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(SRC_ROOT.parent)},
    )
    loaded = set(result.stdout.split())
    assert {package for package in loaded if LAYERS[package] > 0} == set()


def test_neutral_producers_do_not_depend_on_mtp_exceptions():
    """中立服务可以编译 MTP 模型，但不得抛出或捕获协议出口的异常。"""
    exception_module = "hivememory.core.mtp.exceptions"
    exception_tree = ast.parse((SRC_ROOT / "core/mtp/exceptions.py").read_text(encoding="utf-8"))
    exception_names = {node.name for node in exception_tree.body if isinstance(node, ast.ClassDef)}
    producer_packages = {"infrastructure", "engines", "workspace", "patchouli", "gateway"}
    violations = []
    for path in _source_files():
        if _top_package(path) not in producer_packages:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bindings = {}
        references = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bindings[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
                    references.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_relative(path, node) if node.level else node.module
                for alias in node.names:
                    target = f"{base}.{alias.name}"
                    bindings[alias.asname or alias.name] = target
                    references.append(target)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name) and current.id in bindings:
                references.append(".".join([bindings[current.id], *reversed(parts)]))
        for target in references:
            if (
                target == exception_module
                or target.startswith(exception_module + ".")
                or target in {f"hivememory.core.mtp.{name}" for name in exception_names}
                or target == "hivememory.core.mtp.*"
            ):
                violations.append(f"{path.relative_to(SRC_ROOT)}: {target}")
    assert violations == []


LLM_PACKAGE = "hivememory.infrastructure.llm"
LLM_INTERFACE_MODULE = f"{LLM_PACKAGE}.base"
LLM_PACKAGE_DIR = SRC_ROOT / "infrastructure" / "llm"


def _absolute_import_targets(path: Path) -> list[str]:
    """返回文件导入的全部模块与 ``from x import y`` 形式的 ``x.y``，相对导入已解析。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(path, node) if node.level else node.module
            if base is None:
                continue
            targets.append(base)
            targets.extend(f"{base}.{alias.name}" for alias in node.names)
    return targets


def test_litellm_is_only_imported_inside_the_llm_package():
    """提供商 SDK 只在 LLM 服务的实现中出现，引擎与子系统看不到 SDK 类型。"""
    violations = [
        f"{path.relative_to(SRC_ROOT).as_posix()}: {target}"
        for path in _source_files()
        if LLM_PACKAGE_DIR not in path.parents
        for target in _absolute_import_targets(path)
        if target == "litellm" or target.startswith("litellm.")
    ]
    assert violations == []


def test_llm_interface_module_depends_on_no_sdk():
    """LLM 服务接口只依赖标准库与 pydantic，不导入 SDK 或同包的实现模块。"""
    path = LLM_PACKAGE_DIR / "base.py"
    violations = []
    for target in _absolute_import_targets(path):
        top = target.split(".")[0]
        if top == "hivememory":
            if target.startswith(LLM_PACKAGE + ".") and target != LLM_INTERFACE_MODULE:
                violations.append(target)
        elif top not in sys.stdlib_module_names and top != "pydantic":
            violations.append(target)
    assert violations == []


def _llm_implementation_imports(path: Path) -> list[str]:
    """逐条判断导入语句：只允许接口模块本身，包根与其他子模块都算实现。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(path, node) if node.level else node.module
            if base == LLM_PACKAGE:
                # ``from hivememory.infrastructure.llm import base`` 只取接口模块。
                imported.extend(f"{base}.{alias.name}" for alias in node.names)
            elif base == "hivememory.infrastructure":
                imported.extend(
                    f"{base}.{alias.name}" for alias in node.names if alias.name == "llm"
                )
            elif base is not None:
                imported.append(base)
    return [
        module
        for module in imported
        if (module == LLM_PACKAGE or module.startswith(LLM_PACKAGE + "."))
        and not (module == LLM_INTERFACE_MODULE or module.startswith(LLM_INTERFACE_MODULE + "."))
    ]


def test_only_composition_root_imports_the_llm_implementation():
    """除组合根外，从 LLM 包只能导入接口模块；实现模块与包根只供组合根装配。"""
    violations = [
        f"{path.relative_to(SRC_ROOT).as_posix()}: {module}"
        for path in _source_files()
        if _top_package(path) != "system" and LLM_PACKAGE_DIR not in path.parents
        for module in _llm_implementation_imports(path)
    ]
    assert violations == []
