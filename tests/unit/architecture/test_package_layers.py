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

# 已知的向上依赖（engines 直接使用 Patchouli 存储层、Gateway 命令模型与
# AgentRuntime alias 结果），属于分层重构之前就存在的耦合，另行处理。
# 断言为精确相等：新增违规或修复后未同步本清单都会失败。
KNOWN_UPWARD_IMPORTS = {
    ("engines/artifacts/document.py", "hivememory.patchouli.memory_library"),
    ("engines/artifacts/engine.py", "hivememory.patchouli.memory_library.stores"),
    ("engines/artifacts/interaction.py", "hivememory.patchouli.memory_library"),
    ("engines/artifacts/memory.py", "hivememory.patchouli.memory_library"),
    ("engines/gateway/interceptors.py", "hivememory.gateway.commands"),
    ("engines/gateway/models.py", "hivememory.gateway.commands.models"),
    ("engines/generation/engine.py", "hivememory.patchouli.memory_library.stores"),
    ("engines/lifecycle/engine.py", "hivememory.patchouli.memory_library.stores"),
    ("engines/lifecycle/garbage_collector.py", "hivememory.patchouli.memory_library.library"),
    ("engines/lifecycle/reinforcement.py", "hivememory.patchouli.memory_library.stores"),
    ("engines/memory_compiler/builders/resolve_result.py", "hivememory.agent_runtime.aliases"),
    ("engines/memory_compiler/compiler.py", "hivememory.agent_runtime.aliases"),
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
