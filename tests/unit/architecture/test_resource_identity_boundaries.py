"""资源 owner 的身份边界约束：操作 scope 不进入 Patchouli 内部和长期载体。"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.unit.architecture.test_package_layers import SRC_ROOT

IDENTITY_SCOPE_SYMBOLS = {"IdentityScope", "require_identity_scope"}
PATCHOULI_ENGINES = {"generation", "perception", "retrieval", "lifecycle", "artifacts"}


def _source_files(directory: Path) -> list[Path]:
    files = sorted(directory.rglob("*.py"))
    assert files, f"未扫描到源文件，请检查路径: {directory}"
    return files


def _symbol_references(node: ast.AST, symbols: set[str]) -> list[int]:
    """扫描实际符号引用，覆盖导入别名、属性访问和字符串类型注解。"""
    references: list[int] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in symbols:
            references.append(child.lineno)
        elif isinstance(child, ast.Attribute) and child.attr in symbols:
            references.append(child.lineno)
        elif isinstance(child, ast.alias) and child.name.rsplit(".", 1)[-1] in symbols:
            references.append(child.lineno)
        if isinstance(child, (ast.AnnAssign, ast.arg)):
            annotation = child.annotation
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = child.returns
        else:
            annotation = None
        if annotation is not None:
            for part in ast.walk(annotation):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    quoted = ast.parse(part.value, mode="eval")
                    if _symbol_references(quoted, symbols):
                        references.append(child.lineno)
    return references


def _is_public_handler(path: Path) -> bool:
    """只有阶段门面与应用 service 是公开处理者，不给整个 application 包豁免。"""
    relative = path.relative_to(SRC_ROOT)
    return relative.as_posix() == "patchouli/service.py" or (
        relative.parts[:2] == ("patchouli", "application") and relative.name.endswith("service.py")
    )


def test_patchouli_only_public_handlers_reference_identity_scope():
    """Patchouli 内部只使用归属与发起者，scope 仅在公开处理者的入口拆分。"""
    violations = []
    for path in _source_files(SRC_ROOT / "patchouli"):
        if _is_public_handler(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line in sorted(set(_symbol_references(tree, IDENTITY_SCOPE_SYMBOLS))):
            violations.append(f"{path.relative_to(SRC_ROOT).as_posix()}:{line}")
    assert violations == []


def test_patchouli_engines_do_not_reference_identity_scope():
    """Patchouli 驱动的五个引擎不得引用操作 scope 或其入口校验函数。"""
    violations = []
    for engine in sorted(PATCHOULI_ENGINES):
        for path in _source_files(SRC_ROOT / "engines" / engine):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for line in sorted(set(_symbol_references(tree, IDENTITY_SCOPE_SYMBOLS))):
                violations.append(f"{path.relative_to(SRC_ROOT).as_posix()}:{line}")
    assert violations == []


def test_public_records_and_materialize_tasks_do_not_store_identity_scope():
    """公开记录与物化任务不能保存超出一次操作寿命的 scope 字段。"""
    paths = [
        *_source_files(SRC_ROOT / "patchouli" / "contracts"),
        SRC_ROOT / "core" / "models" / "pending.py",
    ]
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        if path.name == "pending.py":
            classes = [node for node in classes if node.name == "PendingAtomMaterializeTask"]
            assert [node.name for node in classes] == ["PendingAtomMaterializeTask"]
        for model in classes:
            for field in model.body:
                if not isinstance(field, ast.AnnAssign) or not isinstance(field.target, ast.Name):
                    continue
                if field.target.id == "identity_scope" or _symbol_references(
                    field, {"IdentityScope"}
                ):
                    violations.append(
                        f"{path.relative_to(SRC_ROOT).as_posix()}:{field.lineno} "
                        f"{model.name}.{field.target.id}"
                    )
    assert violations == []
