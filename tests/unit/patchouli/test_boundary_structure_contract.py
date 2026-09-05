"""边界重构的结构契约测试（Plan §6.4）。

静态检查生产代码不再引用已被删除的边界组件：
- ``TopicBufferService``（短期存储唯一入口的旧混合体）；
- ``SemanticBuffer``（adapter 内部的可变缓冲实体）。

占用权由 ``TopicWorkingSet`` 的 lease 表达，内容归 ``ShortTermMemoryStore``，
编排归 ``PerceptionFamiliar``。
"""

import ast
from pathlib import Path

import hivememory


def _iter_production_sources() -> list[Path]:
    """遍历 src/hivememory 下全部生产 Python 文件（不含测试）。"""
    root = Path(hivememory.__file__).parent
    return sorted(root.rglob("*.py"))


def _iter_import_targets(tree: ast.AST):
    """产出 (import 语句, 目标模块名) 对，覆盖 ImportFrom 与 Import。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                yield node, node.module
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name


def test_no_topic_buffer_service_imports_in_production():
    for py_file in _iter_production_sources():
        text = py_file.read_text(encoding="utf-8-sig")
        assert "TopicBufferService" not in text, (
            f"{py_file} 仍引用 TopicBufferService；编排归 PerceptionFamiliar"
        )
        tree = ast.parse(text)
        for node, target in _iter_import_targets(tree):
            assert "topic_buffer" not in target, (
                f"{py_file} imports {target}; TopicBufferService 已删除，"
                "编排归 PerceptionFamiliar"
            )


def test_no_semantic_buffer_references_in_production():
    for py_file in _iter_production_sources():
        text = py_file.read_text(encoding="utf-8-sig")
        assert "SemanticBuffer" not in text, (
            f"{py_file} 仍引用 SemanticBuffer；adapter 已直接存储 frozen TopicData"
        )
