"""engines/perception 包的结构契约测试（Plan §6.4）。

静态检查感知算法层不导入 ``hivememory.patchouli.*``：engines 是可被其他
runtime 复用的纯算法层，依赖方向必须保持 patchouli -> engines 单向。
"""

import ast
from pathlib import Path

import hivememory.engines.perception


def _iter_perception_sources() -> list[Path]:
    package_dir = Path(hivememory.engines.perception.__file__).parent
    return sorted(package_dir.glob("*.py"))


def test_engines_perception_does_not_import_patchouli():
    for py_file in _iter_perception_sources():
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    "hivememory.patchouli"
                ), f"{py_file.name} imports {node.module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "hivememory.patchouli"
                    ), f"{py_file.name} imports {alias.name}"
