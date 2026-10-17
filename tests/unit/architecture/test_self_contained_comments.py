"""代码目录不反向引用文档的守卫（tokenize/AST 静态扫描）。

文档可以指向代码，代码不反向指向文档：工作文档归档、删除，或描述现状的文档改写后，
指向它们的注释会失去意义或与实现不一致。注释、docstring 与字段 ``description``
必须直接写出行为、不变量与理由。

扫描范围：``src``、``tests``、``scripts`` 下的 Python 注释、裸字符串表达式
（含 docstring）与 ``description=`` 字符串参数，以及 ``frontend/src`` 下
TypeScript 的注释。变量名、用户可见的提示文案与其他字符串字面量不在范围内。
同样的目录下也不放 README：模块用途写在 docstring 中，设计说明只在文档里维护。
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOTS = ("src", "tests", "scripts")
FRONTEND_ROOT = REPO_ROOT / "frontend" / "src"
CODE_ROOTS = (*(REPO_ROOT / root for root in PYTHON_ROOTS), FRONTEND_ROOT)

# 规则名 → 模式。只收录可以机读的引用形式；其余写法靠评审。
FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "文档路径或文件名": re.compile(r"(?<![\w.])docs/|\b[\w-]+\.md\b"),
    "章节号": re.compile(r"§\s*\d|第\s*\d+(?:\.\d+)*\s*节|\b\d+(?:\.\d+)+\s*节|\bSection\s+\d"),
    "决定或问题编号": re.compile(r"\b(?:Q|D|S|W|T|P|E|I|AC)-\d+[a-z]?\b"),
    "计划代号": re.compile(
        r"\bA[1-6](?:-[A-Za-z0-9]+)?\b|\bW[01](?:-[A-F])?\b|\bM-3\b|\bMVL-\d|\bM\d\.\d\b"
    ),
    "计划阶段或批次": re.compile(
        r"返工|宪章|第[一二三四五六七八九十]批|阶段[一二三四五六七八九十]|实现阶段|实施阶段"
        r"|\bPhase\s*\d"
    ),
    "计划与工作文档": re.compile(
        r"(?<!检索)(?<!执行)(?<!查询)计划(?!生成)|\bIdea\b|\bTodo\b|\bADR\b|\bPlan\b"
        r"|设计文档|技术债文档|事实文档"
    ),
    "依赖计划进度的措辞": re.compile(r"暂时保留|暂时先于|下一批|后续批次|一批之前"),
    "版本历史": re.compile(r"\bv0\.\d+\.\d+\b"),
    "文档中的编号条目": re.compile(r"(?:不变量|证据|验收项|条目)\s*#?\d"),
}

_TS_COMMENT = re.compile(r"(?<![:\w])//[^\n]*|/\*[\s\S]*?\*/")


def _python_files() -> Iterator[Path]:
    for root in PYTHON_ROOTS:
        yield from sorted((REPO_ROOT / root).rglob("*.py"))


def _python_texts(path: Path) -> Iterator[tuple[int, str]]:
    """产出注释、裸字符串表达式与 ``description=`` 参数的 (行号, 文本)。"""
    source = path.read_text(encoding="utf-8")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.string
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            yield node.lineno, node.value.value
        elif isinstance(node, ast.keyword) and node.arg == "description":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                yield node.value.lineno, node.value.value


def _frontend_texts(path: Path) -> Iterator[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    for match in _TS_COMMENT.finditer(source):
        yield source.count("\n", 0, match.start()) + 1, match.group()


def _violations(path: Path, texts: Iterator[tuple[int, str]]) -> Iterator[str]:
    relative = path.relative_to(REPO_ROOT)
    for start_line, text in texts:
        for offset, line in enumerate(text.splitlines()):
            for rule, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(line):
                    yield f"{relative}:{start_line + offset} [{rule}] {line.strip()}"


def test_comments_and_docstrings_do_not_reference_documents_or_plans() -> None:
    violations: list[str] = []
    for path in _python_files():
        violations.extend(_violations(path, _python_texts(path)))
    for path in sorted(FRONTEND_ROOT.rglob("*.ts*")):
        violations.extend(_violations(path, _frontend_texts(path)))
    assert (
        not violations
    ), "注释与说明文本应直接写出行为、不变量与理由，不引用文档、计划或决定编号：\n" + "\n".join(
        violations
    )


def test_code_directories_contain_no_readme() -> None:
    readmes = [
        str(path.relative_to(REPO_ROOT))
        for root in CODE_ROOTS
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name.lower().startswith("readme")
    ]
    assert not readmes, "代码目录不放 README，模块用途写在 docstring 中：\n" + "\n".join(readmes)
