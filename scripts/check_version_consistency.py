"""校验 HiveMemory 的构建、运行时和前端版本声明是否一致。"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "src" / "hivememory" / "_version.py"
VERSION_ATTR = "hivememory._version.__version__"


def _read_canonical_version() -> str:
    tree = ast.parse(VERSION_FILE.read_text(encoding="utf-8"), filename=str(VERSION_FILE))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise ValueError(f"{VERSION_FILE} must assign __version__ to a string literal")


def _check_python_metadata(version: str, errors: list[str]) -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    if "version" in project:
        errors.append("pyproject.toml must not duplicate a static project.version")
    if "version" not in project.get("dynamic", []):
        errors.append("pyproject.toml must declare project.dynamic = ['version']")

    configured_attr = (
        pyproject.get("tool", {})
        .get("setuptools", {})
        .get("dynamic", {})
        .get("version", {})
        .get("attr")
    )
    if configured_attr != VERSION_ATTR:
        errors.append(
            f"setuptools dynamic version must read {VERSION_ATTR!r}, got {configured_attr!r}"
        )

    if not re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", version):
        errors.append(
            "canonical version must use stable MAJOR.MINOR.PATCH form; "
            f"pre-release publishing is not configured, got {version!r}"
        )


def _check_frontend_metadata(version: str, errors: list[str]) -> None:
    package = json.loads((PROJECT_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads(
        (PROJECT_ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8")
    )

    declarations = {
        "frontend/package.json": package.get("version"),
        "frontend/package-lock.json": package_lock.get("version"),
        "frontend/package-lock.json packages['']": package_lock.get("packages", {})
        .get("", {})
        .get("version"),
    }
    for source, declared_version in declarations.items():
        if declared_version != version:
            errors.append(f"{source} declares {declared_version!r}; expected {version!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        help="可选的发布标签；提供后必须精确等于 v<规范版本号>",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    try:
        version = _read_canonical_version()
        _check_python_metadata(version, errors)
        _check_frontend_metadata(version, errors)
    except (KeyError, OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        errors.append(str(exc))
        version = "unknown"

    if args.tag:
        tag = args.tag.removeprefix("refs/tags/")
        expected_tag = f"v{version}"
        if tag != expected_tag:
            errors.append(f"release tag {tag!r} does not match {expected_tag!r}")

    if errors:
        for error in errors:
            print(f"version check failed: {error}", file=sys.stderr)
        return 1

    print(f"HiveMemory version declarations are consistent: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
