"""领域模型裸 user_id 字段守卫（v0.6.2 B2 验收）。

扫描 ``core/models`` 与 ``engines`` 的类级字段声明，断言不存在白名单外的
裸 ``user_id`` 字段：actor 语义一律收敛到 ``ActorIdentity``，owner 语义使用
``WorkspaceIdentity.owner_user_id``。

白名单约定（v0.6.2 计划 §2/§7）：
- ``core/models/identity.py``：``ActorIdentity.user_id`` 是权威字段本身；
- ``memory.py`` 的存储平铺投影与 passive 命名键不在扫描范围或以注释说明，
  保留至「Workspace 历史数据转换」独立迁移切片处理。
"""

import ast
from pathlib import Path

# 相对仓库根目录的扫描范围与白名单（文件名 → 允许原因）
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCAN_DIRS = (
    _REPO_ROOT / "src" / "hivememory" / "core" / "models",
    _REPO_ROOT / "src" / "hivememory" / "engines",
)
_WHITELIST = {
    "identity.py": "ActorIdentity.user_id 是权威 actor 字段本身",
}


def _class_level_user_id_fields(path: Path) -> list[str]:
    """返回单个文件中类级 ``user_id`` 注解字段的位置描述列表。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "user_id"
            ):
                hits.append(f"{path.name}:{stmt.lineno} ({node.name})")
    return hits


# DEFAULT_USER_ID / normalize_user_id 的合法使用边界（v0.6.2 计划 §4）：
# 1. core/constants.py —— 定义处；
# 2. core/models/identity.py —— ActorIdentity 字段默认值；
# 3. server/deps.py —— HTTP 顶层身份解析的唯一缺省回退点。
_DEFAULT_USER_ID_ALLOWED = {
    "src/hivememory/core/constants.py",
    "src/hivememory/core/models/identity.py",
    "src/hivememory/server/deps.py",
}


def test_default_user_id_usage_stays_within_sanctioned_boundary():
    """DEFAULT_USER_ID 回退不得扩散到应用服务或引擎层。"""
    src_root = _REPO_ROOT / "src" / "hivememory"
    violations = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in sorted(src_root.rglob("*.py"))
        if path.relative_to(_REPO_ROOT).as_posix() not in _DEFAULT_USER_ID_ALLOWED
        and "DEFAULT_USER_ID" in path.read_text(encoding="utf-8")
    ]

    assert violations == [], (
        "DEFAULT_USER_ID 出现在合法边界之外（应用服务不得解析默认身份），"
        "涉及文件：\n" + "\n".join(violations)
    )


def test_no_bare_user_id_fields_outside_whitelist():
    violations: list[str] = []

    for scan_dir in _SCAN_DIRS:
        for path in sorted(scan_dir.rglob("*.py")):
            if path.name in _WHITELIST:
                continue
            violations.extend(_class_level_user_id_fields(path))

    assert violations == [], (
        "core/models 与 engines 中出现白名单外的裸 user_id 领域字段，"
        "应收敛为 ActorIdentity / WorkspaceIdentity：\n" + "\n".join(violations)
    )
