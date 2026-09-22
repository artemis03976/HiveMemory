"""Memory schema v2 的受控读取与规范化。

legacy v1 兼容解释分支已随存量数据全量迁移完成而删除（迁移工具与报告见
``scripts/migrate_v1_memory_and_artifacts.py``）：缺少 ``schema_version`` 的
记录不再按 v1 解释，直接 fail closed。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from hivememory.core.models import (
    MemoryAtom,
    WorkspaceIdentity,
)

_WORKSPACE_PROJECTION_FIELDS = (
    "owner_user_id",
    "workspace_key",
    "workspace_id",
)


class MemoryDecodeError(ValueError):
    """持久化 Memory 无法安全归一化为 schema v2。"""


def decode_memory_payload(payload: Mapping[str, Any]) -> MemoryAtom:
    """解码持久化 payload，返回唯一 canonical v2 领域对象。

    缺少 ``schema_version``、未知版本、部分 Workspace 投影和冲突 owner 均
    fail closed。
    """
    raw = deepcopy(dict(payload))
    schema_version = raw.get("schema_version")
    if schema_version != 2:
        raise MemoryDecodeError(f"不支持的 Memory schema_version: {schema_version!r}")
    return _decode_v2(raw)


def _decode_v2(raw: dict[str, Any]) -> MemoryAtom:
    meta = _require_mapping(raw.get("meta"), "meta")
    projected = _extract_complete_projection(meta)
    domain_meta = dict(meta)
    for field in _WORKSPACE_PROJECTION_FIELDS:
        domain_meta.pop(field, None)
    raw["meta"] = domain_meta

    try:
        atom = MemoryAtom.model_validate(raw)
    except Exception as exc:
        raise MemoryDecodeError(f"无效的 Memory schema v2: {exc}") from exc

    if projected is not None and projected != atom.workspace_identity:
        raise MemoryDecodeError("Memory v2 嵌套 ownership 与存储索引投影不一致")
    return atom


def _extract_complete_projection(
    meta: Mapping[str, Any],
) -> WorkspaceIdentity | None:
    """提取完整 Workspace 索引投影；无投影字段返回 None，部分投影拒绝猜测补齐。"""
    present = [field for field in _WORKSPACE_PROJECTION_FIELDS if meta.get(field) is not None]
    if not present:
        return None
    if len(present) != len(_WORKSPACE_PROJECTION_FIELDS):
        raise MemoryDecodeError("Memory 包含部分 Workspace 投影，拒绝猜测补齐")
    try:
        return WorkspaceIdentity(
            owner_user_id=meta["owner_user_id"],
            workspace_key=meta["workspace_key"],
            workspace_id=meta["workspace_id"],
        )
    except Exception as exc:
        raise MemoryDecodeError(f"无效的 Workspace 索引投影: {exc}") from exc


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MemoryDecodeError(f"Memory {field_name} 必须是对象")
    return value


__all__ = ["MemoryDecodeError", "decode_memory_payload"]
