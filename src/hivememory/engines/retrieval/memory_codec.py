"""Memory schema v1/v2 的受控读取与规范化。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from hivememory.core.models import (
    MAIN_WORKSPACE_ID,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryVisibility,
    MetaData,
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
    """按版本解码持久化 payload，返回唯一 canonical v2 领域对象。

    缺少 ``schema_version`` 的记录仅在此兼容层按 v1 解释；未知版本、
    部分 Workspace 投影和冲突 owner 均 fail closed。
    """
    raw = deepcopy(dict(payload))
    schema_version = raw.get("schema_version")
    if schema_version is None:
        return _decode_v1(raw)
    if schema_version == 2:
        return _decode_v2(raw)
    raise MemoryDecodeError(f"不支持的 Memory schema_version: {schema_version!r}")


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


def _decode_v1(raw: dict[str, Any]) -> MemoryAtom:
    meta = _require_mapping(raw.get("meta"), "meta")
    workspace = _legacy_workspace_identity(meta)
    source_agent_id = _require_non_empty(meta.get("source_agent_id"), "source_agent_id")
    source_team_id = _optional_non_empty(meta.get("team_id"), "team_id")
    policy = _adapt_v1_policy(
        visibility=meta.get("visibility", "PUBLIC"),
        source_agent_id=source_agent_id,
        source_team_id=source_team_id,
    )

    domain_meta = dict(meta)
    for field in (*_WORKSPACE_PROJECTION_FIELDS, "user_id", "team_id", "visibility"):
        domain_meta.pop(field, None)
    domain_meta.update(
        {
            "workspace_identity": workspace,
            "source_agent_id": source_agent_id,
            "source_team_id": source_team_id,
            "access_policy": policy,
        }
    )

    try:
        normalized_meta = MetaData.model_validate(domain_meta)
        return MemoryAtom(
            id=raw["id"],
            meta=normalized_meta,
            index=raw["index"],
            payload=raw["payload"],
            relations=raw.get("relations", {}),
        )
    except MemoryDecodeError:
        raise
    except Exception as exc:
        raise MemoryDecodeError(f"无效的 legacy Memory: {exc}") from exc


def _legacy_workspace_identity(meta: Mapping[str, Any]) -> WorkspaceIdentity:
    projected = _extract_complete_projection(meta)
    legacy_user_id = _optional_non_empty(meta.get("user_id"), "user_id")
    if projected is not None:
        if legacy_user_id is not None and legacy_user_id != projected.owner_user_id:
            raise MemoryDecodeError("legacy user_id 与 Workspace owner 冲突")
        return projected
    if legacy_user_id is None:
        raise MemoryDecodeError("legacy Memory 缺少 user_id，无法确定资源归属")
    return WorkspaceIdentity(
        owner_user_id=legacy_user_id,
        workspace_key=MAIN_WORKSPACE_ID,
        workspace_id=MAIN_WORKSPACE_ID,
    )


def _extract_complete_projection(
    meta: Mapping[str, Any],
) -> WorkspaceIdentity | None:
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


def _adapt_v1_policy(
    *,
    visibility: Any,
    source_agent_id: str,
    source_team_id: str | None,
) -> MemoryAccessPolicy:
    value = visibility.value if hasattr(visibility, "value") else str(visibility)
    if value == "PUBLIC":
        return MemoryAccessPolicy.public()
    if value == "PRIVATE":
        return MemoryAccessPolicy(
            visibility=MemoryVisibility.PRIVATE,
            target_agent_id=source_agent_id,
        )
    if value == "WORKSPACE":
        if source_team_id is None:
            raise MemoryDecodeError("legacy WORKSPACE Memory 缺少 team_id")
        return MemoryAccessPolicy(
            visibility=MemoryVisibility.TEAM,
            target_team_id=source_team_id,
        )
    raise MemoryDecodeError(f"未知的 legacy Memory visibility: {value!r}")


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MemoryDecodeError(f"Memory {field_name} 必须是对象")
    return value


def _require_non_empty(value: Any, field_name: str) -> str:
    normalized = _optional_non_empty(value, field_name)
    if normalized is None:
        raise MemoryDecodeError(f"Memory {field_name} 不能为空")
    return normalized


def _optional_non_empty(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MemoryDecodeError(f"Memory {field_name} 必须是非空字符串")
    return value.strip()


__all__ = ["MemoryDecodeError", "decode_memory_payload"]
