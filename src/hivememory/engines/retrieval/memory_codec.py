"""Memory 持久化 payload 的受控读取与规范化。

支持两种输入（A2-P §7.1 兼容表）：
- schema ``"2.1"``（字符串）：新 canonical 格式，直接校验为领域对象；
- schema ``2``（整数）：旧格式只读兼容解码——平铺来源/动态字段聚合为
  ``meta.provenance``/``meta.lifecycle``、``agent_config`` 移入新位置、
  丢弃 ``session_id``/``history_summary``。兼容窗口内旧记录只读，不由新
  mutation 路径回写（拒绝点在存储侧 ``get_for_mutation``/``patch_payload``）。

缺少 ``schema_version``、未知版本、部分 Workspace 投影和冲突 owner 均
fail closed。旧记录的 naive 时间按"写入时服务器本地时区"解释并规范化为
UTC（一次性告警提示存在待迁移数据）；该解释只作用于只读兼容路径，不做
任何回写。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any

from hivememory.core.models import (
    MemoryAtom,
    MemoryLifecycleState,
    MemoryProvenance,
    WorkspaceIdentity,
)
from hivememory.utils.time import require_utc

logger = logging.getLogger(__name__)

_WORKSPACE_PROJECTION_FIELDS = (
    "owner_user_id",
    "workspace_key",
    "workspace_id",
)

_LEGACY_NAIVE_WARNING_EMITTED = False


class MemoryDecodeError(ValueError):
    """持久化 Memory 无法安全归一化为 Memory 领域对象。"""


class MemorySchemaReadOnlyError(MemoryDecodeError):
    """旧 schema 记录处于只读兼容窗口，拒绝由 mutation 路径读取后回写。"""


def decode_memory_payload(
    payload: Mapping[str, Any],
    *,
    allow_legacy: bool = True,
) -> MemoryAtom:
    """解码持久化 payload，返回唯一 canonical Memory 领域对象。

    依据 ``schema_version`` 分派：``"2.1"`` 走新格式校验，整数 ``2`` 走只读
    兼容解码；其余值 fail closed。``allow_legacy=False`` 用于 mutation 入口
    （如 ``get_for_mutation``）：旧 schema 记录在兼容窗口内只读，读取后回写
    会用兼容推断固化历史时间，因此直接以
    :class:`MemorySchemaReadOnlyError` 拒绝。
    """
    raw = deepcopy(dict(payload))
    schema_version = raw.get("schema_version")
    if schema_version == "2.1":
        return _decode_current(raw)
    if schema_version == 2:
        if not allow_legacy:
            raise MemorySchemaReadOnlyError(
                "旧 schema Memory 记录在兼容窗口内只读；请先运行 A2-P 临时迁移脚本转换为 2.1"
            )
        return _decode_legacy_v2(raw)
    raise MemoryDecodeError(f"不支持的 Memory schema_version: {schema_version!r}")


def _decode_current(raw: dict[str, Any]) -> MemoryAtom:
    """校验并解码 schema "2.1" 的新 canonical 格式。"""
    meta = _require_mapping(raw.get("meta"), "meta")
    projected = _extract_complete_projection(meta)
    domain_meta = dict(meta)
    for field in _WORKSPACE_PROJECTION_FIELDS:
        domain_meta.pop(field, None)
    raw["meta"] = domain_meta

    try:
        atom = MemoryAtom.model_validate(raw)
    except Exception as exc:
        raise MemoryDecodeError(f"无效的 Memory schema 2.1: {exc}") from exc

    if projected is not None and projected != atom.workspace_identity:
        raise MemoryDecodeError("Memory 嵌套 ownership 与存储索引投影不一致")
    return atom


def _decode_legacy_v2(raw: dict[str, Any]) -> MemoryAtom:
    """把旧整数 schema 2 只读转换为 "2.1" 领域对象（兼容窗口内不回写）。

    转换规则见 A2-P §7.1 与时间边界计划 §6：平铺 provenance 字段聚合为
    ``meta.provenance``，平铺动态字段聚合为 ``meta.lifecycle`` 并以旧
    ``updated_at`` 初始化 ``decay_anchor_at``（保持既有衰减行为，并在报告
    范围标注为兼容推断）；``payload.artifacts.agent_config`` 移入
    ``payload.agent_config``；``session_id``/``history_summary`` 丢弃。
    """
    meta = _require_mapping(raw.get("meta"), "meta")
    payload = _require_mapping(raw.get("payload"), "payload")
    projected = _extract_complete_projection(meta)

    domain_meta = dict(meta)
    for field in _WORKSPACE_PROJECTION_FIELDS:
        domain_meta.pop(field, None)

    domain_meta.pop("session_id", None)

    # 平铺 provenance → 结构化聚合；v2 中 source_agent_id 为必填，缺失即拒绝。
    if not domain_meta.get("source_agent_id"):
        raise MemoryDecodeError("旧 schema 记录缺少 source_agent_id，无法迁移来源 provenance")
    provenance = MemoryProvenance(
        source_agent_id=domain_meta.pop("source_agent_id"),
        source_team_id=domain_meta.pop("source_team_id", None),
        contributing_agent_ids=tuple(domain_meta.pop("contributing_agent_ids", ()) or ()),
    )

    updated_at = _interpret_legacy_time(domain_meta.get("updated_at"), "updated_at")
    last_accessed_raw = domain_meta.pop("last_accessed_at", None)
    lifecycle = MemoryLifecycleState(
        access_count=int(domain_meta.pop("access_count", 0) or 0),
        last_accessed_at=(
            _interpret_legacy_time(last_accessed_raw, "last_accessed_at")
            if last_accessed_raw is not None
            else None
        ),
        event_vitality_boost=float(domain_meta.pop("event_vitality_boost", 0.0) or 0.0),
        vitality_score=float(domain_meta.pop("vitality_score", 100.0)),
        confidence_score=float(domain_meta.pop("confidence_score", 0.6)),
        verification_status=domain_meta.pop("verification_status", "UNVERIFIED"),
        # 兼容推断：旧衰减基准即 updated_at（保持既有衰减行为；正式迁移由
        # 临时脚本按显式来源时区执行并出具报告）。
        decay_anchor_at=updated_at,
    )
    created_at = _interpret_legacy_time(domain_meta.pop("created_at", None), "created_at")

    domain_payload = dict(payload)
    domain_payload.pop("history_summary", None)
    artifacts = domain_payload.get("artifacts")
    legacy_agent_config: Any = None
    if isinstance(artifacts, dict):
        artifacts = dict(artifacts)
        legacy_agent_config = artifacts.pop("agent_config", None)
        domain_payload["artifacts"] = artifacts
    new_agent_config = domain_payload.get("agent_config")
    if new_agent_config is None and legacy_agent_config is not None:
        domain_payload["agent_config"] = legacy_agent_config
    elif (
        new_agent_config is not None
        and legacy_agent_config is not None
        and new_agent_config != legacy_agent_config
    ):
        raise MemoryDecodeError("agent_config 新旧位置同时存在且值不一致，拒绝猜测")

    converted = {
        "schema_version": "2.1",
        "id": raw.get("id"),
        "meta": {
            **domain_meta,
            "created_at": created_at,
            "updated_at": updated_at,
            "provenance": provenance.model_dump(),
            "lifecycle": lifecycle.model_dump(),
        },
        "index": raw.get("index"),
        "payload": domain_payload,
        "relations": raw.get("relations"),
    }

    try:
        atom = MemoryAtom.model_validate(converted)
    except Exception as exc:
        raise MemoryDecodeError(f"无效的旧 schema Memory 记录: {exc}") from exc

    if projected is not None and projected != atom.workspace_identity:
        raise MemoryDecodeError("Memory 嵌套 ownership 与存储索引投影不一致")
    return atom


def _interpret_legacy_time(value: Any, field: str) -> datetime:
    """解释旧记录的时间值并规范化为 UTC。

    aware 值直接转换；naive 值（旧 ``datetime.now()`` 的持久化结果）按写入
    时服务器本地时区解释——兼容读取路径中唯一可用的解释，且不发生回写。
    首次遇到时输出一次告警，提示存量数据仍待临时迁移脚本处理。
    """
    global _LEGACY_NAIVE_WARNING_EMITTED
    if value is None:
        raise MemoryDecodeError(f"旧 schema 记录缺少 {field}，拒绝用当前时间补值")
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise MemoryDecodeError(f"旧 schema 记录的 {field} 无法解析: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if not _LEGACY_NAIVE_WARNING_EMITTED:
            logger.warning(
                "检测到旧 schema Memory 记录携带 naive 时间（按写入时服务器本地时区解释为 UTC）。"
                "请尽快运行 A2-P 临时 Memory 迁移脚本完成存量转换。"
            )
            _LEGACY_NAIVE_WARNING_EMITTED = True
        return parsed.astimezone()
    return require_utc(parsed)


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


__all__ = ["MemoryDecodeError", "MemorySchemaReadOnlyError", "decode_memory_payload"]
