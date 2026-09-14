from __future__ import annotations

from uuid import UUID


def normalize_uuid(value: UUID | str) -> UUID:
    """从现有 UUID 或兼容 UUID 的字符串返回 UUID 对象。"""
    return value if isinstance(value, UUID) else UUID(str(value))
