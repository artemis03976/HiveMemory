"""AgentProfileStore：Agent Profile 的文件存储。

一个 Profile 一个 Markdown 文件，位于 ``<root_dir>/<agent_id>.md``：文件名是 agent_id，
frontmatter 是 title、summary、model、temperature、top_p 与 enabled，正文是人设。它属于
MemoryLibrary，但不是三层记忆存储之一：Profile 不是记忆原子，不参与冷热迁移、检索与
版本记录，手动编辑也不保留历史。

Profile 不按 Workspace 分区：agent 的定义是全局的，它作为工作者进入 Workspace，能否在
某个 Workspace 工作、在那里能做什么，由该 Workspace 的访问登记决定。

读取与列出每次都直接读目录，不设缓存：直接改动文件与经管理接口保存等同，下一次读取就
生效。保存与读取共用同一套校验（:func:`profile_from_fields`）：读取时不合格的文件在列出
时跳过并记一条带原因码的日志，按 agent_id 读取时显式失败；保存先校验，再写临时文件、
替换原文件，替换失败时原文件不变。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from hivememory.config.files import write_atomically
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import InvalidAgentProfileError
from hivememory.core.models import AgentProfile, StoredAgentProfile

logger = logging.getLogger(__name__)

# agent_id 同时是文件名：只用小写字母、数字、下划线与连字符，Windows 上大小写不敏感的
# 文件名因此不会让两个 agent_id 指向同一个文件；长度上限沿用记忆 alias。
_AGENT_ID_PATTERN = re.compile(r"[a-z0-9_-]{1,60}")
# 内置 Profile 与保留的 system 先于文件解析或不能被调用，同名文件永远不会被用到。
_RESERVED_AGENT_IDS = frozenset({"default", "omni_doll", SYSTEM_AGENT_ID})

FRONTMATTER_FIELDS = frozenset({"title", "summary", "model", "temperature", "top_p", "enabled"})
"""frontmatter 只允许的字段；agent_id 取自文件名，人设是正文，都不在 frontmatter 中。"""

_SUFFIX = ".md"
_FRONTMATTER = re.compile(r"\A---\n(.*?)^---[ \t]*(?:\n|\Z)", re.DOTALL | re.MULTILINE)


def profile_from_fields(agent_id: str, fields: Mapping[str, Any], *, persona: str) -> AgentProfile:
    """按唯一的一套规则校验并构造 Profile；不合格时以 ``InvalidAgentProfileError`` 拒绝。

    取值按严格模式校验，不做类型转换：手写的 ``temperature: "0.3"`` 与经接口提交的
    同样被拒绝。
    """
    if not _AGENT_ID_PATTERN.fullmatch(agent_id) or agent_id in _RESERVED_AGENT_IDS:
        raise InvalidAgentProfileError(details={"reason": "invalid_agent_id", "agent_id": agent_id})
    unknown = sorted(set(fields) - FRONTMATTER_FIELDS)
    if unknown:
        raise InvalidAgentProfileError(
            details={"reason": "unknown_field", "agent_id": agent_id, "fields": unknown}
        )
    try:
        return AgentProfile.model_validate(
            {**fields, "agent_id": agent_id, "persona": persona}, strict=True
        )
    except ValidationError as exc:
        field = ".".join(str(part) for part in exc.errors()[0]["loc"])
        raise InvalidAgentProfileError(
            details={"reason": "invalid_value", "agent_id": agent_id, "field": field}
        ) from exc


def _parse(agent_id: str, text: str) -> AgentProfile:
    match = _FRONTMATTER.match(text)
    if match is None:
        raise InvalidAgentProfileError(
            details={"reason": "missing_frontmatter", "agent_id": agent_id}
        )
    try:
        fields = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise InvalidAgentProfileError(
            details={"reason": "invalid_frontmatter", "agent_id": agent_id}
        ) from exc
    if fields is None:
        fields = {}
    if not isinstance(fields, dict):
        raise InvalidAgentProfileError(
            details={"reason": "invalid_frontmatter", "agent_id": agent_id}
        )
    return profile_from_fields(agent_id, fields, persona=text[match.end() :])


def _render(profile: AgentProfile) -> str:
    """固定字段顺序；为空的采样参数不写，读回时同样为空。"""
    fields: dict[str, Any] = {
        "title": profile.title,
        "summary": profile.summary,
        "model": profile.model,
    }
    if profile.temperature is not None:
        fields["temperature"] = profile.temperature
    if profile.top_p is not None:
        fields["top_p"] = profile.top_p
    fields["enabled"] = profile.enabled
    # 每个字段写成一行，便于直接编辑；不按宽度折行。
    frontmatter = yaml.safe_dump(fields, allow_unicode=True, sort_keys=False, width=2**31 - 1)
    body = f"\n{profile.persona}\n" if profile.persona else ""
    return f"---\n{frontmatter}---\n{body}"


class AgentProfileStore:
    """一个 Profile 一个文件的存储；不缓存任何内容。"""

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)

    def list(self) -> list[StoredAgentProfile]:
        """按 agent_id 排序列出全部合格的 Profile；不合格的文件跳过并记日志。"""
        if not self._root.is_dir():
            return []
        profiles: list[StoredAgentProfile] = []
        for path in sorted(self._root.glob(f"*{_SUFFIX}")):
            try:
                profiles.append(self._load(path))
            except InvalidAgentProfileError as exc:
                logger.warning(
                    "跳过不合格的 Agent Profile 文件: path=%s, reason=%s",
                    path,
                    exc.details["reason"],
                )
        return profiles

    def get(self, agent_id: str) -> StoredAgentProfile | None:
        """读取一个 Profile；没有这个文件时返回 ``None``，文件不合格时抛出 ``InvalidAgentProfileError``。

        不能做文件名的 agent_id 不可能有对应的文件，按不存在返回，也不会被拼进路径。
        """
        if not _AGENT_ID_PATTERN.fullmatch(agent_id):
            return None
        path = self._path(agent_id)
        if not path.is_file():
            return None
        return self._load(path)

    def exists(self, agent_id: str) -> bool:
        """是否已有这个 agent_id 的文件，不论内容是否合格；新建据此拒绝覆盖。"""
        if not _AGENT_ID_PATTERN.fullmatch(agent_id):
            return False
        return self._path(agent_id).is_file()

    def save(self, profile: AgentProfile) -> StoredAgentProfile:
        """校验后原子写入，已有同名文件时整体替换；返回读回的 Profile。

        写入的文本先按读取的规则解析一遍，写进去的文件因此一定能读回。
        """
        agent_id = profile.agent_id or ""
        text = _render(profile)
        _parse(agent_id, text)
        path = self._path(agent_id)
        write_atomically(path, text.encode("utf-8"))
        return self._load(path)

    def delete(self, agent_id: str) -> bool:
        """删除一个 Profile，返回是否确有文件被删除。"""
        if not _AGENT_ID_PATTERN.fullmatch(agent_id):
            return False
        path = self._path(agent_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _path(self, agent_id: str) -> Path:
        return self._root / f"{agent_id}{_SUFFIX}"

    @staticmethod
    def _load(path: Path) -> StoredAgentProfile:
        agent_id = path.stem
        try:
            # utf-8-sig 去掉 Windows 编辑器写入的 BOM；文本模式把 CRLF 统一为换行。
            text = path.read_text(encoding="utf-8-sig")
            saved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except (OSError, UnicodeDecodeError) as exc:
            raise InvalidAgentProfileError(
                details={"reason": "unreadable", "agent_id": agent_id}
            ) from exc
        return StoredAgentProfile(profile=_parse(agent_id, text), saved_at=saved_at)


__all__ = ["FRONTMATTER_FIELDS", "AgentProfileStore", "profile_from_fields"]
