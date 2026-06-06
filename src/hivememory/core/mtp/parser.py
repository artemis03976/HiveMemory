"""
MTP parser & filter parser。

协议语法: ⟪ VERB | TARGET | ARGS ⟫
- VERB: SEARCH, READ, RUN, WRITE, UPDATE, CALL
- TARGET: *, alias, [alias1, alias2]
- ARGS: key="value" / key=`raw content` / key=["a","b"]
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from hivememory.core.models import Identity, MemoryType
from hivememory.engines.retrieval.models import QueryFilters
from hivememory.core.mtp.exceptions import MTPParseError
from hivememory.core.mtp.models import (
    MTPCommand,
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTP_SEPARATOR,
    MTPTarget,
    MTPVerb,
)

logger = logging.getLogger(__name__)


class MTPParser:
    """
    MTP 协议解析器。

    负责将原始 MTP 指令文本解析为结构化的 MTPCommand 对象。
    """

    _COMMAND_PATTERN = re.compile(r"\u27EA\s*(.*?)\s*\u27EB", re.DOTALL)
    _KV_PATTERN = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
    _RAW_PATTERN = re.compile(r"(\w+)\s*=\s*`(.*?)`", re.DOTALL)
    _LIST_ARG_PATTERN = re.compile(r"(\w+)\s*=\s*\[\s*([^\]]*)\s*\]")
    _LIST_TARGET_PATTERN = re.compile(r"\[\s*([\w\s,]+)\s*\]")
    _VALID_VERBS = {v.value for v in MTPVerb}

    def parse(self, text: str) -> MTPCommand:
        """
        解析 MTP 指令文本。

        Args:
            text: 原始指令文本（包含定界符）

        Returns:
            MTPCommand: 结构化指令对象
        """
        match = self._COMMAND_PATTERN.search(text)
        if not match:
            raise MTPParseError(
                message_key="mtp.parse.no_command",
                params={
                    "left_delimiter": MTP_LEFT_DELIMITER,
                    "right_delimiter": MTP_RIGHT_DELIMITER,
                },
            )

        inner = match.group(1).strip()
        raw_text = match.group(0)
        verb_str, target_str, args_str = self._split_segments(inner)

        verb_upper = verb_str.upper()
        if verb_upper not in self._VALID_VERBS:
            raise MTPParseError(
                message_key="mtp.parse.unknown_verb",
                params={
                    "verb": verb_str,
                    "valid_verbs": ", ".join(sorted(self._VALID_VERBS)),
                },
            )

        return MTPCommand(
            verb=MTPVerb(verb_upper),
            target=self._parse_target(target_str),
            args=self._parse_args(args_str),
            raw_text=raw_text,
        )

    def complete_and_parse(self, text: str) -> MTPCommand:
        """
        自动补全右定界符后再解析。

        适用于 stop sequence 截断场景。
        """
        if MTP_RIGHT_DELIMITER not in text:
            text = text.rstrip() + " " + MTP_RIGHT_DELIMITER
        return self.parse(text)

    def detect_command(self, text: str) -> bool:
        """快速检测文本中是否存在 MTP 指令前缀。"""
        return MTP_LEFT_DELIMITER in text

    def _split_segments(self, inner: str) -> Tuple[str, str, str]:
        """
        按前两个 `|` 进行分段。

        仅前两个分隔符用于拆分，ARGS 内部的 `|` 视为内容。
        """
        first_pipe = inner.find(MTP_SEPARATOR)
        if first_pipe == -1:
            raise MTPParseError(
                message_key="mtp.parse.missing_separator",
                params={"separator": MTP_SEPARATOR},
            )

        verb_str = inner[:first_pipe].strip()
        rest = inner[first_pipe + 1 :]

        second_pipe = rest.find(MTP_SEPARATOR)
        if second_pipe == -1:
            return verb_str, rest.strip(), ""
        return verb_str, rest[:second_pipe].strip(), rest[second_pipe + 1 :].strip()

    def _parse_target(self, target_str: str) -> MTPTarget:
        """解析 TARGET 字段（通配、单别名、列表别名）。"""
        target_str = target_str.strip()
        if target_str in ("*", "global"):
            return MTPTarget(is_wildcard=True)

        list_match = self._LIST_TARGET_PATTERN.match(target_str)
        if list_match:
            items = list_match.group(1)
            aliases = [a.strip() for a in items.split(",") if a.strip()]
            return MTPTarget(aliases=aliases)

        if target_str:
            return MTPTarget(aliases=[target_str])
        return MTPTarget()

    def _parse_args(self, args_str: str) -> Dict[str, str]:
        """
        解析 ARGS 字段。

        支持:
            - key="value"
            - key=`raw content` (可多行)
            - key=[...]
        """
        if not args_str:
            return {}

        args: Dict[str, str] = {}

        for match in self._LIST_ARG_PATTERN.finditer(args_str):
            key, list_content = match.group(1), match.group(2)
            items = []
            for item in list_content.split(","):
                item = item.strip().strip('"').strip("'")
                if item:
                    items.append(item)
            args[key] = json.dumps(items)

        for match in self._RAW_PATTERN.finditer(args_str):
            key, value = match.group(1), match.group(2)
            if key not in args:
                args[key] = value.strip()

        remaining = self._LIST_ARG_PATTERN.sub("", args_str)
        remaining = self._RAW_PATTERN.sub("", remaining)
        for match in self._KV_PATTERN.finditer(remaining):
            key, value = match.group(1), match.group(2)
            if key not in args:
                args[key] = value

        return args


_FILTER_TYPE_MAP: Dict[str, MemoryType] = {
    "code": MemoryType.CODE_SNIPPET,
    "code_snippet": MemoryType.CODE_SNIPPET,
    "fact": MemoryType.FACT,
    "url": MemoryType.URL_RESOURCE,
    "url_resource": MemoryType.URL_RESOURCE,
    "reflection": MemoryType.REFLECTION,
    "profile": MemoryType.USER_PROFILE,
    "user_profile": MemoryType.USER_PROFILE,
    "wip": MemoryType.WORK_IN_PROGRESS,
    "work_in_progress": MemoryType.WORK_IN_PROGRESS,
    "agent_profile": MemoryType.AGENT_PROFILE,
    "agent": MemoryType.AGENT_PROFILE,
}


class MTPFilterParser:
    """
    SEARCH.filter 参数解析器。

    语法: key:value，多个用空格分隔。
    支持 key: type / tag / agent / confidence。
    """

    def parse(self, filter_str: str) -> Tuple[Optional[QueryFilters], List[str]]:
        """
        宽容解析 filter 字符串并返回 QueryFilters 与警告列表。

        解析失败不会抛异常，而是降级为无 filter 并返回 warnings。
        """
        if not filter_str or not filter_str.strip():
            return None, []

        warnings: List[str] = []
        try:
            memory_type = None
            tags: List[str] = []
            source_agent_id = None
            min_confidence = 0.0

            for token in filter_str.strip().split():
                if ":" not in token:
                    warnings.append(
                        f"Note: Filter token '{token}' was ignored (missing ':' separator)."
                    )
                    continue

                key, _, value = token.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if not key or not value:
                    warnings.append(
                        f"Note: Filter token '{token}' was ignored (empty key or value)."
                    )
                    continue

                if key == "type":
                    mapped = _FILTER_TYPE_MAP.get(value.lower())
                    if mapped is not None:
                        memory_type = mapped
                    else:
                        warnings.append(
                            f"Note: Unknown filter type '{value}' was ignored. "
                            f"Valid types: CODE, FACT, URL, REFLECTION, PROFILE, WIP."
                        )
                elif key == "tag":
                    tags.append(value)
                elif key == "agent":
                    source_agent_id = value
                elif key == "confidence":
                    try:
                        parsed = float(value)
                        if 0.0 < parsed <= 1.0:
                            min_confidence = parsed
                        else:
                            warnings.append(
                                f"Note: Filter confidence value {parsed} is out of range (0,1] and was ignored."
                            )
                    except ValueError:
                        warnings.append(
                            f"Note: Filter confidence value '{value}' is not a valid number and was ignored."
                        )
                else:
                    warnings.append(f"Note: Unknown filter key '{key}' was ignored.")

            mtp_identity = Identity(agent_id=source_agent_id) if source_agent_id else None
            filters = QueryFilters(
                identity=mtp_identity,
                memory_type=memory_type,
                tags=tags,
                min_confidence=min_confidence,
            )
            if filters.is_empty():
                return None, warnings
            return filters, warnings

        except Exception as e:
            logger.warning(f"MTP filter parse failed: {e}")
            warnings.append("Note: Filter parsing failed. Results may be broader than expected.")
            return None, warnings


def create_parser() -> MTPParser:
    """创建 MTPParser 实例。"""
    return MTPParser()


def create_filter_parser() -> MTPFilterParser:
    """创建 MTPFilterParser 实例。"""
    return MTPFilterParser()


__all__ = [
    "MTPParser",
    "MTPFilterParser",
    "_FILTER_TYPE_MAP",
    "create_parser",
    "create_filter_parser",
]
