"""W1-B 共享解析结果模型（计划 7.5 节）。

正文与 locator 校验、大小累计和 canonical hash 只在本模块实现一次；
TXT/Markdown 与 DOCX 解析器在返回前通过 :class:`AttachmentContentBuilder`
完成构造，W1-C 复用结果并核对来源，不再次解释附件正文。

content_object 首轮约定（schema_version=1）：

- ``format``：``plain_text`` 或 ``markdown``，不改变 representation kind；
- ``text``：转换后的完整正文；
- ``source_raw``：当前 asset 唯一 RAW 的 ``revision`` 与 ``content_hash``；
- ``locators``：正文字符区间与源位置的映射，按正文顺序排列；
- ``warnings``：``message_key`` 与受控 ``params`` 组成的 warning record，
  去重且排序稳定，无 warning 时为空列表。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from hivememory.system.services.attachments.errors import (
    CONTENT_UNREADABLE,
    EXECUTION_FAILURE,
    RESOURCE_LIMIT,
    AttachmentParseError,
)
from hivememory.system.services.attachments.limits import ParseLimits

#: 内容结构版本；改变 content_object 字段结构时必须提升并同步 producer version。
SCHEMA_VERSION = 1

FORMAT_PLAIN_TEXT = "plain_text"
FORMAT_MARKDOWN = "markdown"

#: locator 的来源定位粒度；位置属于源文档结构而非页面坐标。
LOCATOR_KIND_LINE = "line"
LOCATOR_KIND_PARAGRAPH = "paragraph"


def canonical_content_bytes(content_object: dict[str, Any]) -> bytes:
    """完整 content_object 的 canonical JSON UTF-8 编码。

    键排序、紧凑分隔符、非 ASCII 字符直接编码、无额外换行；数组顺序由
    各解析器的固定规则保证。W1-E/W1-F 复用该 hash 时必须使用同一规则。
    """
    return json.dumps(
        content_object,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ParsedAttachmentContent:
    """所有解析器共用的成功结果结构。"""

    producer: str
    producer_version: str
    content_object: dict[str, Any]
    content_hash: str

    @property
    def content_format(self) -> str:
        """提取结果的文字格式（不改变 representation kind）。"""
        return self.content_object["format"]

    @property
    def text(self) -> str:
        """转换后的完整正文。"""
        return self.content_object["text"]

    @property
    def source_raw(self) -> dict[str, Any]:
        """当前 asset 唯一 RAW 的版本坐标。"""
        return self.content_object["source_raw"]


class AttachmentContentBuilder:
    """增量构造 content_object 的唯一入口。

    正文 UTF-8 大小、locator 数量与顺序在追加时累计检查；canonical
    内容大小与 hash 在 :meth:`build` 时一次性计算。空正文在这里统一
    归入 ``content_unreadable``（``empty_content``），不产生半份成功结果。
    """

    def __init__(
        self,
        *,
        content_format: str,
        source_raw_revision: int,
        source_raw_hash: str,
        limits: ParseLimits,
    ) -> None:
        if content_format not in {FORMAT_PLAIN_TEXT, FORMAT_MARKDOWN}:
            raise ValueError(f"未知的提取结果格式：{content_format}")
        if not source_raw_hash:
            raise ValueError("source_raw_hash 不能为空")
        self._format = content_format
        self._source_raw = {
            "revision": source_raw_revision,
            "content_hash": source_raw_hash,
        }
        self._limits = limits
        self._segments: list[str] = []
        self._utf8_size = 0
        self._char_count = 0
        self._locators: list[dict[str, Any]] = []
        self._warnings: list[dict[str, Any]] = []

    @property
    def char_count(self) -> int:
        """当前已累计的正文 Unicode 码点数。"""
        return self._char_count

    def append_text(self, text: str) -> None:
        """累计一段正文；按 UTF-8 字节数累计并强制正文大小上限。"""
        if not text:
            return
        self._utf8_size += len(text.encode("utf-8"))
        if self._utf8_size > self._limits.max_extracted_text_bytes:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "提取的正文超过大小上限，请缩小文件后重新上传",
                params={"reason": "extracted_text_limit"},
            )
        self._segments.append(text)
        self._char_count += len(text)

    def add_locator(
        self,
        *,
        kind: str,
        number: int,
        start: int,
        end: int,
    ) -> None:
        """追加一条 locator；区间为正文码点坐标，按正文顺序排列。"""
        if start < 0 or end < start:
            raise AttachmentParseError(
                EXECUTION_FAILURE,
                "附件解析失败，请重新上传",
                params={"reason": "invalid_locator_interval"},
            )
        if len(self._locators) >= self._limits.max_locator_count:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "附件结构过于复杂，请缩小文件后重新上传",
                params={"reason": "locator_count_limit"},
            )
        self._locators.append(
            {"kind": kind, "number": number, "start": start, "end": end},
        )

    def add_warning(self, message_key: str, params: dict[str, Any] | None = None) -> None:
        """登记一条覆盖范围 warning；去重与稳定排序在 build 时统一完成。"""
        self._warnings.append({"message_key": message_key, "params": dict(params or {})})

    def build(self, *, producer: str, producer_version: str) -> ParsedAttachmentContent:
        """校验累计结果并冻结为带 canonical hash 的最终内容。"""
        text = "".join(self._segments)
        if not text.strip():
            raise AttachmentParseError(
                CONTENT_UNREADABLE,
                "附件中没有可提取的正文内容",
                params={"reason": "empty_content"},
            )

        for locator in self._locators:
            if locator["end"] > len(text):
                raise AttachmentParseError(
                    EXECUTION_FAILURE,
                    "附件解析失败，请重新上传",
                    params={"reason": "locator_out_of_range"},
                )
        starts = [locator["start"] for locator in self._locators]
        if starts != sorted(starts):
            raise AttachmentParseError(
                EXECUTION_FAILURE,
                "附件解析失败，请重新上传",
                params={"reason": "locator_order_violation"},
            )

        deduplicated = {
            (_warning["message_key"], canonical_content_bytes(_warning["params"])): _warning
            for _warning in self._warnings
        }
        warnings = [
            deduplicated[key] for key in sorted(deduplicated, key=lambda item: (item[0], item[1]))
        ]

        content_object: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "format": self._format,
            "text": text,
            "source_raw": self._source_raw,
            "locators": self._locators,
            "warnings": warnings,
        }
        canonical = canonical_content_bytes(content_object)
        if len(canonical) > self._limits.max_canonical_content_bytes:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "附件解析结果超过大小上限，请缩小文件后重新上传",
                params={"reason": "canonical_content_limit"},
            )
        return ParsedAttachmentContent(
            producer=producer,
            producer_version=producer_version,
            content_object=content_object,
            content_hash=hashlib.sha256(canonical).hexdigest(),
        )


__all__ = [
    "FORMAT_MARKDOWN",
    "FORMAT_PLAIN_TEXT",
    "LOCATOR_KIND_LINE",
    "LOCATOR_KIND_PARAGRAPH",
    "SCHEMA_VERSION",
    "AttachmentContentBuilder",
    "ParsedAttachmentContent",
    "canonical_content_bytes",
]
