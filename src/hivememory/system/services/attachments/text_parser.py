"""TXT 与 Markdown 的确定性文本解码器（计划 7.3 节）。

两种格式共用同一条解码链路，仅在 content_object 的 ``format`` 字段上
区分：Markdown 保留标题、列表、代码围栏、表格、链接、frontmatter 和
内嵌 HTML 的原始文本，不构建 AST、不渲染、不执行、不展开 include。

解码采用固定顺序：UTF-8 BOM → UTF-16 LE/BE BOM → 无 BOM 时仅 UTF-8
严格解码；不尝试 GBK/GB18030 或无 BOM 的 UTF-16，不使用系统默认编码、
概率探测或 ``errors="ignore"``/``errors="replace"``。
"""

from __future__ import annotations

import re
from collections.abc import Callable

from hivememory.system.services.attachments.errors import (
    CONTENT_UNREADABLE,
    RESOURCE_LIMIT,
    AttachmentParseError,
)
from hivememory.system.services.attachments.limits import ParseBudget, ParseLimits
from hivememory.system.services.attachments.models import (
    FORMAT_MARKDOWN,
    FORMAT_PLAIN_TEXT,
    LOCATOR_KIND_LINE,
    AttachmentContentBuilder,
    ParsedAttachmentContent,
)

#: UTF-8 / UTF-16 LE / UTF-16 BE 的字节顺序标记。
_UTF8_BOM = b"\xef\xbb\xbf"
_UTF16LE_BOM = b"\xff\xfe"
_UTF16BE_BOM = b"\xfe\xff"
_UTF32LE_BOM = b"\xff\xfe\x00\x00"
_UTF32BE_BOM = b"\x00\x00\xfe\xff"

#: 规范化后仍非法的控制字符：除 TAB/LF/FF 外的 C0 控制符、DEL 与 C1 控制符。
_INVALID_CONTROL_PATTERN = re.compile("[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]")


def _decode_bytes(raw: bytes) -> str:
    """按固定 BOM 顺序严格解码；UTF-32 显式拒绝，失败归入非法编码。"""
    if raw.startswith(_UTF32LE_BOM) or raw.startswith(_UTF32BE_BOM):
        raise AttachmentParseError(
            CONTENT_UNREADABLE,
            "附件使用了不支持的 UTF-32 编码，请转存为 UTF-8 后重新上传",
            params={"reason": "unsupported_encoding", "encoding": "utf-32"},
        )
    try:
        if raw.startswith(_UTF8_BOM):
            return raw[len(_UTF8_BOM) :].decode("utf-8")
        if raw.startswith(_UTF16LE_BOM):
            return raw[len(_UTF16LE_BOM) :].decode("utf-16-le")
        if raw.startswith(_UTF16BE_BOM):
            return raw[len(_UTF16BE_BOM) :].decode("utf-16-be")
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AttachmentParseError(
            CONTENT_UNREADABLE,
            "附件不是有效的 UTF-8/UTF-16 文本，请转存为 UTF-8 后重新上传",
            params={"reason": "unsupported_encoding", "detail": exc.reason},
        ) from exc


class TextAttachmentParser:
    """TXT/Markdown 解析器：确定性解码 + 行级 locator。

    输出将 CRLF 和 CR 统一为 LF，保留其他空白、缩进、空行和末尾换行，
    不执行全文 ``strip()``、Unicode 兼容归一化或自动重排。
    """

    def __init__(self, *, content_format: str) -> None:
        if content_format not in {FORMAT_PLAIN_TEXT, FORMAT_MARKDOWN}:
            raise ValueError(f"文本解析器不接受格式：{content_format}")
        self.content_format = content_format

    @property
    def producer(self) -> str:
        """解析实现的稳定身份。"""
        return "text_decode"

    @property
    def producer_version(self) -> str:
        """解码/定位规则版本；改变规则时必须提升。"""
        return "1"

    def parse(
        self,
        raw: bytes,
        *,
        limits: ParseLimits,
        source_raw_revision: int,
        source_raw_hash: str,
        clock: Callable[[], float] | None = None,
    ) -> ParsedAttachmentContent:
        """把 RAW bytes 转为带行级 locator 的确定性正文。"""
        if len(raw) > limits.max_raw_bytes:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "文件超过大小上限，请缩小后重新上传",
                params={"reason": "raw_input_limit"},
            )
        budget = ParseBudget(limits, clock) if clock else ParseBudget(limits)

        text = _decode_bytes(raw)
        budget.check()
        # 换行规范化：CRLF 与 CR 统一为 LF；FF 作为换页符保留。
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        budget.check()

        invalid = _INVALID_CONTROL_PATTERN.search(text)
        if invalid is not None:
            raise AttachmentParseError(
                CONTENT_UNREADABLE,
                "附件包含无法展示的控制字符，请清理后重新上传",
                params={"reason": "invalid_text"},
            )
        if not text.strip():
            raise AttachmentParseError(
                CONTENT_UNREADABLE,
                "附件中没有可提取的正文内容",
                params={"reason": "empty_content"},
            )

        builder = AttachmentContentBuilder(
            content_format=self.content_format,
            source_raw_revision=source_raw_revision,
            source_raw_hash=source_raw_hash,
            limits=limits,
        )
        # 行号从 1 开始且计入空行；只有非空行记录 line locator。
        # split 后仅在行间补分隔符，原始末尾换行结构保持不变。
        lines = text.split("\n")
        offset = 0
        for index, line in enumerate(lines):
            if line:
                builder.add_locator(
                    kind=LOCATOR_KIND_LINE,
                    number=index + 1,
                    start=offset,
                    end=offset + len(line),
                )
            builder.append_text(line)
            if index < len(lines) - 1:
                builder.append_text("\n")
            offset += len(line) + 1

        budget.check()
        return builder.build(producer=self.producer, producer_version=self.producer_version)


__all__ = ["TextAttachmentParser"]
