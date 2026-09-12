"""附件解析器的窄端口与固定格式分派（计划 7.2 / 7.8 节）。

解析器只消费明确的 RAW 输入（bytes、RAW 版本坐标、固定解析限制），
不依赖 HTTP UploadFile、临时路径、Store 对象、Agent 或 LLM。分派使用
与 W1-A 上传白名单相同的冻结格式集合；不凭 MIME 选择任意解析器，也
不在 DOCX 解析失败后回退为 TXT。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from hivememory.system.config.attachments import AttachmentParserConfig
from hivememory.system.services.attachments.docx_parser import DocxAttachmentParser
from hivememory.system.services.attachments.errors import (
    EXECUTION_FAILURE,
    AttachmentParseError,
)
from hivememory.system.services.attachments.formats import (
    MEDIA_TYPE_DOCX,
    MEDIA_TYPE_MARKDOWN,
    MEDIA_TYPE_PLAIN_TEXT,
)
from hivememory.system.services.attachments.models import ParsedAttachmentContent
from hivememory.system.services.attachments.text_parser import TextAttachmentParser

#: docx 规范媒体类型在分派表中显式出现，供类型检查与测试引用。
_SUPPORTED_MEDIA_TYPES = frozenset(
    {MEDIA_TYPE_PLAIN_TEXT, MEDIA_TYPE_MARKDOWN, MEDIA_TYPE_DOCX},
)


@runtime_checkable
class AttachmentParser(Protocol):
    """单一格式确定性解析器的窄端口。"""

    @property
    def producer(self) -> str: ...

    @property
    def producer_version(self) -> str: ...

    def parse(
        self,
        raw: bytes,
        *,
        config: AttachmentParserConfig,
        source_raw_revision: int,
        source_raw_hash: str,
        clock: Callable[[], float] | None = None,
    ) -> ParsedAttachmentContent: ...


def resolve_parser(media_type: str) -> AttachmentParser:
    """按规范化媒体类型分派第一方解析器。

    上传边界（W1-A）已保证媒体类型属于批准集合；到达分派仍无法识别时
    属于内部交接故障，使用 ``execution_failure`` 收尾。
    """
    if media_type == MEDIA_TYPE_PLAIN_TEXT:
        return TextAttachmentParser(content_format="plain_text")
    if media_type == MEDIA_TYPE_MARKDOWN:
        return TextAttachmentParser(content_format="markdown")
    if media_type == MEDIA_TYPE_DOCX:
        return DocxAttachmentParser()
    raise AttachmentParseError(
        EXECUTION_FAILURE,
        "附件解析失败，请重新上传",
        params={"reason": "parser_dispatch_failed", "media_type": media_type},
    )


__all__ = ["AttachmentParser", "resolve_parser", "_SUPPORTED_MEDIA_TYPES"]
