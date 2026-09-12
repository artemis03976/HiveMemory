"""Chat 附件服务：上传接收、解析交接、确定性 parser 与共享结果模型。"""

from .docx_parser import DocxAttachmentParser
from .errors import (
    CONTENT_UNREADABLE,
    EXECUTION_FAILURE,
    RESOURCE_LIMIT,
    AttachmentParseError,
)
from .formats import (
    APPROVED_ATTACHMENT_FORMATS,
    MEDIA_TYPE_DOCX,
    MEDIA_TYPE_MARKDOWN,
    MEDIA_TYPE_PLAIN_TEXT,
    AttachmentFormat,
    UnsupportedAttachmentFormatError,
    resolve_attachment_format,
)
from .limits import ParseBudget
from .models import (
    FORMAT_MARKDOWN,
    FORMAT_PLAIN_TEXT,
    LOCATOR_KIND_LINE,
    LOCATOR_KIND_PARAGRAPH,
    SCHEMA_VERSION,
    AttachmentContentBuilder,
    ParsedAttachmentContent,
    canonical_content_bytes,
)
from .parse_service import AttachmentParseService
from .parser import AttachmentParser, resolve_parser
from .text_parser import TextAttachmentParser

__all__ = [
    "APPROVED_ATTACHMENT_FORMATS",
    "CONTENT_UNREADABLE",
    "EXECUTION_FAILURE",
    "FORMAT_MARKDOWN",
    "FORMAT_PLAIN_TEXT",
    "LOCATOR_KIND_LINE",
    "LOCATOR_KIND_PARAGRAPH",
    "MEDIA_TYPE_DOCX",
    "MEDIA_TYPE_MARKDOWN",
    "MEDIA_TYPE_PLAIN_TEXT",
    "RESOURCE_LIMIT",
    "SCHEMA_VERSION",
    "AttachmentContentBuilder",
    "AttachmentParseError",
    "AttachmentParseService",
    "AttachmentParser",
    "AttachmentFormat",
    "DocxAttachmentParser",
    "ParseBudget",
    "ParsedAttachmentContent",
    "TextAttachmentParser",
    "UnsupportedAttachmentFormatError",
    "canonical_content_bytes",
    "resolve_attachment_format",
    "resolve_parser",
]
