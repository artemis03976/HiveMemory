"""Chat 附件领域服务：固定格式分派、（W1-B）确定性 parser 与共享结果模型。"""

from .formats import (
    APPROVED_ATTACHMENT_FORMATS,
    MEDIA_TYPE_DOCX,
    MEDIA_TYPE_MARKDOWN,
    MEDIA_TYPE_PLAIN_TEXT,
    AttachmentFormat,
    UnsupportedAttachmentFormatError,
    resolve_attachment_format,
)

__all__ = [
    "APPROVED_ATTACHMENT_FORMATS",
    "AttachmentFormat",
    "MEDIA_TYPE_DOCX",
    "MEDIA_TYPE_MARKDOWN",
    "MEDIA_TYPE_PLAIN_TEXT",
    "UnsupportedAttachmentFormatError",
    "resolve_attachment_format",
]
