"""附件固定格式分派表。

W1-A 上传白名单、前端 file input ``accept`` 与 W1-B parser 分派必须共用
本模块冻结的同一格式集合（计划 7.1 节）。客户端声明的 MIME 只作为提示，
不能作为格式判断的唯一依据；格式冲突时返回稳定错误，不凭 MIME 选择
解析器，也不在 DOCX 解析失败后回退为 TXT。
"""

from __future__ import annotations

from dataclasses import dataclass

# 7.1 节冻结的规范媒体类型；扩展名大小写不敏感。
MEDIA_TYPE_PLAIN_TEXT = "text/plain"
MEDIA_TYPE_MARKDOWN = "text/markdown"
MEDIA_TYPE_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# 被视为"未声明"的通用 MIME 提示：此时完全按扩展名规范化。
_GENERIC_MEDIA_TYPES = frozenset({"", "application/octet-stream"})

# 旧版 Word 不进入本轮；单独保留提示文案，引导用户另存为 .docx。
_LEGACY_DOC_HINT = "旧版 .doc 文档暂不支持，请另存为 .docx 后上传"


@dataclass(frozen=True)
class AttachmentFormat:
    """一种批准附件格式的规范坐标。

    ``extensions`` 是含点的小写扩展名集合；首轮全部批准格式都在 Store
    中登记为 ``kind="document"``，required representation 为
    ``EXTRACTED_TEXT``。
    """

    canonical_media_type: str
    extensions: frozenset[str]
    display_label: str


#: 首轮批准格式的唯一事实源；顺序固定，供错误文案与分派遍历使用。
APPROVED_ATTACHMENT_FORMATS: tuple[AttachmentFormat, ...] = (
    AttachmentFormat(
        canonical_media_type=MEDIA_TYPE_PLAIN_TEXT,
        extensions=frozenset({".txt"}),
        display_label="纯文本（.txt）",
    ),
    AttachmentFormat(
        canonical_media_type=MEDIA_TYPE_MARKDOWN,
        extensions=frozenset({".md", ".markdown"}),
        display_label="Markdown（.md / .markdown）",
    ),
    AttachmentFormat(
        canonical_media_type=MEDIA_TYPE_DOCX,
        extensions=frozenset({".docx"}),
        display_label="Word 文档（.docx）",
    ),
)

_EXTENSION_TO_FORMAT: dict[str, AttachmentFormat] = {
    extension: format_
    for format_ in APPROVED_ATTACHMENT_FORMATS
    for extension in format_.extensions
}


class UnsupportedAttachmentFormatError(Exception):
    """上传格式不受支持或声明冲突的受控错误。

    ``message`` 是可直接展示给用户的安全文案；``reason`` 供日志选择
    文案分支，不作为客户端机器分支的稳定错误码。
    """

    def __init__(self, message: str, *, reason: str) -> None:
        self.reason = reason
        super().__init__(message)


def _normalize_declared_media_type(declared_media_type: str | None) -> str:
    """去掉参数部分并小写化声明的 MIME（``text/plain; charset=utf-8`` → ``text/plain``）。"""
    if declared_media_type is None:
        return ""
    return declared_media_type.split(";", 1)[0].strip().lower()


def _file_extension(file_name: str) -> str:
    """取文件名的小写扩展名（含点）；无扩展名或隐藏名返回空串。"""
    dot_index = file_name.rfind(".")
    if dot_index <= 0:
        # 无扩展名，或只有以点开头的隐藏名（``.docx`` 这类隐藏文件按无扩展名处理）。
        return ""
    return file_name[dot_index:].lower()


def resolve_attachment_format(
    file_name: str,
    declared_media_type: str | None,
) -> AttachmentFormat:
    """按扩展名分派批准格式，并核对声明的 MIME 是否与之冲突。

    - 扩展名不在批准集合内：返回稳定拒绝文案；``.doc`` 单独提示另存为
      ``.docx``；
    - 声明 MIME 缺失、为 ``application/octet-stream`` 等通用类型时直接按
      扩展名规范化；Markdown 声明为 ``text/plain`` 同样接受；
    - 声明 MIME 与扩展名对应规范类型明确冲突时拒绝，不猜测真实格式。
    """
    extension = _file_extension(file_name)
    format_ = _EXTENSION_TO_FORMAT.get(extension)
    if format_ is None:
        if extension == ".doc":
            raise UnsupportedAttachmentFormatError(
                _LEGACY_DOC_HINT,
                reason="legacy_doc_format",
            )
        approved = "、".join(item.display_label for item in APPROVED_ATTACHMENT_FORMATS)
        raise UnsupportedAttachmentFormatError(
            f"不支持的附件格式，仅支持：{approved}",
            reason="unsupported_format",
        )

    declared = _normalize_declared_media_type(declared_media_type)
    generic = declared in _GENERIC_MEDIA_TYPES
    markdown_as_plain = (
        format_.canonical_media_type == MEDIA_TYPE_MARKDOWN and declared == MEDIA_TYPE_PLAIN_TEXT
    )
    if not generic and not markdown_as_plain and declared != format_.canonical_media_type:
        raise UnsupportedAttachmentFormatError(
            f"声明的文件类型与扩展名 {extension} 冲突，请确认文件格式后重新上传",
            reason="conflicting_media_type",
        )
    return format_


__all__ = [
    "APPROVED_ATTACHMENT_FORMATS",
    "AttachmentFormat",
    "MEDIA_TYPE_DOCX",
    "MEDIA_TYPE_MARKDOWN",
    "MEDIA_TYPE_PLAIN_TEXT",
    "UnsupportedAttachmentFormatError",
    "resolve_attachment_format",
]
