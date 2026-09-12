"""附件固定格式分派表的单元测试。

被测对象：``system/services/attachments/formats.py`` 的批准格式集合与
``resolve_attachment_format`` 分派规则（计划 7.1 节）。
"""

import pytest

from hivememory.system.services.attachments import (
    APPROVED_ATTACHMENT_FORMATS,
    MEDIA_TYPE_DOCX,
    MEDIA_TYPE_MARKDOWN,
    MEDIA_TYPE_PLAIN_TEXT,
    UnsupportedAttachmentFormatError,
    resolve_attachment_format,
)


def test_approved_formats_match_frozen_format_table() -> None:
    """捕获批准格式集合被静默增删，导致前后端 accept 与白名单漂移。"""
    table = [
        (sorted(format_.extensions), format_.canonical_media_type)
        for format_ in APPROVED_ATTACHMENT_FORMATS
    ]
    assert table == [
        ([".txt"], MEDIA_TYPE_PLAIN_TEXT),
        ([".markdown", ".md"], MEDIA_TYPE_MARKDOWN),
        ([".docx"], MEDIA_TYPE_DOCX),
    ]


def test_extension_dispatch_is_case_insensitive() -> None:
    """捕获大写扩展名被误拒或错误分派。"""
    assert resolve_attachment_format("Report.TXT", None).canonical_media_type == (
        MEDIA_TYPE_PLAIN_TEXT
    )
    assert resolve_attachment_format("Notes.Md", None).canonical_media_type == (MEDIA_TYPE_MARKDOWN)
    assert resolve_attachment_format("Doc.DOCX", None).canonical_media_type == MEDIA_TYPE_DOCX


def test_generic_or_missing_mime_is_normalized_by_extension() -> None:
    """捕获浏览器 MIME 缺失或为通用类型时被误判为冲突。"""
    for declared in (None, "", "application/octet-stream"):
        assert resolve_attachment_format("a.md", declared).canonical_media_type == (
            MEDIA_TYPE_MARKDOWN
        )
        assert resolve_attachment_format("a.txt", declared).canonical_media_type == (
            MEDIA_TYPE_PLAIN_TEXT
        )


def test_markdown_declared_as_plain_text_is_accepted() -> None:
    """捕获浏览器把 Markdown 报告为 text/plain 时被误判为冲突。"""
    assert resolve_attachment_format("a.md", "text/plain").canonical_media_type == (
        MEDIA_TYPE_MARKDOWN
    )


def test_mime_with_parameters_is_matched_against_canonical_type() -> None:
    """捕获带参数的 MIME（如 charset）被误判为冲突。"""
    assert (
        resolve_attachment_format("a.md", "text/markdown; charset=utf-8").canonical_media_type
        == MEDIA_TYPE_MARKDOWN
    )
    assert (
        resolve_attachment_format("a.txt", "text/plain; charset=utf-8").canonical_media_type
        == MEDIA_TYPE_PLAIN_TEXT
    )


def test_conflicting_media_type_is_rejected_stably() -> None:
    """捕获声明 MIME 与扩展名明确冲突时被静默放行。"""
    with pytest.raises(UnsupportedAttachmentFormatError) as text_error:
        resolve_attachment_format("a.txt", MEDIA_TYPE_DOCX)
    assert text_error.value.reason == "conflicting_media_type"
    with pytest.raises(UnsupportedAttachmentFormatError) as docx_error:
        resolve_attachment_format("a.docx", "text/plain")
    assert docx_error.value.reason == "conflicting_media_type"


def test_legacy_doc_gets_save_as_docx_hint() -> None:
    """捕获旧版 .doc 得到通用错误而非"另存为 .docx"提示。"""
    with pytest.raises(UnsupportedAttachmentFormatError) as error:
        resolve_attachment_format("old.doc", "application/msword")
    assert error.value.reason == "legacy_doc_format"
    assert ".docx" in error.value.args[0]


@pytest.mark.parametrize("file_name", ["archive.pdf", "doc.rtf", "docx", "noext"])
def test_unsupported_or_missing_extension_is_rejected(file_name: str) -> None:
    """捕获不批准格式或无扩展名文件被静默接受。"""
    with pytest.raises(UnsupportedAttachmentFormatError) as error:
        resolve_attachment_format(file_name, None)
    assert error.value.reason == "unsupported_format"
