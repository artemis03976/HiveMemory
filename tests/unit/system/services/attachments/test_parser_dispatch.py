"""附件解析器固定分派的单元测试（计划 7.1 / 7.8 节）。

被测对象：``system/services/attachments/parser.py`` 的 ``resolve_parser``；
分派集合与 W1-A 上传白名单共用同一冻结格式表。
"""

import pytest

from hivememory.system.services.attachments import (
    EXECUTION_FAILURE,
    MEDIA_TYPE_DOCX,
    MEDIA_TYPE_MARKDOWN,
    MEDIA_TYPE_PLAIN_TEXT,
    AttachmentParseError,
    DocxAttachmentParser,
    TextAttachmentParser,
    resolve_parser,
)


@pytest.mark.parametrize(
    ("media_type", "expected_type", "expected_format"),
    [
        (MEDIA_TYPE_PLAIN_TEXT, TextAttachmentParser, "plain_text"),
        (MEDIA_TYPE_MARKDOWN, TextAttachmentParser, "markdown"),
        (MEDIA_TYPE_DOCX, DocxAttachmentParser, None),
    ],
)
def test_approved_media_types_dispatch_to_dedicated_parsers(
    media_type: str,
    expected_type: type,
    expected_format: str | None,
) -> None:
    """捕获分派结果与批准格式集合漂移。"""
    parser = resolve_parser(media_type)

    assert isinstance(parser, expected_type)
    if expected_format is not None:
        assert parser.content_format == expected_format


def test_unapproved_media_type_fails_as_internal_execution_failure() -> None:
    """捕获未批准媒体类型在分派层被放行（上传边界应已拒绝）。"""
    with pytest.raises(AttachmentParseError) as error:
        resolve_parser("application/pdf")

    assert (error.value.category, error.value.params["reason"]) == (
        EXECUTION_FAILURE,
        "parser_dispatch_failed",
    )
