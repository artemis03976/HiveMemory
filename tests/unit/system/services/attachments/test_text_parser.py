"""TXT/Markdown 确定性文本解码器的单元测试（计划 15.4 节）。

被测对象：``system/services/attachments/text_parser.py``。正文与 locator
采用独立确认的期望值；码点区间不得误用 UTF-16 下标。
"""

import pytest

from hivememory.system.services.attachments import (
    CONTENT_UNREADABLE,
    RESOURCE_LIMIT,
    AttachmentParseError,
    ParseLimits,
    TextAttachmentParser,
)

#: test_result_model 独立冻结样例的哈希：文本 "a\nb" + source_raw(rev=1, raw-hash)。
SAMPLE_SHA256 = "be54143b87a6c9271f0f1f42caa1a5e67a7e617d8a8bc445c0a5a3ad2c64402a"


def _parse(
    raw: bytes,
    *,
    content_format: str = "plain_text",
    **overrides,
):
    return TextAttachmentParser(content_format=content_format).parse(
        raw,
        limits=ParseLimits(**overrides),
        source_raw_revision=1,
        source_raw_hash="raw-hash",
    )


def test_decoding_normalizes_newlines_and_locates_non_empty_lines() -> None:
    """捕获 CRLF/CR 未统一、空行编号未计入或 locator 偏移漂移。"""
    result = _parse("第一行\r\nsecond line\n\n  indented  \n".encode())

    assert result.text == "第一行\nsecond line\n\n  indented  \n"
    assert result.content_object["locators"] == [
        {"kind": "line", "number": 1, "start": 0, "end": 3},
        {"kind": "line", "number": 2, "start": 4, "end": 15},
        {"kind": "line", "number": 4, "start": 17, "end": 29},
    ]
    assert result.content_object["warnings"] == []


def test_full_result_matches_frozen_sample_hash() -> None:
    """捕获 hash 计算与冻结样例（独立确认值）不一致。"""
    result = _parse(b"a\nb")

    assert result.text == "a\nb"
    assert result.content_hash == SAMPLE_SHA256
    assert result.content_object["source_raw"] == {
        "revision": 1,
        "content_hash": "raw-hash",
    }
    assert (result.producer, result.producer_version) == ("text_decode", "1")


def test_repeated_parse_is_deterministic() -> None:
    """捕获同一输入重复解析产生不同 hash 或内容。"""
    first = _parse("重复解析\n确定性\n".encode("utf-8"))
    second = _parse("重复解析\n确定性\n".encode("utf-8"))

    assert second.content_hash == first.content_hash
    assert second.content_object == first.content_object


def test_utf8_bom_is_stripped_and_content_strictly_decoded() -> None:
    """捕获 BOM 残留进入正文或非严格解码吞掉坏字节。"""
    result = _parse(b"\xef\xbb\xbfline one\n")

    assert result.text == "line one\n"
    assert result.content_object["locators"] == [
        {"kind": "line", "number": 1, "start": 0, "end": 8},
    ]


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_utf16_bom_decodes_with_codepoint_locators(encoding: str) -> None:
    """捕获端序错误解码或 locator 误用 UTF-16 下标（含非 BMP 字符）。"""
    raw = b"\xff\xfe" if encoding == "utf-16-le" else b"\xfe\xff"
    payload = "中文🚀测试\n".encode(encoding)
    result = _parse(raw + payload)

    assert result.text == "中文🚀测试\n"
    # 5 个 Unicode 码点（🚀 是 1 个码点、2 个 UTF-16 单元），不是 UTF-16 下标。
    assert result.content_object["locators"] == [
        {"kind": "line", "number": 1, "start": 0, "end": 5},
    ]


def test_invalid_utf8_fails_as_unsupported_encoding() -> None:
    """捕获坏字节被忽略或替换为乱码伪装成功。"""
    with pytest.raises(AttachmentParseError) as error:
        _parse(b"\x2d\x4e\x87\x65")  # 无 BOM 的 UTF-16LE "中文"，按 UTF-8 非法
    assert (error.value.category, error.value.params["reason"]) == (
        CONTENT_UNREADABLE,
        "unsupported_encoding",
    )


def test_utf32_bom_is_explicitly_rejected() -> None:
    """捕获 UTF-32 LE BOM 被误认成 UTF-16。"""
    for bom in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        with pytest.raises(AttachmentParseError) as error:
            _parse(bom + "x".encode("utf-32-le"))
        assert error.value.params["reason"] == "unsupported_encoding"


def test_control_characters_fail_with_stable_category() -> None:
    """捕获 NUL/DEL/C1 控制符进入正文；TAB 与换页符被保留。"""
    for bad in (b"a\x00b\n", b"a\x7fb\n", "a\x85b\n".encode()):
        with pytest.raises(AttachmentParseError) as error:
            _parse(bad)
        assert (error.value.category, error.value.params["reason"]) == (
            CONTENT_UNREADABLE,
            "invalid_text",
        )

    kept = _parse(b"col1\tcol2\x0cpage2\n")
    assert kept.text == "col1\tcol2\x0cpage2\n"


def test_bom_or_whitespace_only_content_fails_as_empty() -> None:
    """捕获 BOM/纯空白内容伪装成空正文成功。"""
    for raw in (b"\xef\xbb\xbf", b"  \n\t \n"):
        with pytest.raises(AttachmentParseError) as error:
            _parse(raw)
        assert (error.value.category, error.value.params["reason"]) == (
            CONTENT_UNREADABLE,
            "empty_content",
        )


def test_markdown_syntax_is_preserved_verbatim() -> None:
    """捕获 Markdown 被渲染、重排或丢失语法/空白。"""
    source = (
        "---\n"
        "title: 样例\n"
        "---\n"
        "\n"
        "# 标题\n"
        "\n"
        "    indented code\n"
        "```py\ncode fence  \n```\n"
        "| a | b |\n"
        "| --- | --- |\n"
        "[link](https://example.com) <b>html</b> trailing  \n"
    )
    result = _parse(source.encode("utf-8"), content_format="markdown")

    assert result.text == source
    assert result.content_format == "markdown"
    assert result.content_object["locators"][0] == {
        "kind": "line",
        "number": 1,
        "start": 0,
        "end": 3,
    }


def test_raw_input_over_limit_fails_before_decoding() -> None:
    """捕获 RAW 输入上限未在 W1-B 复核。"""
    with pytest.raises(AttachmentParseError) as error:
        _parse(b"x" * 11, max_raw_bytes=10)
    assert (error.value.category, error.value.params["reason"]) == (
        RESOURCE_LIMIT,
        "raw_input_limit",
    )


def test_parse_budget_is_enforced_with_injectable_clock() -> None:
    """捕获协作式预算检查点失效或测试依赖真实等待。"""

    class _SteppingClock:
        """每次调用推进固定步长，模拟解析耗时推进。"""

        def __init__(self, step: float) -> None:
            self.step = step
            self.now = 0.0

        def __call__(self) -> float:
            self.now += self.step
            return self.now

    parser = TextAttachmentParser(content_format="plain_text")
    limits = ParseLimits(parse_budget_seconds=5.0)

    # 预算内正常完成。
    result = parser.parse(
        b"ok\n",
        limits=limits,
        source_raw_revision=1,
        source_raw_hash="h",
        clock=_SteppingClock(step=1.0),
    )
    assert result.text == "ok\n"

    # 检查点跨过预算后稳定失败。
    with pytest.raises(AttachmentParseError) as error:
        parser.parse(
            b"late\n",
            limits=limits,
            source_raw_revision=1,
            source_raw_hash="h",
            clock=_SteppingClock(step=10.0),
        )
    assert (error.value.category, error.value.params["reason"]) == (
        RESOURCE_LIMIT,
        "parse_budget_exceeded",
    )
