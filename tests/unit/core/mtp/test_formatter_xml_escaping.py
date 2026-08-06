"""MTP formatter 的 XML escaping 协议测试。"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from hivememory.core.mtp import (
    MTPCallResponse,
    MTPErrorInfo,
    MTPErrorSeverity,
    MTPFormatter,
    MTPResponse,
    MTPResponseStatus,
    MTPWarningInfo,
)
from hivememory.i18n.mtp_runtime import get_mtp_error_text, get_mtp_warning_text


def _extract_xml_block(formatted: str) -> str:
    """从 Agent 回填文本中提取可独立解析的 XML 块。"""
    return formatted[formatted.index("<mtp_response") :]


def _parse_xml_block(formatted: str) -> tuple[ET.Element, str]:
    """严格解析 formatter 生成的 MTP XML 块。"""
    xml_block = _extract_xml_block(formatted)
    return ET.fromstring(xml_block), xml_block


@pytest.mark.parametrize(
    "content",
    [
        "if a < b && c > d; quotes=\"value\"/'value'",
        "already encoded: &lt;tag&gt; &amp;",
        "before ]]> after",
        '<content><nested attr="a&b">值</nested></content>',
    ],
)
def test_content_with_reserved_characters_round_trips_as_text(content: str):
    response = MTPResponse(status=MTPResponseStatus.SUCCESS, content=content)

    root, xml_block = _parse_xml_block(MTPFormatter.format_response(response, "en"))

    assert root.text == f"\n{content}\n"
    assert list(root) == []
    assert "<![CDATA[" in xml_block


def test_preencoded_entities_are_not_escaped_again():
    content = "already encoded: &lt;tag&gt; &amp;"

    root, xml_block = _parse_xml_block(
        MTPFormatter.format_response(
            MTPResponse(status=MTPResponseStatus.SUCCESS, content=content),
            "en",
        )
    )

    assert "&lt;tag&gt; &amp;" in xml_block
    assert "&amp;lt;" not in xml_block
    assert root.text == f"\n{content}\n"


def test_unicode_and_long_content_round_trip():
    content = "中文🙂e\u0301<&>" + "x" * 100_000

    root, _ = _parse_xml_block(
        MTPFormatter.format_response(
            MTPResponse(status=MTPResponseStatus.SUCCESS, content=content),
            "zh",
        )
    )

    assert root.text == f"\n{content}\n"


def test_newlines_and_invalid_xml_characters_are_normalized():
    content = "first\r\nsecond\rthird\n\x00\x01\x0b\ud800"
    expected = "first\nsecond\nthird\n\ufffd\ufffd\ufffd\ufffd"

    root, xml_block = _parse_xml_block(
        MTPFormatter.format_response(
            MTPResponse(status=MTPResponseStatus.SUCCESS, content=content),
            "en",
        )
    )

    assert "\r" not in xml_block
    assert root.text == f"\n{expected}\n"


def test_empty_content_keeps_existing_wire_shape():
    root, xml_block = _parse_xml_block(
        MTPFormatter.format_response(
            MTPResponse(status=MTPResponseStatus.SUCCESS, content=""),
            "en",
        )
    )

    assert xml_block == '<mtp_response status="success">\n</mtp_response>'
    assert root.text == "\n"
    assert list(root) == []


def test_safe_content_keeps_existing_plain_text_shape():
    content = "plain response"

    root, xml_block = _parse_xml_block(
        MTPFormatter.format_response(
            MTPResponse(status=MTPResponseStatus.SUCCESS, content=content),
            "en",
        )
    )

    assert "<![CDATA[" not in xml_block
    assert root.text == f"\n{content}\n"


def test_error_reason_and_attributes_are_xml_safe():
    code = "mtp.bad\"'<>&\t\r\x00"
    normalized_code = "mtp.bad\"'<>&\t\n\ufffd"
    params = {"alias": "bad<&alias"}
    response = MTPResponse(
        status=MTPResponseStatus.ERROR,
        error=MTPErrorInfo(
            code=code,
            message_key="mtp.run.alias_not_found",
            severity=MTPErrorSeverity.AGENT_FAULT,
            params=params,
        ),
    )

    root, _ = _parse_xml_block(MTPFormatter.format_response(response, "en"))
    error = root.find("error")

    assert error is not None
    assert error.attrib == {
        "code": normalized_code,
        "severity": "agent_fault",
    }
    expected_reason = get_mtp_error_text("mtp.run.alias_not_found", params, "en")
    assert error.text == f"\n{expected_reason}\n"


def test_warning_text_is_xml_safe():
    params = {"key": "bad<&key"}
    response = MTPResponse(
        status=MTPResponseStatus.SUCCESS,
        warnings=[
            MTPWarningInfo(
                message_key="mtp.filter.unknown_key",
                params=params,
            )
        ],
    )

    root, _ = _parse_xml_block(MTPFormatter.format_response(response, "en"))
    warning = root.find("./warnings/warning")

    assert warning is not None
    assert warning.text == get_mtp_warning_text("mtp.filter.unknown_key", params, "en")


def test_call_reply_and_artifact_aliases_are_xml_safe():
    reply = "reply <ok> & &lt;encoded&gt; ]]> 中文"
    artifact_alias = 'mem<&"alias'
    response = MTPCallResponse(
        status=MTPResponseStatus.SUCCESS,
        agent_alias="coder_doll",
        reply=reply,
        artifact_aliases=[artifact_alias],
    )

    root, _ = _parse_xml_block(MTPFormatter.format_call_response(response, "en"))

    assert list(root) == []
    assert root.text is not None
    assert reply in root.text
    assert artifact_alias in root.text


def test_empty_call_reply_keeps_label_and_empty_body():
    response = MTPCallResponse(
        status=MTPResponseStatus.SUCCESS,
        agent_alias="coder_doll",
        reply="",
    )

    root, _ = _parse_xml_block(MTPFormatter.format_call_response(response, "en"))

    assert root.text is not None
    assert "[Sub-Agent Reply]:" in root.text
    assert list(root) == []
