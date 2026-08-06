"""MTP 响应格式化器。"""

from __future__ import annotations

from xml.sax.saxutils import escape

from hivememory.core.mtp.models import MTPCallResponse, MTPResponse, MTPResponseStatus
from hivememory.i18n.mtp_runtime import (
    get_mtp_error_text,
    get_mtp_info_text,
    get_mtp_warning_text,
)
from hivememory.i18n.syscall_runtime import get_syscall_error_text
from hivememory.i18n.types import Language


def _is_valid_xml_char(char: str) -> bool:
    """判断字符是否属于 XML 1.0 允许范围。"""
    code_point = ord(char)
    return (
        code_point in {0x09, 0x0A, 0x0D}
        or 0x20 <= code_point <= 0xD7FF
        or 0xE000 <= code_point <= 0xFFFD
        or 0x10000 <= code_point <= 0x10FFFF
    )


def _normalize_xml_text(value: str) -> str:
    """统一换行，并替换 XML 1.0 禁止的字符。"""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(char if _is_valid_xml_char(char) else "\ufffd" for char in normalized)


def _format_xml_text(value: str) -> str:
    """将原始业务文本安全放入 XML 文本节点。"""
    normalized = _normalize_xml_text(value)
    if not any(char in "<>&" for char in normalized):
        return normalized
    cdata = normalized.replace("]]>", "]]]]><![CDATA[>")
    return f"<![CDATA[{cdata}]]>"


def _format_xml_attribute(value: str) -> str:
    """转义 XML 属性值并保留其原始语义。"""
    normalized = _normalize_xml_text(value)
    return escape(
        normalized,
        {
            '"': "&quot;",
            "'": "&apos;",
            "\n": "&#10;",
            "\t": "&#9;",
        },
    )


class MTPFormatter:
    """将结构化 MTP 响应格式化为 Agent 可见的 XML 容器。"""

    @staticmethod
    def format_response(
        response: MTPResponse,
        language: str | Language | None = None,
    ) -> str:
        """格式化普通 MTP 执行回填。"""
        response_xml = MTPFormatter._format_response_xml(response, language)
        return MTPFormatter._format_execution_result(response_xml, language)

    @staticmethod
    def format_call_response(
        call_response: MTPCallResponse,
        language: str | Language | None = None,
    ) -> str:
        """格式化 CALL 响应回填。"""
        response_xml = MTPFormatter._format_call_response_xml(call_response, language)
        title = get_mtp_info_text("mtp.call_response.title", language=language)
        return f"{title}\n{response_xml}"

    @staticmethod
    def _format_execution_result(
        content: str,
        language: str | Language | None = None,
    ) -> str:
        title = get_mtp_info_text("mtp.loop.execution_result_title", language=language)
        return f"{title}\n{content}"

    @staticmethod
    def _format_response_xml(
        response: MTPResponse,
        language: str | Language | None = None,
    ) -> str:
        time_attr = ""
        if response.execution_time_ms > 0:
            time_value = _format_xml_attribute(f"{response.execution_time_ms:.0f}ms")
            time_attr = f' time="{time_value}"'

        status = _format_xml_attribute(response.status.value)
        parts: list[str] = [f'<mtp_response status="{status}"{time_attr}>']
        if response.content:
            parts.append(_format_xml_text(response.content))
        if response.error is not None:
            parts.append(MTPFormatter._format_error(response, language))
        if response.warnings:
            parts.append(MTPFormatter._format_warnings(response, language))
        parts.append("</mtp_response>")
        return "\n".join(parts)

    @staticmethod
    def _format_error(
        response: MTPResponse,
        language: str | Language | None = None,
    ) -> str:
        if response.error is None:
            return ""
        return MTPFormatter._format_error_info(response.error, language)

    @staticmethod
    def _format_error_info(error, language: str | Language | None = None) -> str:
        if error.message_key.startswith("syscall."):
            text = get_syscall_error_text(error.message_key, error.params, language)
        else:
            text = get_mtp_error_text(error.message_key, error.params, language)
        code = _format_xml_attribute(error.code)
        severity = _format_xml_attribute(error.severity.value)
        return "\n".join(
            [
                f'<error code="{code}" severity="{severity}">',
                _format_xml_text(text),
                "</error>",
            ]
        )

    @staticmethod
    def _format_call_response_xml(
        call_response: MTPCallResponse,
        language: str | Language | None = None,
    ) -> str:
        status = _format_xml_attribute(call_response.status.value)
        response_type = _format_xml_attribute("call_response")
        lines = [f'<mtp_response status="{status}" type="{response_type}">']
        if call_response.status == MTPResponseStatus.ERROR:
            if call_response.error is not None:
                lines.append(MTPFormatter._format_error_info(call_response.error, language))
        elif call_response.status == MTPResponseStatus.CANCELLED:
            lines.append(
                _format_xml_text(
                    get_mtp_info_text("mtp.call_response.cancelled", language=language)
                )
            )
        else:
            lines.append(
                _format_xml_text(
                    get_mtp_info_text("mtp.call_response.reply_label", language=language)
                )
            )
            lines.append(_format_xml_text(call_response.reply or ""))
            if call_response.artifact_aliases:
                lines.append("")
                lines.append(
                    _format_xml_text(
                        get_mtp_info_text(
                            "mtp.call_response.artifacts_label",
                            language=language,
                        )
                    )
                )
                state = get_mtp_info_text("mtp.call_response.artifact_state", language=language)
                for alias in call_response.artifact_aliases:
                    lines.append(_format_xml_text(f"- {alias} {state}"))
        lines.append("</mtp_response>")
        return "\n".join(lines)

    @staticmethod
    def _format_warnings(
        response: MTPResponse,
        language: str | Language | None = None,
    ) -> str:
        lines = ["<warnings>"]
        for warning in response.warnings:
            text = get_mtp_warning_text(warning.message_key, warning.params, language)
            lines.append(f"<warning>{_format_xml_text(text)}</warning>")
        lines.append("</warnings>")
        return "\n".join(lines)


__all__ = ["MTPFormatter"]
