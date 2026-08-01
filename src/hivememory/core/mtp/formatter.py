"""MTP response formatter."""

from __future__ import annotations

from hivememory.core.mtp.models import MTPCallResponse, MTPResponse, MTPResponseStatus
from hivememory.i18n.mtp_runtime import (
    get_mtp_error_text,
    get_mtp_info_text,
    get_mtp_warning_text,
)
from hivememory.i18n.syscall_runtime import get_syscall_error_text
from hivememory.i18n.types import Language


class MTPFormatter:
    """Format structured MTP responses into the agent-facing XML container."""

    @staticmethod
    def format_response(
        response: MTPResponse,
        language: str | Language | None = None,
    ) -> str:
        """Format a response body for agent-facing MTP execution backfill."""
        response_xml = MTPFormatter._format_response_xml(response, language)
        return MTPFormatter._format_execution_result(response_xml, language)

    @staticmethod
    def format_call_response(
        call_response: MTPCallResponse,
        language: str | Language | None = None,
    ) -> str:
        """Format a CALL response for agent-facing backfill."""
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
            time_attr = f' time="{response.execution_time_ms:.0f}ms"'

        parts: list[str] = [f'<mtp_response status="{response.status.value}"{time_attr}>']
        if response.content:
            parts.append(response.content)
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
        return "\n".join(
            [
                f'<error code="{error.code}" severity="{error.severity.value}">',
                text,
                "</error>",
            ]
        )

    @staticmethod
    def _format_call_response_xml(
        call_response: MTPCallResponse,
        language: str | Language | None = None,
    ) -> str:
        lines = [f'<mtp_response status="{call_response.status.value}" type="call_response">']
        if call_response.status == MTPResponseStatus.ERROR:
            if call_response.error is not None:
                lines.append(MTPFormatter._format_error_info(call_response.error, language))
        elif call_response.status == MTPResponseStatus.CANCELLED:
            lines.append(get_mtp_info_text("mtp.call_response.cancelled", language=language))
        else:
            lines.append(get_mtp_info_text("mtp.call_response.reply_label", language=language))
            lines.append(call_response.reply or "")
            if call_response.artifact_aliases:
                lines.append("")
                lines.append(
                    get_mtp_info_text("mtp.call_response.artifacts_label", language=language)
                )
                state = get_mtp_info_text("mtp.call_response.artifact_state", language=language)
                for alias in call_response.artifact_aliases:
                    lines.append(f"- {alias} {state}")
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
            lines.append(f"<warning>{text}</warning>")
        lines.append("</warnings>")
        return "\n".join(lines)


__all__ = ["MTPFormatter"]
