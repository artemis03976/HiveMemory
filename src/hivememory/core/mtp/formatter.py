"""MTP response formatter."""

from __future__ import annotations

from hivememory.core.mtp.models import MTPResponse
from hivememory.i18n.mtp_runtime import (
    get_mtp_error_text,
    get_mtp_info_text,
    get_mtp_warning_text,
)
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

        parts: list[str] = [
            f'<mtp_response status="{response.status.value}"{time_attr}>'
        ]
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
        error = response.error
        text = get_mtp_error_text(error.message_key, error.params, language)
        return "\n".join(
            [
                f'<error code="{error.code}" severity="{error.severity.value}">',
                text,
                "</error>",
            ]
        )

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
