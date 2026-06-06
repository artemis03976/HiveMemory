"""
MTP response formatter。

负责将内核执行结果统一格式化为 `<mtp_response>` XML 容器，
以便注入 Fake User Message 并继续递归生成。
"""

from hivememory.core.mtp.models import MTPCommand, MTPResponse


class MTPFormatter:
    """MTP 响应格式化器。"""

    @staticmethod
    def format_response(response: MTPResponse) -> str:
        """仅格式化响应体为 XML 容器。"""
        time_attr = ""
        if response.execution_time_ms > 0:
            time_attr = f' time="{response.execution_time_ms:.0f}ms"'

        # TODO: Render response.warnings in a dedicated XML section once the
        # warning prompt contract is finalized.
        return (
            f'<mtp_response status="{response.status.value}"{time_attr}>\n'
            f"{response.content}\n"
            f"</mtp_response>"
        )

    @staticmethod
    def format_command_with_response(command: MTPCommand, response: MTPResponse) -> str:
        """
        格式化 `指令 + 响应` 文本块。

        用于回填历史消息，形态如下:
            ⟪ VERB | TARGET | ARGS ⟫
            <mtp_response ...>...</mtp_response>
        """
        time_attr = ""
        if response.execution_time_ms > 0:
            time_attr = f' time="{response.execution_time_ms:.0f}ms"'

        # TODO: Render response.warnings in a dedicated XML section once the
        # warning prompt contract is finalized.
        response_xml = (
            f'<mtp_response status="{response.status.value}"{time_attr}>\n'
            f"{response.content}\n"
            f"</mtp_response>"
        )
        return f"{command.raw_text}\n{response_xml}"


__all__ = ["MTPFormatter"]
