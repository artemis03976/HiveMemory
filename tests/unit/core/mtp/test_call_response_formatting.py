"""MTP CALL response payload 渲染测试。"""

from hivememory.core.mtp import (
    MTPCallResponse,
    MTPErrorInfo,
    MTPErrorSeverity,
    MTPFormatter,
    MTPResponseStatus,
)


class TestMTPCallResponseFormatting:
    """CALL response payload 渲染测试"""

    def test_assemble_success_no_artifacts(self):
        """成功返回，无 artifacts"""
        payload = MTPFormatter.format_call_response(
            MTPCallResponse(
                status=MTPResponseStatus.SUCCESS,
                agent_alias="coder_doll",
                reply="Task completed successfully.",
            ),
            "en",
        )

        assert payload.startswith("[System MTP Call Response]\n")
        assert '<mtp_response status="success" type="call_response">' in payload
        assert "[Sub-Agent Reply]:" in payload
        assert "Task completed successfully." in payload
        assert "[Artifacts" not in payload
        assert "</mtp_response>" in payload

    def test_assemble_success_with_artifacts(self):
        """成功返回，含 artifacts"""
        payload = MTPFormatter.format_call_response(
            MTPCallResponse(
                status=MTPResponseStatus.SUCCESS,
                agent_alias="coder_doll",
                reply="Code written.",
                artifact_aliases=["mem_code_1", "mem_code_2"],
            ),
            "en",
        )

        assert "[Artifacts Generated / Updated]:" in payload
        assert "- mem_code_1 (pending, readable now)" in payload
        assert "- mem_code_2 (pending, readable now)" in payload

    def test_assemble_error(self):
        """错误返回使用结构化 error 渲染"""
        payload = MTPFormatter.format_call_response(
            MTPCallResponse(
                status=MTPResponseStatus.ERROR,
                agent_alias="coder_doll",
                error=MTPErrorInfo(
                    code="mtp.call_response.sub_agent_error",
                    message_key="mtp.call_response.sub_agent_error",
                    severity=MTPErrorSeverity.SYSTEM_FAULT,
                    params={"agent_alias": "coder_doll"},
                ),
            ),
            "en",
        )

        assert '<mtp_response status="error" type="call_response">' in payload
        assert '<error code="mtp.call_response.sub_agent_error" severity="system_fault">' in payload
        assert "[Sub-Agent Error]" in payload
        assert "coder_doll" in payload
