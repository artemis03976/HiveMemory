"""MTP runtime i18n 文本 helper 测试。"""

import pytest

from hivememory.i18n.mtp_runtime import (
    get_mtp_error_text,
    get_mtp_info_text,
    get_mtp_warning_text,
)
from hivememory.i18n.syscall_runtime import (
    get_syscall_error_text,
    get_syscall_info_text,
)
from hivememory.core.mtp.exceptions import StorageOfflineError, SyscallInternalError, SystemFault


def test_get_mtp_warning_text_en():
    text = get_mtp_warning_text(
        "mtp.filter.unknown_key",
        {"key": "unknown"},
        "en",
    )

    assert text == "Note: Unknown filter key 'unknown' was ignored."


def test_get_mtp_info_text_execution_result_title():
    """普通 MTP 执行结果标题应从 info 文本表读取。"""
    assert (
        get_mtp_info_text("mtp.loop.execution_result_title", language="en")
        == "[System MTP Execution Result]"
    )


def test_get_mtp_info_text_write_update_ack():
    """WRITE / UPDATE ACK 属于成功信息，应从 info 文本表读取。"""
    write_text = get_mtp_info_text(
        "mtp.write.ack",
        {"pending_alias": "draft_note_1"},
        "en",
    )
    update_text = get_mtp_info_text(
        "mtp.update.ack",
        {"base_alias": "fact_note", "pending_alias": "rev_note_1"},
        "en",
    )

    assert "pending atom 'draft_note_1'" in write_text
    assert "pending revision 'rev_note_1'" in update_text


def test_write_update_ack_not_in_error_text():
    """WRITE / UPDATE ACK 不应再从 error 文本表读取，避免成功信息被错误归类。"""
    with pytest.raises(KeyError):
        get_mtp_error_text("mtp.write.ack", {"pending_alias": "draft_note_1"}, "en")
    with pytest.raises(KeyError):
        get_mtp_error_text(
            "mtp.update.ack",
            {"base_alias": "fact_note", "pending_alias": "rev_note_1"},
            "en",
        )


def test_get_mtp_warning_text_read_partial_alias_not_found():
    """READ 部分 alias 未解析片段属于 nonfatal warning，应从 warning 文本表读取。"""
    text = get_mtp_warning_text(
        "mtp.read.partial_alias_not_found",
        {"alias": "missing_alias"},
        "zh",
    )

    assert "[missing_alias]: [Alias Not Found]" in text
    assert "未找到" in text


def test_get_mtp_info_text_call_response_en():
    """CALL response 英文 info 文本应覆盖标题、reply 与 artifact 标签。"""
    assert (
        get_mtp_info_text("mtp.call_response.title", language="en")
        == "[System MTP Call Response]"
    )
    assert (
        get_mtp_info_text("mtp.call_response.reply_label", language="en")
        == "[Sub-Agent Reply]:"
    )
    assert (
        get_mtp_info_text("mtp.call_response.artifacts_label", language="en")
        == "[Artifacts Generated / Updated]:"
    )
    assert (
        get_mtp_info_text("mtp.call_response.artifact_state", language="en")
        == "(pending, readable now)"
    )


def test_get_mtp_info_text_call_response_zh():
    """CALL response 中文 info 文本应覆盖标题和 pending 状态说明。"""
    assert (
        get_mtp_info_text("mtp.call_response.title", language="zh")
        == "[System MTP Call Response]"
    )
    assert (
        get_mtp_info_text("mtp.call_response.artifact_state", language="zh")
        == "(pending, 本次运行可读)"
    )


def test_get_mtp_error_text_call_response_sub_agent_error():
    """子代理异常应通过 call_response error key 渲染。"""
    text = get_mtp_error_text(
        "mtp.call_response.sub_agent_error",
        {"agent_alias": "coder_doll"},
        "en",
    )

    assert "[Sub-Agent Error]" in text
    assert "coder_doll" in text


def test_get_mtp_warning_text_zh():
    text = get_mtp_warning_text(
        "mtp.filter.unknown_key",
        {"key": "unknown"},
        "zh",
    )

    assert "未知 filter key 'unknown'" in text


def test_system_fault_defaults_to_structured_message_key():
    error = SystemFault()
    info = error.to_error_info()

    assert info.code == "mtp.system.fault"
    assert info.message_key == "mtp.system.unexpected_error"


def test_storage_offline_defaults_to_structured_message_key():
    error = StorageOfflineError()
    info = error.to_error_info()

    assert info.code == "mtp.system.storage_offline"
    assert info.message_key == "mtp.system.storage_offline"


def test_syscall_internal_error_uses_i18n_template():
    error = SyscallInternalError(params={"alias": "sys_tool", "detail": "boom"})
    info = error.to_error_info()

    assert "Tool 'sys_tool'" in get_mtp_error_text(info.message_key, info.params, "en")
    assert info.message_key == "mtp.system.tool_error"


def test_get_syscall_error_text_en():
    """syscall 错误文本使用独立 namespace，不再依赖 mtp.* 文本表。"""
    text = get_syscall_error_text(
        "syscall.file_read.missing_path",
        {"arg": "path"},
        "en",
    )

    assert 'file_read requires a "path" argument' in text


def test_get_syscall_error_text_zh():
    """syscall 错误文本应支持中文渲染。"""
    text = get_syscall_error_text(
        "syscall.repl.timeout",
        {"timeout_seconds": 1},
        "zh",
    )

    assert "Python 执行在 1s 后超时" in text


def test_get_syscall_info_text_en():
    """syscall 成功提示从 info 文本表读取。"""
    text = get_syscall_info_text(
        "syscall.file_write.success",
        {"name": "note.txt", "bytes": 12},
        "en",
    )

    assert text == "Success: File 'note.txt' saved (12 bytes)."


def test_get_syscall_text_missing_key_and_param_raise_key_error():
    """缺失 key 或 params 应继续暴露 KeyError，避免空错误响应。"""
    with pytest.raises(KeyError):
        get_syscall_error_text("syscall.missing", {}, "en")

    with pytest.raises(KeyError):
        get_syscall_info_text("syscall.file_write.success", {"name": "note.txt"}, "en")
