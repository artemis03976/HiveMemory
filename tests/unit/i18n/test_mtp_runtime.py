"""Tests for MTP runtime i18n text helpers."""

from hivememory.i18n.mtp_runtime import get_mtp_warning_text
from hivememory.core.mtp.exceptions import StorageOfflineError, SyscallInternalError, SystemFault


def test_get_mtp_warning_text_en():
    text = get_mtp_warning_text(
        "mtp.filter.unknown_key",
        {"key": "unknown"},
        "en",
    )

    assert text == "Note: Unknown filter key 'unknown' was ignored."


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

    assert "Tool 'sys_tool'" in error.to_agent_prompt("en")
    assert error.to_error_info().message_key == "mtp.system.tool_error"
