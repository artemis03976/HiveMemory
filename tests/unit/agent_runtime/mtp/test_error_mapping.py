"""中立存储失败经 MTP 执行入口保留协议类别与安全呈现。"""

import pytest

from hivememory.agent_runtime.frame import FrameExecutionPolicy
from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.mtp.error_mapping import mtp_error_info
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.core.errors import (
    OperationDeniedError,
    ResourceNotFoundError,
    ResourceUnavailableError,
    StorageOfflineError,
    StorageReadError,
    StorageWriteError,
    ThreadContextRevokedError,
)
from hivememory.core.mtp.exceptions import InvalidArgumentError, SyscallExecutionError
from hivememory.core.mtp.models import MTPErrorSeverity
from hivememory.i18n.mtp_runtime import get_mtp_error_text
from hivememory.i18n.syscall_runtime import get_syscall_error_text
from hivememory.workspace.contracts.errors import (
    IntentDispatchRejectedError,
    IntentDispatchUnknownError,
    PendingUpdateNotAllowedError,
)
from tests.helpers.workspace import make_runtime_scope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (StorageOfflineError, "mtp.system.storage_offline"),
        (StorageReadError, "mtp.system.storage_error"),
        (StorageWriteError, "mtp.system.storage_write_error"),
    ],
)
async def test_neutral_storage_failure_keeps_mtp_wire_category(error_type, code):
    """不同存储失败仍呈现既有系统错误，内部操作、地址和 cause 不交付给模型。"""

    async def fail(request):
        raise error_type(details={"operation": "internal-operation"}) from RuntimeError(
            "secret-host"
        )

    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        execution_policy=FrameExecutionPolicy.from_verbs(["READ"]),
        submit_operation=fail,
        language="zh",
    )
    result = await KoakumaRuntime().execute_mtp("⟪ READ | known_alias ⟫", context)

    assert result.response_status == "error"
    assert f'code="{code}"' in result.formatted_response
    assert 'severity="system_fault"' in result.formatted_response
    assert "internal-operation" not in result.formatted_response
    assert "secret-host" not in result.formatted_response


@pytest.mark.parametrize(
    ("error", "verb", "alias", "code", "key", "params"),
    [
        (
            OperationDeniedError(details={"operation": "secret"}),
            "WRITE",
            "",
            "mtp.permission.denied",
            "mtp.permission.verb_denied",
            {"verb": "WRITE"},
        ),
        (
            ResourceUnavailableError("secret"),
            "READ",
            "a",
            "mtp.system.service_unavailable",
            "mtp.system.service_unavailable",
            {},
        ),
        (
            PendingUpdateNotAllowedError(details={"alias": "secret"}),
            "UPDATE",
            "a",
            "mtp.argument.invalid",
            "mtp.update.pending_not_updatable",
            {"alias": "a"},
        ),
        (
            ResourceNotFoundError("secret"),
            "UPDATE",
            "a",
            "mtp.alias.not_found",
            "mtp.update.alias_not_found",
            {"alias": "a"},
        ),
        (
            IntentDispatchRejectedError(details={"pending_alias": "p1", "reason": "secret"}),
            "WRITE",
            "",
            "mtp.memory.intent_rejected",
            "mtp.memory.intent_rejected",
            {"pending_alias": "p1"},
        ),
        (
            RuntimeError("secret-token@internal-host"),
            "READ",
            "a",
            "mtp.system.fault",
            "mtp.system.unexpected_error",
            {},
        ),
        (
            InvalidArgumentError(
                message_key="mtp.run.missing_single_target", params={"token": "secret"}
            ),
            "RUN",
            "",
            "mtp.argument.invalid",
            "mtp.run.missing_single_target",
            {},
        ),
    ],
)
def test_mapping_keeps_categories_and_whitelists_public_parameters(
    error, verb, alias, code, key, params
):
    """中立异常不透传 details，未知异常仍是固定系统故障。"""
    info = mtp_error_info(error, verb=verb, alias=alias)
    assert info.code == code
    assert info.message_key == key
    assert info.params == params
    assert "secret" not in info.model_dump_json()


@pytest.mark.parametrize(
    ("can_read", "language", "expected"),
    [
        (True, "zh", "请用 READ 回读原句柄"),
        (False, "zh", "当前没有 READ 能力，无法核实结果"),
        (True, "en", "READ the existing handle"),
        (False, "en", "READ is unavailable, so you cannot verify the result"),
    ],
)
def test_unknown_intent_recommends_verification_only_when_read_is_allowed(
    can_read, language, expected
):
    """不能把接纳未知误报为可安全重提，也不能建议调用未获准的 READ。"""
    info = mtp_error_info(
        IntentDispatchUnknownError(details={"pending_alias": "p1", "reason": "secret"}),
        can_read=can_read,
    )
    text = get_mtp_error_text(info.message_key, info.params, language)
    assert info.code == "mtp.memory.intent_unknown"
    assert info.severity == MTPErrorSeverity.SYSTEM_FAULT
    assert info.params == {"pending_alias": "p1"}
    assert expected in text
    assert (
        "不要重复提交 WRITE/UPDATE" if language == "zh" else "do not submit another WRITE/UPDATE"
    ) in text


@pytest.mark.parametrize(
    ("language", "expected"), [("zh", "原线程已结束"), ("en", "The original thread has ended")]
)
def test_revoked_thread_uses_stop_message_with_existing_system_code(language, expected):
    info = mtp_error_info(ThreadContextRevokedError("secret"))
    assert (info.code, info.severity) == ("mtp.system.fault", MTPErrorSeverity.SYSTEM_FAULT)
    assert expected in get_mtp_error_text(info.message_key, info.params, language)
    assert info.params == {}


def test_tool_internal_detail_is_not_exposed_even_when_attached_to_protocol_error():
    info = mtp_error_info(
        SyscallExecutionError(
            message_key="syscall.repl.execution_failed", params={"detail": "secret"}
        )
    )
    assert info.params == {}
    assert "secret" not in get_syscall_error_text(info.message_key, info.params, "zh")


@pytest.mark.asyncio
async def test_execution_cancellation_is_not_converted_to_protocol_failure():
    """调用方取消必须离开执行边界，不能被统一错误出口变成普通失败。"""
    import asyncio

    async def cancelled(request):
        raise asyncio.CancelledError()

    context = MTPExecutionContext(
        runtime_scope=make_runtime_scope(),
        execution_policy=FrameExecutionPolicy.from_verbs(["READ"]),
        submit_operation=cancelled,
        language="zh",
    )
    with pytest.raises(asyncio.CancelledError):
        await KoakumaRuntime().execute_mtp("⟪ READ | known_alias ⟫", context)
