from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from hivememory.agent_runtime import FrameExecutionResult, FrameExecutionStatus, mtp_error_info
from hivememory.core.errors import (
    AdmissionDeniedError,
    ModelNotFoundError,
    OperationDeniedError,
    ResourceNotFoundError,
    ResourceUnavailableError,
    StorageOfflineError,
    StorageReadError,
    StorageWriteError,
    ThreadEndedError,
)
from hivememory.core.mtp.exceptions import MTPError
from hivememory.core.mtp.models import (
    MTPCallResponse,
    MTPErrorInfo,
    MTPErrorSeverity,
    MTPResponseStatus,
)
from hivememory.patchouli.contracts.errors import ProfileResolutionError

if TYPE_CHECKING:
    from hivememory.core.models import AgentProfile


def success_response(
    agent_alias: str,
    *,
    reply: str = "",
    artifact_aliases: Sequence[str] = (),
) -> MTPCallResponse:
    return MTPCallResponse(
        status=MTPResponseStatus.SUCCESS,
        agent_alias=agent_alias,
        reply=reply,
        artifact_aliases=list(artifact_aliases),
    )


def cancelled_response(agent_alias: str) -> MTPCallResponse:
    return MTPCallResponse(
        status=MTPResponseStatus.CANCELLED,
        agent_alias=agent_alias,
    )


def error_response(agent_alias: str, error: MTPErrorInfo) -> MTPCallResponse:
    return MTPCallResponse(
        status=MTPResponseStatus.ERROR,
        agent_alias=agent_alias,
        error=error,
    )


def sub_agent_error_info(agent_alias: str, error: Exception | None = None) -> MTPErrorInfo:
    """子帧失败只向调用方透露任务未完成，内部根因不参与模板参数。"""
    return MTPErrorInfo(
        code="mtp.call_response.sub_agent_error",
        message_key="mtp.call_response.sub_agent_error",
        severity=MTPErrorSeverity.SYSTEM_FAULT,
        params={"agent_alias": agent_alias},
        cause=str(error) if error is not None else None,
    )


def sub_agent_budget_exhausted_info(agent_alias: str) -> MTPErrorInfo:
    """预算终态直接形成响应，不构造不会被抛出的异常。"""
    return MTPErrorInfo(
        code="mtp.call_response.budget_exhausted",
        message_key="mtp.call_response.budget_exhausted",
        severity=MTPErrorSeverity.SYSTEM_FAULT,
        params={"agent_alias": agent_alias},
    )


def agent_model_unavailable_info(agent_alias: str, model_name: str) -> MTPErrorInfo:
    """只公开配置中声明的模型名称，不包含 Provider 诊断或凭据。"""
    return MTPErrorInfo(
        code="mtp.system.service_unavailable",
        message_key="mtp.call_response.model_unavailable",
        severity=MTPErrorSeverity.SYSTEM_FAULT,
        params={"agent_alias": agent_alias, "model_name": model_name},
    )


def preparation_error_response(
    agent_alias: str,
    error: Exception,
    *,
    stage: Literal["context", "frame"] = "frame",
) -> MTPCallResponse:
    """CALL 准备失败的唯一映射；上下文解析与子帧装配保留各自协议类别。"""
    if isinstance(
        error,
        (
            StorageOfflineError,
            StorageReadError,
            StorageWriteError,
            ThreadEndedError,
            OperationDeniedError,
            ResourceUnavailableError,
            MTPError,
        ),
    ):
        return error_response(agent_alias, mtp_error_info(error, verb="CALL"))
    params = {"agent_alias": agent_alias}
    code, key, severity = (
        "mtp.system.fault",
        "mtp.call.profile_load_failed",
        MTPErrorSeverity.SYSTEM_FAULT,
    )
    if isinstance(error, AdmissionDeniedError):
        code, key, severity = (
            "mtp.permission.denied",
            "mtp.call.agent_unavailable",
            MTPErrorSeverity.AGENT_FAULT,
        )
    elif isinstance(error, ResourceNotFoundError):
        code, key, severity = (
            "mtp.alias.not_found",
            "mtp.call.profile_not_found",
            MTPErrorSeverity.AGENT_FAULT,
        )
    elif isinstance(error, ProfileResolutionError):
        code, key = (
            ("mtp.memory.type_mismatch", "mtp.call.profile_type_mismatch")
            if error.details["reason"] == "type_mismatch"
            else ("mtp.argument.invalid", "mtp.call.profile_invalid")
        )
        severity = MTPErrorSeverity.AGENT_FAULT
    elif stage == "frame":
        return error_response(agent_alias, sub_agent_error_info(agent_alias, error))
    return error_response(
        agent_alias,
        MTPErrorInfo(
            code=code,
            message_key=key,
            severity=severity,
            params=params,
            cause=str(error),
        ),
    )


def response_for_frame_result(
    agent_alias: str,
    result: FrameExecutionResult,
    *,
    reply: str = "",
    artifact_aliases: Sequence[str] = (),
    profile: AgentProfile | None = None,
    generation_options: Mapping[str, Any] | None = None,
) -> MTPCallResponse:
    """把被调方已定稿的结果映射为调用方唯一可见的响应。"""
    match result.status:
        case FrameExecutionStatus.COMPLETED:
            return success_response(
                agent_alias,
                reply=reply,
                artifact_aliases=artifact_aliases,
            )
        case FrameExecutionStatus.CANCELLED:
            return cancelled_response(agent_alias)
        case FrameExecutionStatus.BUDGET_EXHAUSTED:
            return error_response(
                agent_alias,
                sub_agent_budget_exhausted_info(agent_alias),
            )
        case FrameExecutionStatus.FAILED:
            if isinstance(result.error, ThreadEndedError):
                return error_response(agent_alias, mtp_error_info(result.error, verb="CALL"))
            if isinstance(result.error, ModelNotFoundError):
                model_name = (generation_options or {}).get("model") or getattr(
                    profile, "model_name", "unknown"
                )
                return error_response(
                    agent_alias,
                    agent_model_unavailable_info(agent_alias, str(model_name)),
                )
            return error_response(
                agent_alias,
                sub_agent_error_info(agent_alias, result.error),
            )
        case FrameExecutionStatus.SUSPENDED:
            raise ValueError("A suspended callee result cannot be mapped to a CALL response.")
        case unexpected_status:
            raise ValueError(f"Unsupported callee execution status: {unexpected_status!r}")


__all__ = [
    "agent_model_unavailable_info",
    "sub_agent_budget_exhausted_info",
    "sub_agent_error_info",
    "cancelled_response",
    "error_response",
    "preparation_error_response",
    "response_for_frame_result",
    "success_response",
]
