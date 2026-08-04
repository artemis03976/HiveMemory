from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.core.mtp.exceptions import (
    AgentModelUnavailableError,
    MTPError,
    SubAgentBudgetExhaustedError,
    SubAgentExecutionError,
)
from hivememory.core.mtp.models import MTPCallResponse, MTPErrorInfo, MTPResponseStatus
from hivememory.system.model_registry import ModelNotFoundError

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


def preparation_error_response(agent_alias: str, error: Exception) -> MTPCallResponse:
    if isinstance(error, MTPError):
        return error_response(agent_alias, error.to_error_info())
    return error_response(
        agent_alias,
        SubAgentExecutionError(
            params={"agent_alias": agent_alias},
            cause=error,
        ).to_error_info(),
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
    """Map a finalized callee result to the only response visible to its caller."""
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
                SubAgentBudgetExhaustedError(
                    params={"agent_alias": agent_alias},
                ).to_error_info(),
            )
        case FrameExecutionStatus.FAILED:
            if isinstance(result.error, ModelNotFoundError):
                model_name = (generation_options or {}).get("model") or getattr(
                    profile, "model_name", "unknown"
                )
                return error_response(
                    agent_alias,
                    AgentModelUnavailableError(
                        params={
                            "agent_alias": agent_alias,
                            "model_name": model_name,
                        },
                        cause=result.error,
                    ).to_error_info(),
                )
            return error_response(
                agent_alias,
                SubAgentExecutionError(
                    params={"agent_alias": agent_alias},
                    cause=result.error,
                ).to_error_info(),
            )
        case FrameExecutionStatus.SUSPENDED:
            raise ValueError("A suspended callee result cannot be mapped to a CALL response.")
        case unexpected_status:
            raise ValueError(f"Unsupported callee execution status: {unexpected_status!r}")


__all__ = [
    "cancelled_response",
    "error_response",
    "preparation_error_response",
    "response_for_frame_result",
    "success_response",
]
