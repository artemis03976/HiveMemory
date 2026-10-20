import pytest

from hivememory.agent_runtime.frame import FrameExecutionResult, FrameExecutionStatus
from hivememory.alice.orchestration.call.response import (
    cancelled_response,
    preparation_error_response,
    response_for_frame_result,
)
from hivememory.core.errors import (
    AdmissionDeniedError,
    ModelNotFoundError,
    OperationDeniedError,
    ResourceNotFoundError,
    ResourceUnavailableError,
)
from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.core.mtp import MTPResponseStatus
from hivememory.core.mtp.exceptions import PermissionDeniedError
from hivememory.patchouli.contracts.errors import ProfileResolutionError


def test_cancelled_response_is_empty_cancelled_envelope():
    response = cancelled_response("helper")

    assert response.status == MTPResponseStatus.CANCELLED
    assert response.agent_alias == "helper"
    assert response.reply == ""
    assert response.artifact_aliases == []
    assert response.error is None


def test_preparation_mtp_error_preserves_structured_error():
    response = preparation_error_response("helper", PermissionDeniedError("denied"))

    assert response.status == MTPResponseStatus.ERROR
    assert response.error is not None
    assert response.error.code == "mtp.permission.denied"


def test_preparation_generic_error_maps_to_sub_agent_error():
    response = preparation_error_response("helper", RuntimeError("broken"))

    assert response.status == MTPResponseStatus.ERROR
    assert response.error is not None
    assert response.error.code == "mtp.call_response.sub_agent_error"


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (FrameExecutionStatus.BUDGET_EXHAUSTED, "mtp.call_response.budget_exhausted"),
        (FrameExecutionStatus.FAILED, "mtp.call_response.sub_agent_error"),
    ],
)
def test_terminal_errors_map_to_stable_call_errors(status, expected_code):
    response = response_for_frame_result(
        "helper",
        FrameExecutionResult(status=status),
    )

    assert response.status == MTPResponseStatus.ERROR
    assert response.error is not None
    assert response.error.code == expected_code


def test_model_not_found_maps_to_model_unavailable():
    response = response_for_frame_result(
        "helper",
        FrameExecutionResult(status=FrameExecutionStatus.FAILED, error=ModelNotFoundError()),
        profile=OMNI_DOLL_PROFILE,
    )

    assert response.status == MTPResponseStatus.ERROR
    assert response.error is not None
    assert response.error.code == "mtp.system.service_unavailable"
    assert response.error.params["model_name"] == "default"


def test_completed_result_maps_to_success():
    response = response_for_frame_result(
        "helper",
        FrameExecutionResult(status=FrameExecutionStatus.COMPLETED),
        reply="done",
        artifact_aliases=("draft-1",),
    )

    assert response.status == MTPResponseStatus.SUCCESS
    assert response.reply == "done"
    assert response.artifact_aliases == ["draft-1"]


def test_cancelled_result_maps_to_cancelled_response():
    response = response_for_frame_result(
        "helper",
        FrameExecutionResult(status=FrameExecutionStatus.CANCELLED),
    )

    assert response.status == MTPResponseStatus.CANCELLED


def test_suspended_result_is_a_coordinator_invariant_violation():
    with pytest.raises(ValueError, match="cannot be mapped"):
        response_for_frame_result(
            "helper",
            FrameExecutionResult(status=FrameExecutionStatus.SUSPENDED),
        )


def test_preparation_storage_failure_keeps_storage_wire_category():
    """CALL 准备的中立存储失败走共享映射，不变成通用子 Agent 执行错误。"""
    from hivememory.core.errors import StorageReadError

    response = preparation_error_response("helper", StorageReadError(details={"operation": "read"}))

    assert response.status == MTPResponseStatus.ERROR
    assert response.error.code == "mtp.system.storage_error"
    assert response.error.params == {}
    assert "cause" not in response.error.model_dump()


@pytest.mark.parametrize(
    ("error", "expected_code", "expected_key"),
    [
        (
            OperationDeniedError("agent.call 未获授权"),
            "mtp.permission.denied",
            "mtp.permission.verb_denied",
        ),
        (
            AdmissionDeniedError("目标没有准入", details={"reason": "actor_not_admitted"}),
            "mtp.permission.denied",
            "mtp.call.agent_unavailable",
        ),
        (
            ResourceUnavailableError("Profile 存储路由不可达"),
            "mtp.system.service_unavailable",
            "mtp.system.service_unavailable",
        ),
        (
            ResourceNotFoundError(details={"resource": "agent_profile"}),
            "mtp.alias.not_found",
            "mtp.call.profile_not_found",
        ),
        (
            ProfileResolutionError(agent_alias="helper", reason="type_mismatch"),
            "mtp.memory.type_mismatch",
            "mtp.call.profile_type_mismatch",
        ),
        (
            ProfileResolutionError(agent_alias="helper", reason="invalid_config"),
            "mtp.argument.invalid",
            "mtp.call.profile_invalid",
        ),
        (
            RuntimeError("profile unavailable"),
            "mtp.system.fault",
            "mtp.call.profile_load_failed",
        ),
    ],
)
def test_context_failure_maps_to_safe_call_error(error, expected_code, expected_key):
    """准备失败保持稳定类别，公开参数不透传 owner 的诊断数据。"""
    response = preparation_error_response("helper", error, stage="context")
    assert response.error.code == expected_code
    assert response.error.message_key == expected_key
    assert response.error.params == (
        {"verb": "CALL"}
        if isinstance(error, OperationDeniedError)
        else {} if isinstance(error, ResourceUnavailableError) else {"agent_alias": "helper"}
    )
    assert "cause" not in response.model_dump()["error"]


@pytest.mark.parametrize("stage", ["context", "result"])
def test_revoked_call_thread_uses_stop_message_instead_of_profile_retry_advice(stage):
    """CALL 共享线程撤销的安全提示，不把迟到操作误报为 Profile 不可用。"""
    from hivememory.core.errors import ThreadEndedError
    from hivememory.core.mtp import MTPFormatter

    error = ThreadEndedError("secret")
    response = (
        preparation_error_response("helper", error, stage="context")
        if stage == "context"
        else response_for_frame_result(
            "helper", FrameExecutionResult(status=FrameExecutionStatus.FAILED, error=error)
        )
    )
    assert response.error.code == "mtp.system.fault"
    assert "原线程已结束" in MTPFormatter.format_call_response(response, "zh")
    assert "secret" not in response.model_dump_json()
