import pytest

from hivememory.agent_runtime.models import FrameExecutionResult, FrameExecutionStatus
from hivememory.alice.orchestration.sub_agent.call_response import (
    cancelled_response,
    preparation_error_response,
    response_for_frame_result,
    success_response,
)
from hivememory.core.models import OMNI_DOLL_PROFILE
from hivememory.core.mtp import MTPResponseStatus
from hivememory.core.mtp.exceptions import PermissionDeniedError
from hivememory.system.model_registry import ModelNotFoundError


def test_success_response_contains_reply_and_artifacts():
    response = success_response(
        "helper",
        reply="done",
        artifact_aliases=("draft-1",),
    )

    assert response.status == MTPResponseStatus.SUCCESS
    assert response.agent_alias == "helper"
    assert response.reply == "done"
    assert response.artifact_aliases == ["draft-1"]
    assert response.error is None


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
