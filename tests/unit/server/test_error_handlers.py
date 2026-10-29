"""HTTP 出口的权威映射与安全投影；不复制 owner 的业务判断矩阵。"""

from functools import partial

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException

from hivememory.core.errors import (
    AdmissionDeniedError,
    AssetExpiredError,
    AssetFailedError,
    AssetNotFoundError,
    AssetNotReadyError,
    AssetOperationConflictError,
    AssetRemovedError,
    InvalidMemoryFieldError,
    MemoryAliasConflictError,
    ModelNotFoundError,
    OperationDeniedError,
    ResourceNotFoundError,
    ResourceNotVisibleError,
    ResourceUnavailableError,
    ScopeRequiredError,
    StaleAssetResultError,
    StorageOfflineError,
    StorageReadError,
    StorageWriteError,
    WorkspaceMismatchError,
)
from hivememory.infrastructure.attachments import UnsupportedAttachmentFormatError
from hivememory.infrastructure.attachments.errors import (
    AttachmentTooLargeError,
    EmptyAttachmentError,
    InvalidAttachmentNameError,
)
from hivememory.patchouli.contracts.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.server.error_handlers import install_error_handlers, project_error
from hivememory.system.model_registry import DuplicateModelIdError
from hivememory.system.provider_registry import ProviderNotFoundError
from hivememory.system.services.import_bus import ImportBusUnavailableError
from hivememory.workspace.contracts.errors import (
    SessionBusyError,
    SessionCapacityError,
    SessionClosedError,
    SessionNotFoundError,
)


@pytest.mark.parametrize(
    ("error_type", "status", "code"),
    [
        (OperationDeniedError, 403, "workspace.operation_denied"),
        (ScopeRequiredError, 500, "workspace.scope_required"),
        (WorkspaceMismatchError, 409, "workspace.mismatch"),
        (SessionNotFoundError, 404, "workspace.session.not_found"),
        (SessionBusyError, 409, "workspace.session.busy"),
        (SessionCapacityError, 409, "workspace.session.capacity_exceeded"),
        (SessionClosedError, 503, "workspace.session.closed"),
        (ResourceNotFoundError, 404, "workspace.resource.not_found"),
        (ResourceNotVisibleError, 404, "workspace.resource.not_found"),
        (ResourceUnavailableError, 503, "workspace.resource.unavailable"),
        (InvalidMemoryFieldError, 422, "memory.field.invalid"),
        (MemoryAliasConflictError, 409, "workspace.memory.alias_conflict"),
        (ModelNotFoundError, 404, "model.not_found"),
        (DuplicateModelIdError, 409, "model.duplicate_id"),
        (ProviderNotFoundError, 404, "provider.not_found"),
        (TopicBusyError, 409, "patchouli.topic.busy"),
        (TopicSettleAdmissionError, 503, "patchouli.topic.settle_rejected"),
        (AssetNotFoundError, 404, "workspace.asset.not_found"),
        (AssetExpiredError, 410, "workspace.asset.expired"),
        (AssetNotReadyError, 409, "workspace.asset.not_ready"),
        (AssetFailedError, 409, "workspace.asset.failed"),
        (AssetRemovedError, 410, "workspace.asset.removed"),
        (StaleAssetResultError, 409, "workspace.asset.stale_result"),
        (AssetOperationConflictError, 409, "workspace.asset.operation_conflict"),
        (EmptyAttachmentError, 400, "attachment.empty"),
        (InvalidAttachmentNameError, 400, "attachment.invalid_name"),
        (AttachmentTooLargeError, 413, "attachment.too_large"),
        (
            partial(UnsupportedAttachmentFormatError, reason="unsupported_extension"),
            415,
            "attachment.unsupported_format",
        ),
        (StorageOfflineError, 503, "storage.offline"),
        (StorageReadError, 500, "storage.read_failed"),
        (StorageWriteError, 500, "storage.write_failed"),
    ],
)
def test_domain_errors_keep_http_categories_without_diagnostics(error_type, status, code):
    """每个映射条目的状态与机器码稳定，异常原文绝不能用作 detail。"""
    error = error_type("secret-token@internal-host")
    actual_status, payload = project_error(error)
    assert (actual_status, payload.error) == (status, code)
    assert "secret-token" not in payload.model_dump_json()


@pytest.mark.parametrize(
    ("error", "status", "reason"),
    [
        (
            AdmissionDeniedError(details={"reason": "actor_not_admitted", "principal": "secret"}),
            403,
            "actor_not_admitted",
        ),
        (
            AdmissionDeniedError(details={"reason": "invalid_credential"}),
            403,
            "invalid_credential",
        ),
        (
            AdmissionDeniedError(details={"reason": "authentication_gateway_closed"}),
            503,
            "authentication_gateway_closed",
        ),
        (AssetOperationConflictError(details={"reason": "store_closed"}), 503, "store_closed"),
        (ImportBusUnavailableError(reason="gateway_unavailable"), 503, "gateway_unavailable"),
        (ImportBusUnavailableError(reason="queue_unavailable"), 503, "queue_unavailable"),
        (ImportBusUnavailableError(reason="shutting_down"), 503, "shutting_down"),
        (ScopeRequiredError(details={"reason": "actor_not_admitted"}), 500, "actor_not_admitted"),
        (
            OperationDeniedError(details={"reason": "target_workspace_not_resident"}),
            403,
            "target_workspace_not_resident",
        ),
        (
            OperationDeniedError(details={"reason": "target_owner_mismatch"}),
            403,
            "target_owner_mismatch",
        ),
        (OperationDeniedError(details={"reason": "agent_required"}), 403, "agent_required"),
        (OperationDeniedError(details={"reason": "secret-token"}), 403, None),
    ],
)
def test_only_declared_reasons_cross_http_boundary(error, status, reason):
    actual_status, payload = project_error(error)
    assert actual_status == status
    assert payload.reason == reason
    assert "secret" not in payload.model_dump_json()


def test_request_validation_strips_input_context_and_custom_validator_text():
    """自定义 validator 即使包含凭据也只公开受控类别和字段位置。"""
    error = RequestValidationError(
        [
            {
                "loc": ("body", "providers", "api_key"),
                "type": "secret-custom-error",
                "msg": "secret-token",
                "input": {"api_key": "secret-token"},
                "ctx": {"error": ValueError("secret-token")},
            }
        ]
    )
    status, payload = project_error(error)
    assert status == 422
    assert payload.model_dump(exclude_none=True) == {
        "error": "request.validation_failed",
        "detail": "请求输入不符合要求",
        "issues": [
            {
                "loc": ["body", "providers", "api_key"],
                "type": "value_error",
                "message": "字段取值不合法",
            }
        ],
    }


@pytest.mark.parametrize(
    ("error", "status", "body"),
    [
        (
            HTTPException(401, detail={"token": "secret"}, headers={"WWW-Authenticate": "Bearer"}),
            401,
            {"error": "http.401", "detail": "请求无法完成，请检查输入或访问权限"},
        ),
        (
            HTTPException(503, detail="secret", headers={"Retry-After": "10"}),
            503,
            {"error": "server.internal_error", "detail": "服务器内部错误，请稍后再试"},
        ),
        (
            RuntimeError("secret"),
            500,
            {"error": "server.internal_error", "detail": "服务器内部错误，请稍后再试"},
        ),
    ],
)
def test_installed_handlers_preserve_status_headers_and_hide_unknown_detail(error, status, body):
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/failure")
    async def fail():
        raise error

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/failure")
    assert response.status_code == status
    assert response.json() == body
    assert response.headers.get("www-authenticate") == ("Bearer" if status == 401 else None)
    assert response.headers.get("retry-after") == ("10" if status == 503 else None)


def test_api_schema_uses_error_envelope_for_validation_and_domain_errors():
    """公开路由的错误响应不能继续声明框架默认的 detail 列表。"""
    from hivememory.server.app import app

    schema = app.openapi()
    expected = {"$ref": "#/components/schemas/ErrorResponse"}
    for path, methods in schema["paths"].items():
        if not path.startswith("/api/"):
            continue
        for operation in methods.values():
            for status, response in operation["responses"].items():
                if status in {"400", "403", "404", "409", "410", "413", "415", "422", "500", "503"}:
                    model = response["content"]["application/json"]["schema"]
                    if path == "/api/v1/runtime-events/stream" and status == "503":
                        assert model == {
                            "$ref": "#/components/schemas/RuntimeEventDisabledResponse"
                        }
                    else:
                        assert model == expected
    assert schema["components"]["schemas"]["ErrorResponse"]["required"] == ["error", "detail"]
