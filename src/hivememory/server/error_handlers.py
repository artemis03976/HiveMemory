"""HTTP 出口的错误分类、文案筛选与注册；路由只负责传输解析和业务调用。"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from hivememory.core.errors import (
    AccessRegistrationConflictError,
    AccessRegistrationNotFoundError,
    AdmissionDeniedError,
    AssetExpiredError,
    AssetFailedError,
    AssetNotFoundError,
    AssetNotReadyError,
    AssetOperationConflictError,
    AssetRemovedError,
    DomainError,
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
from hivememory.i18n.http_errors import get_http_error_text
from hivememory.infrastructure.attachments import UnsupportedAttachmentFormatError
from hivememory.infrastructure.attachments.errors import (
    AttachmentTooLargeError,
    EmptyAttachmentError,
    InvalidAttachmentNameError,
)
from hivememory.patchouli.contracts.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.server.models.errors import ErrorResponse, ValidationIssue
from hivememory.system.model_registry import DuplicateModelIdError
from hivememory.system.provider_registry import ProviderNotFoundError
from hivememory.system.services.import_bus import (
    ImportBusApplyFailedError,
    ImportBusUnavailableError,
)
from hivememory.workspace.contracts.errors import (
    IntentDispatchRejectedError,
    IntentDispatchUnknownError,
    PendingUpdateNotAllowedError,
    SessionBusyError,
    SessionCapacityError,
    SessionClosedError,
    SessionNotFoundError,
)

logger = logging.getLogger(__name__)

type HTTPErrorCode = Literal[
    "request.invalid_memory_id",
    "request.identity_empty",
    "request.principal_required",
    "request.agent_required",
    "request.agent_reserved",
    "request.workspace_not_found",
    "request.empty_update",
    "request.validation_failed",
    "request.multipart_invalid",
    "request.file_required",
    "request.single_file_required",
    "request.upload_fields_invalid",
    "request.file_content_required",
    "request.idempotency_key_required",
    "request.idempotency_key_too_long",
    "request.body_too_large",
    "request.not_found",
    "request.turn_required",
    "request.management_credential_invalid",
]

# 显式传输错误由调用点选择机器类别，不能根据异常 detail 反查分类；说明按错误码取自 i18n，
# 随语言设置。
_HTTP_ERRORS: dict[HTTPErrorCode, int] = {
    "request.invalid_memory_id": 400,
    "request.identity_empty": 400,
    "request.principal_required": 400,
    "request.agent_required": 400,
    "request.agent_reserved": 400,
    "request.workspace_not_found": 404,
    "request.empty_update": 400,
    "request.validation_failed": 400,
    "request.multipart_invalid": 400,
    "request.file_required": 400,
    "request.single_file_required": 400,
    "request.upload_fields_invalid": 400,
    "request.file_content_required": 400,
    "request.idempotency_key_required": 400,
    "request.idempotency_key_too_long": 400,
    "request.body_too_large": 413,
    "request.not_found": 404,
    "request.turn_required": 400,
    "request.management_credential_invalid": 401,
}


class APIError(HTTPException):
    """带具名类别的传输拒绝；安全呈现由本模块决定，不接受任意 detail。"""

    def __init__(self, code: HTTPErrorCode, *, issues: list[ValidationIssue] | None = None) -> None:
        status = _HTTP_ERRORS[code]
        message = get_http_error_text(code)
        self.payload = ErrorResponse(error=code, detail=message, issues=issues)
        super().__init__(status_code=status, detail=message)


# 映射只决定 HTTP 类别与错误码，不改变 owner 的判断与重试策略；安全说明按错误码取自 i18n。
_DOMAIN_ERRORS: dict[type[Exception], tuple[int, str]] = {
    OperationDeniedError: (403, "workspace.operation_denied"),
    AccessRegistrationNotFoundError: (404, "access.registration.not_found"),
    AccessRegistrationConflictError: (409, "access.registration.conflict"),
    ScopeRequiredError: (500, "workspace.scope_required"),
    WorkspaceMismatchError: (409, "workspace.mismatch"),
    SessionNotFoundError: (404, "workspace.session.not_found"),
    SessionBusyError: (409, "workspace.session.busy"),
    SessionCapacityError: (409, "workspace.session.capacity_exceeded"),
    SessionClosedError: (503, "workspace.session.closed"),
    ResourceNotFoundError: (404, "workspace.resource.not_found"),
    ResourceNotVisibleError: (404, "workspace.resource.not_found"),
    ResourceUnavailableError: (503, "workspace.resource.unavailable"),
    InvalidMemoryFieldError: (422, "memory.field.invalid"),
    MemoryAliasConflictError: (409, "workspace.memory.alias_conflict"),
    PendingUpdateNotAllowedError: (409, "workspace.memory.intent.pending_update_not_allowed"),
    IntentDispatchRejectedError: (503, "workspace.memory.intent.dispatch_rejected"),
    IntentDispatchUnknownError: (500, "workspace.memory.intent.dispatch_unknown"),
    ModelNotFoundError: (404, "model.not_found"),
    DuplicateModelIdError: (409, "model.duplicate_id"),
    ProviderNotFoundError: (404, "provider.not_found"),
    TopicBusyError: (409, "patchouli.topic.busy"),
    TopicSettleAdmissionError: (503, "patchouli.topic.settle_rejected"),
    ImportBusUnavailableError: (503, "import_bus.unavailable"),
    ImportBusApplyFailedError: (500, "import_bus.apply_failed"),
    AssetNotFoundError: (404, "workspace.asset.not_found"),
    AssetExpiredError: (410, "workspace.asset.expired"),
    AssetNotReadyError: (409, "workspace.asset.not_ready"),
    AssetFailedError: (409, "workspace.asset.failed"),
    AssetRemovedError: (410, "workspace.asset.removed"),
    StaleAssetResultError: (409, "workspace.asset.stale_result"),
    AssetOperationConflictError: (409, "workspace.asset.operation_conflict"),
    EmptyAttachmentError: (400, "attachment.empty"),
    InvalidAttachmentNameError: (400, "attachment.invalid_name"),
    AttachmentTooLargeError: (413, "attachment.too_large"),
    UnsupportedAttachmentFormatError: (415, "attachment.unsupported_format"),
    StorageOfflineError: (503, "storage.offline"),
    StorageReadError: (500, "storage.read_failed"),
    StorageWriteError: (500, "storage.write_failed"),
}

_REASON_ALLOWLIST: dict[type[Exception], frozenset[str]] = {
    AdmissionDeniedError: frozenset(
        {
            "authentication_gateway_closed",
            "unknown_principal",
            "adapter_mismatch",
            "invalid_credential",
            "actor_not_allowed_for_principal",
            "agent_not_allowed_for_principal",
            "actor_not_owner",
            "actor_not_admitted",
            "system_actor_not_allowed",
        }
    ),
    OperationDeniedError: frozenset(
        {
            "actor_not_admitted",
            "operation_not_allowed",
            "root_required",
            "target_workspace_not_resident",
            "target_owner_mismatch",
        }
    ),
    AccessRegistrationConflictError: frozenset(
        {"protected", "duplicate", "credential_from_env", "file_modified"}
    ),
    WorkspaceMismatchError: frozenset({"target_workspace_not_resident", "target_owner_mismatch"}),
    ResourceUnavailableError: frozenset({"lifecycle_unavailable"}),
    MemoryAliasConflictError: frozenset(
        {"alias_occupied", "alias_candidates_exhausted", "ambiguous_alias"}
    ),
    AssetOperationConflictError: frozenset({"store_closed"}),
    ImportBusUnavailableError: frozenset(
        {"gateway_unavailable", "queue_unavailable", "apply_pending", "shutting_down"}
    ),
}


# 有专门说明的校验类别；其余类别都按 value_error 呈现。
_VALIDATION_KINDS = frozenset(
    {
        "missing",
        "extra_forbidden",
        "string_type",
        "int_type",
        "int_parsing",
        "float_type",
        "float_parsing",
        "bool_type",
        "bool_parsing",
        "list_type",
        "dict_type",
        "json_invalid",
        "uuid_parsing",
        "string_too_short",
        "string_too_long",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "literal_error",
        "enum",
        "value_error",
    }
)


def validation_issues(errors: Sequence[Mapping[str, Any]]) -> list[ValidationIssue]:
    """只投影字段位置与受控类别，忽略原始 msg、input 和 ctx。"""
    result = []
    for error in errors:
        kind = error.get("type", "value_error")
        kind = kind if isinstance(kind, str) and kind in _VALIDATION_KINDS else "value_error"
        result.append(
            ValidationIssue(
                loc=[part for part in error.get("loc", ()) if isinstance(part, (str, int))],
                type=kind,
                message=get_http_error_text(f"validation.{kind}"),
            )
        )
    return result


def project_error(error: Exception) -> tuple[int, ErrorResponse]:
    """按类型投影 API 失败，未知异常及未知 HTTP detail 均不公开正文。"""
    if isinstance(error, APIError):
        return error.status_code, error.payload
    if isinstance(error, RequestValidationError):
        return 422, ErrorResponse(
            error="request.validation_failed",
            detail=get_http_error_text("request.input_invalid"),
            issues=validation_issues(error.errors()),
        )
    if isinstance(error, UnsupportedAttachmentFormatError):
        # 格式 owner 的受控分类可以决定帮助文案，异常正文仍不直接公开。
        key = "attachment.unsupported_format"
        if error.reason in ("legacy_doc_format", "conflicting_media_type"):
            key = f"{key}.{error.reason}"
        message = get_http_error_text(key)
        return 415, ErrorResponse(error="attachment.unsupported_format", detail=message)
    reason = error.details.get("reason") if isinstance(error, DomainError) else None
    allowed = _REASON_ALLOWLIST.get(type(error), frozenset())
    reason = reason if isinstance(reason, str) and reason in allowed else None
    if isinstance(error, AdmissionDeniedError):
        status = 503 if reason == "authentication_gateway_closed" else 403
        message = get_http_error_text(
            "admission.gateway_closed" if status == 503 else "admission.denied"
        )
        return status, ErrorResponse(error=error.code, detail=message, reason=reason)
    if isinstance(error, AssetOperationConflictError) and reason == "store_closed":
        return 503, ErrorResponse(
            error=error.code,
            detail=get_http_error_text("workspace.asset.store_closed"),
            reason=reason,
        )
    for error_type, (status, code) in _DOMAIN_ERRORS.items():
        if isinstance(error, error_type):
            return status, ErrorResponse(
                error=code, detail=get_http_error_text(code), reason=reason
            )
    if isinstance(error, HTTPException) and error.status_code < 500:
        return error.status_code, ErrorResponse(
            error=f"http.{error.status_code}", detail=get_http_error_text("http.client_error")
        )
    status = error.status_code if isinstance(error, HTTPException) else 500
    return status, ErrorResponse(
        error="server.internal_error", detail=get_http_error_text("server.internal_error")
    )


async def handle_error(request: Request, exc: Exception) -> JSONResponse:
    """HTTP 完整响应尚未开始时的唯一错误出口；保留认证等必需响应头。"""
    status, payload = project_error(exc)
    if status >= 500:
        # 不重复打印正文；完整原因链只在此请求边界记录一次。
        logger.error(
            "HTTP request failed: %s", payload.error, exc_info=(type(exc), exc, exc.__traceback__)
        )
    return JSONResponse(
        status_code=status,
        content=payload.model_dump(exclude_none=True),
        headers=exc.headers if isinstance(exc, HTTPException) else None,
    )


def install_error_handlers(app: FastAPI) -> None:
    """装配 API 异常处理器；已开始的 SSE、readiness 和静态响应不经此重写。"""
    for error_type in (
        *_DOMAIN_ERRORS,
        AdmissionDeniedError,
        HTTPException,
        RequestValidationError,
        Exception,
    ):
        app.add_exception_handler(error_type, handle_error)


def error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """让路由声明引用同一错误 schema，包括替换框架默认的 422 结构。"""
    return {status: {"model": ErrorResponse, "description": "API 错误"} for status in statuses}
