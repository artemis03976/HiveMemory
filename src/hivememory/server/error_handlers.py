"""HTTP 出口的错误分类、文案筛选与注册；路由只负责传输解析和业务调用。"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from hivememory.core.errors import (
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
from hivememory.workspace.contracts.errors import (
    SessionBusyError,
    SessionCapacityError,
    SessionClosedError,
    SessionNotFoundError,
)

logger = logging.getLogger(__name__)

type HTTPErrorCode = Literal[
    "request.invalid_memory_id",
    "request.identity_conflict",
    "request.identity_empty",
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
    "request.not_found",
]

# 显式传输错误由调用点选择机器类别，不能根据异常 detail 反查分类。
_HTTP_ERRORS: dict[HTTPErrorCode, tuple[int, str]] = {
    "request.invalid_memory_id": (400, "无效的记忆 ID 格式"),
    "request.identity_conflict": (409, "身份选择冲突，请在同一请求中只表达一种选择"),
    "request.identity_empty": (400, "身份字段不能为空"),
    "request.agent_required": (400, "该操作必须显式提供 agent_id"),
    "request.agent_reserved": (400, "不能使用保留的 system 作为 agent_id"),
    "request.workspace_not_found": (404, "Workspace 不存在或未开放"),
    "request.empty_update": (400, "请求体中没有需要更新的字段"),
    "request.validation_failed": (400, "配置输入不符合要求"),
    "request.multipart_invalid": (400, "multipart 表单解析失败"),
    "request.file_required": (400, "缺少 file 文件字段"),
    "request.single_file_required": (
        400,
        "一个上传请求只能包含一个 file 文件部分，多文件请逐个上传",
    ),
    "request.upload_fields_invalid": (400, "上传请求包含未知的业务字段"),
    "request.file_content_required": (400, "file 字段必须携带文件内容"),
    "request.idempotency_key_required": (
        400,
        "缺少 Idempotency-Key 请求头；同一文件重试必须沿用同一取值",
    ),
    "request.idempotency_key_too_long": (400, "Idempotency-Key 过长"),
    "request.not_found": (404, "请求的资源不存在"),
}


class APIError(HTTPException):
    """带具名类别的传输拒绝；安全呈现由本模块决定，不接受任意 detail。"""

    def __init__(self, code: HTTPErrorCode, *, issues: list[ValidationIssue] | None = None) -> None:
        status, message = _HTTP_ERRORS[code]
        self.payload = ErrorResponse(error=code, detail=message, issues=issues)
        super().__init__(status_code=status, detail=message)


# 映射只决定 HTTP 类别与安全呈现，不改变 owner 的判断与重试策略。
_DOMAIN_ERRORS: dict[type[Exception], tuple[int, str, str]] = {
    OperationDeniedError: (403, "workspace.operation_denied", "当前身份无权执行该操作"),
    ScopeRequiredError: (500, "workspace.scope_required", "请求处理失败，请联系服务维护者"),
    WorkspaceMismatchError: (409, "workspace.mismatch", "请求与目标 Workspace 不一致"),
    SessionNotFoundError: (404, "workspace.session.not_found", "会话不存在"),
    SessionBusyError: (409, "workspace.session.busy", "会话正在处理，请稍后重试"),
    SessionCapacityError: (409, "workspace.session.capacity_exceeded", "会话容量已满"),
    SessionClosedError: (503, "workspace.session.closed", "会话服务已关闭"),
    ResourceNotFoundError: (404, "workspace.resource.not_found", "资源不存在或不可访问"),
    ResourceNotVisibleError: (404, "workspace.resource.not_found", "资源不存在或不可访问"),
    ResourceUnavailableError: (503, "workspace.resource.unavailable", "服务当前不可用，请稍后重试"),
    InvalidMemoryFieldError: (422, "memory.field.invalid", "记忆字段不符合要求，请检查输入"),
    MemoryAliasConflictError: (409, "workspace.memory.alias_conflict", "记忆别名冲突，请更换别名"),
    ModelNotFoundError: (404, "model.not_found", "模型不存在"),
    DuplicateModelIdError: (409, "model.duplicate_id", "模型 ID 已存在"),
    ProviderNotFoundError: (404, "provider.not_found", "提供商不存在或不能删除"),
    TopicBusyError: (409, "patchouli.topic.busy", "话题正在处理，请稍后重试"),
    TopicSettleAdmissionError: (
        503,
        "patchouli.topic.settle_rejected",
        "结算材料暂未被生成队列接纳，话题内容已保留，可重试",
    ),
    AssetNotFoundError: (404, "workspace.asset.not_found", "附件不存在或已失效"),
    AssetExpiredError: (410, "workspace.asset.expired", "附件已失效，请重新上传"),
    AssetNotReadyError: (409, "workspace.asset.not_ready", "附件尚未就绪"),
    AssetFailedError: (409, "workspace.asset.failed", "附件处理失败，请重新上传"),
    AssetRemovedError: (410, "workspace.asset.removed", "原上传操作对应的资产已被移除，请重新上传"),
    StaleAssetResultError: (409, "workspace.asset.stale_result", "上传结果提交冲突，请重新上传"),
    AssetOperationConflictError: (
        409,
        "workspace.asset.operation_conflict",
        "同一 Idempotency-Key 携带了不一致的文件内容或元数据",
    ),
    EmptyAttachmentError: (400, "attachment.empty", "文件内容为空"),
    InvalidAttachmentNameError: (400, "attachment.invalid_name", "文件名不符合要求"),
    AttachmentTooLargeError: (413, "attachment.too_large", "附件超过允许的大小上限"),
    UnsupportedAttachmentFormatError: (415, "attachment.unsupported_format", "不支持该附件格式"),
    StorageOfflineError: (503, "storage.offline", "记忆存储当前不可用"),
    StorageReadError: (500, "storage.read_failed", "记忆读取失败，请稍后再试"),
    StorageWriteError: (
        500,
        "storage.write_failed",
        "记忆修改未确认生效，请先核实结果，避免重复提交",
    ),
}

_REASON_ALLOWLIST: dict[type[Exception], frozenset[str]] = {
    AdmissionDeniedError: frozenset(
        {
            "authentication_gateway_closed",
            "unknown_principal",
            "adapter_mismatch",
            "actor_not_allowed_for_principal",
            "actor_not_owner",
            "actor_not_admitted",
            "system_actor_not_allowed",
        }
    ),
    OperationDeniedError: frozenset(
        {
            "operation_not_allowed",
            "root_required",
            "target_workspace_not_resident",
            "target_owner_mismatch",
        }
    ),
    ScopeRequiredError: frozenset({"context_not_issued", "actor_not_admitted"}),
    WorkspaceMismatchError: frozenset({"target_workspace_not_resident", "target_owner_mismatch"}),
    ResourceUnavailableError: frozenset({"lifecycle_unavailable"}),
    MemoryAliasConflictError: frozenset(
        {"alias_occupied", "alias_candidates_exhausted", "ambiguous_alias"}
    ),
    AssetOperationConflictError: frozenset({"store_closed"}),
}


def validation_issues(errors: Sequence[Mapping[str, Any]]) -> list[ValidationIssue]:
    """只投影字段位置与受控类别，忽略原始 msg、input 和 ctx。"""
    messages = {
        "missing": "缺少必填字段",
        "extra_forbidden": "不允许此字段",
        "string_type": "应提供字符串",
        "int_type": "应提供整数",
        "int_parsing": "应提供整数",
        "float_type": "应提供数字",
        "float_parsing": "应提供数字",
        "bool_type": "应提供布尔值",
        "bool_parsing": "应提供布尔值",
        "list_type": "应提供列表",
        "dict_type": "应提供对象",
        "json_invalid": "JSON 格式不合法",
        "uuid_parsing": "标识符格式不合法",
        "string_too_short": "文本过短",
        "string_too_long": "文本过长",
        "greater_than": "数值超出允许范围",
        "greater_than_equal": "数值超出允许范围",
        "less_than": "数值超出允许范围",
        "less_than_equal": "数值超出允许范围",
        "literal_error": "取值不在允许范围内",
        "enum": "取值不在允许范围内",
        "value_error": "字段取值不合法",
    }
    result = []
    for error in errors:
        kind = error.get("type", "value_error")
        kind = kind if isinstance(kind, str) and kind in messages else "value_error"
        result.append(
            ValidationIssue(
                loc=[part for part in error.get("loc", ()) if isinstance(part, (str, int))],
                type=kind,
                message=messages[kind],
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
            detail="请求输入不符合要求",
            issues=validation_issues(error.errors()),
        )
    if isinstance(error, UnsupportedAttachmentFormatError):
        # 格式 owner 的受控分类可以决定帮助文案，异常正文仍不直接公开。
        message = {
            "legacy_doc_format": "不支持 .doc 格式，请另存为 .docx 后重新上传",
            "conflicting_media_type": "声明的文件类型与扩展名冲突，请确认文件格式后重新上传",
        }.get(error.reason, "不支持该附件格式")
        return 415, ErrorResponse(error="attachment.unsupported_format", detail=message)
    reason = error.details.get("reason") if isinstance(error, DomainError) else None
    allowed = _REASON_ALLOWLIST.get(type(error), frozenset())
    reason = reason if isinstance(reason, str) and reason in allowed else None
    if isinstance(error, AdmissionDeniedError):
        status = 503 if reason == "authentication_gateway_closed" else 403
        message = "认证服务已关闭" if status == 503 else "当前身份未获准访问该 Workspace"
        return status, ErrorResponse(error=error.code, detail=message, reason=reason)
    if isinstance(error, AssetOperationConflictError) and reason == "store_closed":
        return 503, ErrorResponse(
            error=error.code, detail="附件存储当前不可用，请稍后重试", reason=reason
        )
    for error_type, (status, code, message) in _DOMAIN_ERRORS.items():
        if isinstance(error, error_type):
            return status, ErrorResponse(error=code, detail=message, reason=reason)
    if isinstance(error, HTTPException) and error.status_code < 500:
        return error.status_code, ErrorResponse(
            error=f"http.{error.status_code}", detail="请求无法完成，请检查输入或访问权限"
        )
    status = error.status_code if isinstance(error, HTTPException) else 500
    return status, ErrorResponse(error="server.internal_error", detail="服务器内部错误，请稍后再试")


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
