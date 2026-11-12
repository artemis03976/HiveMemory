"""HTTP 出口与 MCP 工具错误结果中的安全说明。

key 是错误码，或错误码下按受控原因细分的说明；错误码、HTTP 状态与 ``reason`` 机器值不在这里。
校验问题的说明以 ``validation.<类别>`` 为 key。
"""

from __future__ import annotations

from hivememory.i18n.resolver import resolve_language
from hivememory.i18n.types import Language

_HTTP_ERROR_TEXT_ZH = {
    # 显式的传输拒绝
    "request.invalid_memory_id": "无效的记忆 ID 格式",
    "request.identity_empty": "身份字段不能为空",
    "request.principal_required": "外部入口必须以 X-Principal-ID 声明调用来源",
    "request.agent_required": "该操作必须显式提供 agent_id",
    "request.agent_reserved": "不能使用保留的 system 作为 agent_id",
    "request.workspace_not_found": "Workspace 不存在或未开放",
    "request.empty_update": "请求体中没有需要更新的字段",
    "request.validation_failed": "配置输入不符合要求",
    "request.multipart_invalid": "multipart 表单解析失败",
    "request.file_required": "缺少 file 文件字段",
    "request.single_file_required": "一个上传请求只能包含一个 file 文件部分，多文件请逐个上传",
    "request.upload_fields_invalid": "上传请求包含未知的业务字段",
    "request.file_content_required": "file 字段必须携带文件内容",
    "request.idempotency_key_required": "缺少 Idempotency-Key 请求头；同一文件重试必须沿用同一取值",
    "request.idempotency_key_too_long": "Idempotency-Key 过长",
    "request.body_too_large": "请求体超过允许的大小上限",
    "request.not_found": "请求的资源不存在",
    "request.turn_required": "写入与更新必须由连接器补上当前一轮的信息",
    "request.management_credential_invalid": "管理接口需要管理凭据：缺少凭据或凭据不正确",
    "request.input_invalid": "请求输入不符合要求",
    # 领域错误
    "workspace.operation_denied": "当前身份无权执行该操作",
    "access.registration.not_found": "访问登记不存在",
    "access.registration.conflict": "访问登记未修改：与现有登记冲突，或该登记只能经登记文件修改",
    "workspace.scope_required": "请求处理失败，请联系服务维护者",
    "workspace.mismatch": "请求与目标 Workspace 不一致",
    "workspace.session.not_found": "会话不存在",
    "workspace.session.busy": "会话正在处理，请稍后重试",
    "workspace.session.capacity_exceeded": "会话容量已满",
    "workspace.session.closed": "会话服务已关闭",
    "workspace.resource.not_found": "资源不存在或不可访问",
    "workspace.resource.unavailable": "服务当前不可用，请稍后重试",
    "memory.field.invalid": "记忆字段不符合要求，请检查输入",
    "workspace.memory.alias_conflict": "记忆别名冲突，请更换别名",
    "workspace.memory.intent.pending_update_not_allowed": (
        "这条记忆仍在生成中，不能作为更新的对象，生成完成后请改用正式 alias"
    ),
    "workspace.memory.intent.dispatch_rejected": "生成队列没有接纳这次提交，请稍后重新提交",
    "workspace.memory.intent.dispatch_unknown": (
        "无法确认这次提交是否已被接纳，请稍后检索核实，避免重复提交"
    ),
    "model.not_found": "模型不存在",
    "model.duplicate_id": "模型 ID 已存在",
    "provider.not_found": "提供商不存在或不能删除",
    "patchouli.topic.busy": "话题正在处理，请稍后重试",
    "patchouli.topic.settle_rejected": "结算材料暂未被生成队列接纳，话题内容已保留，可重试",
    "import_bus.unavailable": "导入暂时无法完成，请保留该轮并稍后重发",
    "import_bus.apply_failed": "导入的轮次在应用时失败，重发时会重新路由并再次提交",
    "workspace.asset.not_found": "附件不存在或已失效",
    "workspace.asset.expired": "附件已失效，请重新上传",
    "workspace.asset.not_ready": "附件尚未就绪",
    "workspace.asset.failed": "附件处理失败，请重新上传",
    "workspace.asset.removed": "原上传操作对应的资产已被移除，请重新上传",
    "workspace.asset.stale_result": "上传结果提交冲突，请重新上传",
    "workspace.asset.operation_conflict": "同一 Idempotency-Key 携带了不一致的文件内容或元数据",
    "workspace.asset.store_closed": "附件存储当前不可用，请稍后重试",
    "attachment.empty": "文件内容为空",
    "attachment.invalid_name": "文件名不符合要求",
    "attachment.too_large": "附件超过允许的大小上限",
    "attachment.unsupported_format": "不支持该附件格式",
    "attachment.unsupported_format.legacy_doc_format": "不支持 .doc 格式，请另存为 .docx 后重新上传",
    "attachment.unsupported_format.conflicting_media_type": (
        "声明的文件类型与扩展名冲突，请确认文件格式后重新上传"
    ),
    "storage.offline": "记忆存储当前不可用",
    "storage.read_failed": "记忆读取失败，请稍后再试",
    "storage.write_failed": "记忆修改未确认生效，请先核实结果，避免重复提交",
    "admission.gateway_closed": "认证服务已关闭",
    "admission.denied": "当前身份未获准访问该 Workspace",
    "http.client_error": "请求无法完成，请检查输入或访问权限",
    "server.internal_error": "服务器内部错误，请稍后再试",
    # 校验问题
    "validation.missing": "缺少必填字段",
    "validation.extra_forbidden": "不允许此字段",
    "validation.string_type": "应提供字符串",
    "validation.int_type": "应提供整数",
    "validation.int_parsing": "应提供整数",
    "validation.float_type": "应提供数字",
    "validation.float_parsing": "应提供数字",
    "validation.bool_type": "应提供布尔值",
    "validation.bool_parsing": "应提供布尔值",
    "validation.list_type": "应提供列表",
    "validation.dict_type": "应提供对象",
    "validation.json_invalid": "JSON 格式不合法",
    "validation.uuid_parsing": "标识符格式不合法",
    "validation.string_too_short": "文本过短",
    "validation.string_too_long": "文本过长",
    "validation.greater_than": "数值超出允许范围",
    "validation.greater_than_equal": "数值超出允许范围",
    "validation.less_than": "数值超出允许范围",
    "validation.less_than_equal": "数值超出允许范围",
    "validation.literal_error": "取值不在允许范围内",
    "validation.enum": "取值不在允许范围内",
    "validation.value_error": "字段取值不合法",
}

_HTTP_ERROR_TEXT_EN = {
    "request.invalid_memory_id": "Invalid memory ID format",
    "request.identity_empty": "Identity fields must not be empty",
    "request.principal_required": "External entries must declare the caller with X-Principal-ID",
    "request.agent_required": "This operation requires an explicit agent_id",
    "request.agent_reserved": "The reserved system cannot be used as agent_id",
    "request.workspace_not_found": "The workspace does not exist or is not open",
    "request.empty_update": "The request body has no fields to update",
    "request.validation_failed": "The configuration input is invalid",
    "request.multipart_invalid": "Could not parse the multipart form",
    "request.file_required": "The file field is missing",
    "request.single_file_required": (
        "An upload request may contain only one file part; upload several files one by one"
    ),
    "request.upload_fields_invalid": "The upload request contains unknown fields",
    "request.file_content_required": "The file field must carry the file content",
    "request.idempotency_key_required": (
        "The Idempotency-Key header is missing; retries of the same file must reuse the same value"
    ),
    "request.idempotency_key_too_long": "Idempotency-Key is too long",
    "request.body_too_large": "The request body exceeds the size limit",
    "request.not_found": "The requested resource does not exist",
    "request.turn_required": (
        "Writes and updates need the current turn, which the connector fills in"
    ),
    "request.management_credential_invalid": (
        "Management endpoints need the management credential: it is missing or incorrect"
    ),
    "request.input_invalid": "The request input is invalid",
    "workspace.operation_denied": "The current identity is not allowed to perform this operation",
    "access.registration.not_found": "The access registration does not exist",
    "access.registration.conflict": (
        "The access registration was not changed: it conflicts with an existing registration, "
        "or it can only be changed in the registration file"
    ),
    "workspace.scope_required": "The request could not be processed; contact the maintainer",
    "workspace.mismatch": "The request does not match the target workspace",
    "workspace.session.not_found": "The session does not exist",
    "workspace.session.busy": "The session is busy; try again later",
    "workspace.session.capacity_exceeded": "The session capacity is full",
    "workspace.session.closed": "The session service is closed",
    "workspace.resource.not_found": "The resource does not exist or is not accessible",
    "workspace.resource.unavailable": "The service is unavailable right now; try again later",
    "memory.field.invalid": "The memory fields are invalid; check the input",
    "workspace.memory.alias_conflict": "The memory alias conflicts; choose another alias",
    "workspace.memory.intent.pending_update_not_allowed": (
        "This memory is still being generated and cannot be updated; once generation finishes, "
        "use its formal alias"
    ),
    "workspace.memory.intent.dispatch_rejected": (
        "The generation queue did not accept this submission; submit it again later"
    ),
    "workspace.memory.intent.dispatch_unknown": (
        "Could not confirm whether this submission was accepted; search later to verify before "
        "submitting it again"
    ),
    "model.not_found": "The model does not exist",
    "model.duplicate_id": "The model ID already exists",
    "provider.not_found": "The provider does not exist or cannot be deleted",
    "patchouli.topic.busy": "The topic is busy; try again later",
    "patchouli.topic.settle_rejected": (
        "The generation queue has not accepted the settlement yet; the topic content is kept "
        "and you can retry"
    ),
    "import_bus.unavailable": "The import cannot complete right now; keep the turn and resend later",
    "import_bus.apply_failed": (
        "The imported turn failed while being applied; resending routes and submits it again"
    ),
    "workspace.asset.not_found": "The attachment does not exist or has expired",
    "workspace.asset.expired": "The attachment has expired; upload it again",
    "workspace.asset.not_ready": "The attachment is not ready yet",
    "workspace.asset.failed": "Processing the attachment failed; upload it again",
    "workspace.asset.removed": "The asset of the original upload was removed; upload it again",
    "workspace.asset.stale_result": "The upload result conflicted; upload it again",
    "workspace.asset.operation_conflict": (
        "The same Idempotency-Key came with different file content or metadata"
    ),
    "workspace.asset.store_closed": "Attachment storage is unavailable right now; try again later",
    "attachment.empty": "The file is empty",
    "attachment.invalid_name": "The file name is invalid",
    "attachment.too_large": "The attachment exceeds the size limit",
    "attachment.unsupported_format": "This attachment format is not supported",
    "attachment.unsupported_format.legacy_doc_format": (
        "The .doc format is not supported; save it as .docx and upload again"
    ),
    "attachment.unsupported_format.conflicting_media_type": (
        "The declared file type conflicts with the extension; check the file format and upload again"
    ),
    "storage.offline": "Memory storage is unavailable right now",
    "storage.read_failed": "Reading memory failed; try again later",
    "storage.write_failed": (
        "The memory change was not confirmed; verify the result before submitting again"
    ),
    "admission.gateway_closed": "The authentication service is closed",
    "admission.denied": "The current identity is not admitted to this workspace",
    "http.client_error": "The request cannot be completed; check the input or access rights",
    "server.internal_error": "Internal server error; try again later",
    "validation.missing": "Required field is missing",
    "validation.extra_forbidden": "This field is not allowed",
    "validation.string_type": "A string is expected",
    "validation.int_type": "An integer is expected",
    "validation.int_parsing": "An integer is expected",
    "validation.float_type": "A number is expected",
    "validation.float_parsing": "A number is expected",
    "validation.bool_type": "A boolean is expected",
    "validation.bool_parsing": "A boolean is expected",
    "validation.list_type": "A list is expected",
    "validation.dict_type": "An object is expected",
    "validation.json_invalid": "Invalid JSON",
    "validation.uuid_parsing": "Invalid identifier format",
    "validation.string_too_short": "The text is too short",
    "validation.string_too_long": "The text is too long",
    "validation.greater_than": "The value is out of the allowed range",
    "validation.greater_than_equal": "The value is out of the allowed range",
    "validation.less_than": "The value is out of the allowed range",
    "validation.less_than_equal": "The value is out of the allowed range",
    "validation.literal_error": "The value is not one of the allowed values",
    "validation.enum": "The value is not one of the allowed values",
    "validation.value_error": "The field value is invalid",
}


def get_http_error_text(key: str, language: str | Language | None = None) -> str:
    """返回一条错误的安全说明；未知 key 抛 ``KeyError``。"""
    resolved = resolve_language(explicit=language)
    texts = _HTTP_ERROR_TEXT_EN if resolved == Language.EN else _HTTP_ERROR_TEXT_ZH
    try:
        return texts[key]
    except KeyError as exc:
        raise KeyError(f"Unknown HTTP error i18n key: {key}") from exc


__all__ = ["get_http_error_text"]
