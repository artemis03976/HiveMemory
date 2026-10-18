"""WorkspaceAsset 上传路由 — Chat 附件的 HTTP 入口。

路由只负责 HTTP 解析、依赖注入和错误翻译：multipart 严格校验（单文件、
拒绝额外业务字段）、``Idempotency-Key`` 提取与身份冻结都在这里完成，
上传用例由应用服务编排，输入校验与解析交给附件服务。
"""

from fastapi import APIRouter, Depends, Header, Request, Response, status
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException

from hivememory.server.deps import (
    RequestAccess,
    get_request_access,
    get_root_workspace_asset_service,
)
from hivememory.server.error_handlers import APIError, error_responses
from hivememory.server.models.workspace_asset import WorkspaceAssetUploadResponse
from hivememory.workspace.capability.root.assets import (
    RootWorkspaceAssetService,
)

router = APIRouter(
    responses=error_responses(400, 403, 404, 409, 410, 413, 415, 422, 500, 503),
    tags=["workspace-assets"],
)

#: operation identity 的长度上限；与 Store 的幂等索引键长度约束一致。
_MAX_OPERATION_ID_LENGTH = 200

#: starlette 表单解析的部件数量界限；业务合法性仍由下方严格校验决定。
_MAX_FORM_PARTS = 8


async def _parse_upload_form(request: Request) -> list[UploadFile]:
    """解析 multipart 表单并执行单文件契约校验。

    契约：文件字段名为 ``file``，一个请求只处理一个文件；
    重复的 ``file`` part、多文件 part 或额外业务字段按非法请求拒绝，
    不能只取第一份文件而忽略其余内容。所有拒绝路径都会关闭已解析的
    上传流及框架临时文件。
    """
    try:
        form = await request.form(max_files=_MAX_FORM_PARTS, max_fields=_MAX_FORM_PARTS)
    except MultiPartException as exc:
        raise APIError("request.multipart_invalid") from exc
    except HTTPException as exc:
        # Request 已把表单解析失败翻译为 HTTP 400；此处仍使用具名传输类别。
        if exc.status_code != 400:
            raise
        raise APIError("request.multipart_invalid") from exc

    file_parts = form.getlist("file")
    uploads = [part for part in file_parts if isinstance(part, UploadFile)]
    extra_fields = sorted({key for key in form.keys() if key != "file"})
    try:
        if not file_parts:
            raise APIError("request.file_required")
        if len(file_parts) > 1:
            raise APIError("request.single_file_required")
        if extra_fields:
            raise APIError("request.upload_fields_invalid")
        if not uploads:
            raise APIError("request.file_content_required")
    except Exception:
        # 未知字段也可能是文件；关闭整个表单才能释放全部临时文件。
        await form.close()
        raise
    return uploads


@router.post("/workspace/assets", response_model=WorkspaceAssetUploadResponse, status_code=201)
async def upload_workspace_asset(
    request: Request,
    response: Response,
    idempotency_key: str | None = Header(default=None),
    request_access: RequestAccess = Depends(get_request_access),
    service: RootWorkspaceAssetService = Depends(get_root_workspace_asset_service),
) -> WorkspaceAssetUploadResponse:
    """上传单个附件，创建 WorkspaceAsset 并返回 bound ref 与 RAW 摘要。

    首次创建返回 201；同一 ``Idempotency-Key`` 且内容一致的重放返回 200
    和同一逻辑资产的当前快照。上传成功只表示 RAW 已注册，不把附件自动
    加入当前 Chat run。上传经 root API：请求经统一认证网关取得保留的
    system actor 访问 context，注册使用的 scope 由授权者组装，调用方不能
    另行传入，自报的 scope 无法越过授权。
    """
    operation_id = idempotency_key.strip() if idempotency_key else ""
    if not operation_id:
        raise APIError("request.idempotency_key_required")
    if len(operation_id) > _MAX_OPERATION_ID_LENGTH:
        raise APIError("request.idempotency_key_too_long")

    uploads = await _parse_upload_form(request)
    try:
        upload = uploads[0]
        receipt = await service.upload_asset(
            target_workspace=request_access.target_workspace,
            file_name=upload.filename or "",
            declared_media_type=upload.content_type,
            source=upload,
            client_operation_id=operation_id,
            access=request_access.access,
        )
    finally:
        # 所有路径（含成功与拒绝）都关闭上传流及框架临时文件。
        for upload in uploads:
            await upload.close()

    if not receipt.created:
        # 幂等重放：复用同一 DTO 结构返回当前快照，HTTP 状态取 200。
        response.status_code = status.HTTP_200_OK
    return WorkspaceAssetUploadResponse.from_receipt(receipt)


__all__ = ["router"]
