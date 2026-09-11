"""WorkspaceAsset 上传路由 — Chat 附件的 HTTP 入口。

路由只负责 HTTP 解析、依赖注入和错误翻译：multipart 严格校验（单文件、
拒绝额外业务字段）、``Idempotency-Key`` 提取与身份冻结都在这里完成，
资产注册、受限读取与哈希计算由上传应用服务承担。
"""

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException

from hivememory.core.errors import (
    AssetNotFoundError,
    AssetOperationConflictError,
    AssetRemovedError,
    StaleAssetResultError,
)
from hivememory.core.models import IdentityScope
from hivememory.server.deps import (
    RequestIdentitySelection,
    get_identity_selection,
    get_workspace_asset_service,
    resolve_request_identity_scope,
)
from hivememory.server.models.workspace_asset import WorkspaceAssetUploadResponse
from hivememory.system.application.workspace_asset_service import (
    AttachmentTooLargeError,
    EmptyAttachmentError,
    InvalidAttachmentNameError,
    WorkspaceAssetApplicationService,
)
from hivememory.system.services.attachments import UnsupportedAttachmentFormatError

router = APIRouter(tags=["workspace-assets"])

#: operation identity 的长度上限；与 Store 的幂等索引键长度约束一致。
_MAX_OPERATION_ID_LENGTH = 200

#: starlette 表单解析的部件数量界限；业务合法性仍由下方严格校验决定。
_MAX_FORM_PARTS = 8


async def _parse_upload_form(request: Request) -> list[UploadFile]:
    """解析 multipart 表单并执行单文件契约校验。

    契约（计划 A1 节）：文件字段名为 ``file``，一个请求只处理一个文件；
    重复的 ``file`` part、多文件 part 或额外业务字段按非法请求拒绝，
    不能只取第一份文件而忽略其余内容。所有拒绝路径都会关闭已解析的
    上传流及框架临时文件。
    """
    try:
        form = await request.form(max_files=_MAX_FORM_PARTS, max_fields=_MAX_FORM_PARTS)
    except MultiPartException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"multipart 表单解析失败：{exc}",
        ) from exc

    file_parts = form.getlist("file")
    uploads = [part for part in file_parts if isinstance(part, UploadFile)]
    extra_fields = sorted({key for key in form.keys() if key != "file"})
    try:
        if not file_parts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="缺少 file 文件字段",
            )
        if len(file_parts) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="一个上传请求只能包含一个 file 文件部分，多文件请逐个上传",
            )
        if extra_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"上传请求包含未知的业务字段：{', '.join(extra_fields)}",
            )
        if not uploads:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="file 字段必须携带文件内容",
            )
    except Exception:
        for upload in uploads:
            await upload.close()
        raise
    return uploads


@router.post("/workspace/assets", response_model=WorkspaceAssetUploadResponse, status_code=201)
async def upload_workspace_asset(
    request: Request,
    response: Response,
    idempotency_key: str | None = Header(default=None),
    selection: RequestIdentitySelection = Depends(get_identity_selection),
    service: WorkspaceAssetApplicationService = Depends(get_workspace_asset_service),
) -> WorkspaceAssetUploadResponse:
    """上传单个附件，创建 WorkspaceAsset 并返回 bound ref 与 RAW 摘要。

    首次创建返回 201；同一 ``Idempotency-Key`` 且内容一致的重放返回 200
    和同一逻辑资产的当前快照。上传成功只表示 RAW 已注册，不把附件自动
    加入当前 Chat run。
    """
    identity_scope: IdentityScope = resolve_request_identity_scope(selection)

    operation_id = idempotency_key.strip() if idempotency_key else ""
    if not operation_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少 Idempotency-Key 请求头；同一文件重试必须沿用同一取值",
        )
    if len(operation_id) > _MAX_OPERATION_ID_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Idempotency-Key 过长",
        )

    uploads = await _parse_upload_form(request)
    try:
        upload = uploads[0]
        receipt = await service.upload_asset(
            identity_scope=identity_scope,
            file_name=upload.filename or "",
            declared_media_type=upload.content_type,
            source=upload,
            client_operation_id=operation_id,
        )
    except EmptyAttachmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    except InvalidAttachmentNameError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    except AttachmentTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=exc.message,
        ) from exc
    except UnsupportedAttachmentFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=exc.args[0],
        ) from exc
    except AssetRemovedError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="原上传操作对应的资产已被移除，请重新上传",
        ) from exc
    except StaleAssetResultError as exc:
        # complete/fail 与 remove 等竞态中已有有效提交决定终态；结束本次请求。
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="上传结果提交冲突，请重新上传",
        ) from exc
    except AssetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="附件不存在或已失效",
        ) from exc
    except AssetOperationConflictError as exc:
        if exc.details.get("reason") == "store_closed":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="附件存储当前不可用，请稍后重试",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="同一 Idempotency-Key 携带了不一致的文件内容或元数据",
        ) from exc
    finally:
        # 所有路径（含成功与拒绝）都关闭上传流及框架临时文件。
        for upload in uploads:
            await upload.close()

    if not receipt.created:
        # 幂等重放：复用同一 DTO 结构返回当前快照，HTTP 状态取 200。
        response.status_code = status.HTTP_200_OK
    return WorkspaceAssetUploadResponse.from_receipt(receipt)


__all__ = ["router"]
