"""root WorkspaceAsset 上传 API：授权后接收、原子注册并交接附件解析。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.components.serial_gate import KeyedSerialGate
from hivememory.config.attachments import AttachmentParserConfig
from hivememory.core.models import WorkspaceIdentity
from hivememory.core.models.workspace_asset import (
    WorkspaceAssetHandle,
    WorkspaceAssetUploadReceipt,
)
from hivememory.core.ports.workspace_assets import WorkspaceAssetCommandPort
from hivememory.workspace.assets.parse_service import AttachmentParseService
from hivememory.workspace.assets.upload import (
    UPLOAD_PRODUCER,
    UPLOAD_PRODUCER_VERSION,
    SupportsAsyncRead,
    receive_upload,
)

if TYPE_CHECKING:
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer


class RootWorkspaceAssetService:
    """只向 system 提供上传编排，输入规则与解析接纳由附件服务负责。

    同 key 的门覆盖完整请求，等待方在前次请求终态或错误收尾后进入
    Store 的重放/冲突判定；授权先于文件接收，scope 仅在本次上传内流动。
    """

    def __init__(
        self,
        store: WorkspaceAssetCommandPort,
        parser_config: AttachmentParserConfig,
        parse_service: AttachmentParseService,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._store = store
        self._parser_config = parser_config
        self._parse_service = parse_service
        self._authorizer = operation_authorizer
        self._serial_gate = KeyedSerialGate[tuple[WorkspaceIdentity, str]]()

    async def upload_asset(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        file_name: str,
        declared_media_type: str | None,
        source: SupportsAsyncRead,
        client_operation_id: str,
        access: WorkspaceAccessContext,
    ) -> WorkspaceAssetUploadReceipt:
        """接收一个文件，注册资产并返回解析终态，保留首次创建/重放标记。"""
        # 行为检查先于接收与注册副作用：不允许未经许可的上传消耗解析与
        # 存储资源；后续全部使用操作授权者返回的可信 scope。
        authorized_scope = self._authorizer.authorize_root(access, target_workspace)
        key = (authorized_scope.workspace_identity, client_operation_id)
        async with self._serial_gate.hold(key):
            metadata, content, content_hash = await receive_upload(
                file_name=file_name,
                declared_media_type=declared_media_type,
                source=source,
                config=self._parser_config,
            )
            receipt = self._store.register_uploaded_asset(
                authorized_scope,
                metadata,
                client_operation_id,
                raw_content_object=content,
                raw_content_hash=content_hash,
                raw_producer=UPLOAD_PRODUCER,
                raw_producer_version=UPLOAD_PRODUCER_VERSION,
            )
            snapshot = await self._parse_service.parse_required_representation(
                authorized_scope,
                receipt.handle,
            )
            return WorkspaceAssetUploadReceipt(
                handle=WorkspaceAssetHandle(asset_ref=receipt.handle.asset_ref, asset=snapshot),
                created=receipt.created,
            )


__all__ = ["RootWorkspaceAssetService"]
