"""WorkspaceAsset 上传用例：串行接收、原子注册与请求内解析交接。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models.identity import IdentityScope, WorkspaceIdentity
from hivememory.core.models.workspace_asset import (
    WorkspaceAssetHandle,
    WorkspaceAssetUploadReceipt,
)
from hivememory.system.config import AttachmentParserConfig
from hivememory.system.runtime.serial_gate import KeyedSerialGate
from hivememory.system.runtime.workspace.ports import WorkspaceAssetCommandPort
from hivememory.system.services.attachments.parse_service import AttachmentParseService
from hivememory.system.services.attachments.upload import (
    UPLOAD_PRODUCER,
    UPLOAD_PRODUCER_VERSION,
    SupportsAsyncRead,
    receive_upload,
)
from hivememory.workspace.access import WorkspaceOperation

if TYPE_CHECKING:
    from hivememory.workspace import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard


class WorkspaceAssetApplicationService:
    """只编排上传用例，输入规则与解析接纳由附件服务负责。

    接收 server 已冻结的 IdentityScope。同 key 的门覆盖完整请求，等待方
    只会在前次请求终态或错误收尾后进入 Store 的重放/冲突判定。

    访问边界（A1 计划第 1.1/3.3 节）：WorkspaceAsset 不属于 Patchouli，
    在自己的公共入口调用同一共享行为检查——上传绑定 ``management.asset``
    （``asset.acquire`` 只授权解析/获取，不自动授权上传）。``access``
    缺省时走 A1 第 6 节兼容清单中的既有上传 HTTP 链路受信适配，A6 完成
    生产切换后收紧。
    """

    def __init__(
        self,
        store: WorkspaceAssetCommandPort,
        parser_config: AttachmentParserConfig,
        parse_service: AttachmentParseService,
        *,
        access_guard: WorkspaceAccessGuard,
    ) -> None:
        self._store = store
        self._parser_config = parser_config
        self._parse_service = parse_service
        self._access_guard = access_guard
        self._serial_gate = KeyedSerialGate[tuple[WorkspaceIdentity, str]]()

    async def upload_asset(
        self,
        *,
        identity_scope: IdentityScope,
        file_name: str,
        declared_media_type: str | None,
        source: SupportsAsyncRead,
        client_operation_id: str,
        access: WorkspaceAccessContext | None = None,
    ) -> WorkspaceAssetUploadReceipt:
        """接收一个文件，注册资产并返回解析终态，保留首次创建/重放标记。"""
        if access is not None:
            # 行为检查先于接收与注册副作用：不允许未经许可的上传消耗
            # 解析与存储资源。
            self._access_guard.authorize_operation(access, WorkspaceOperation.MANAGEMENT_ASSET)
        key = (identity_scope.workspace_identity, client_operation_id)
        async with self._serial_gate.hold(key):
            metadata, content, content_hash = await receive_upload(
                file_name=file_name,
                declared_media_type=declared_media_type,
                source=source,
                config=self._parser_config,
            )
            receipt = self._store.register_uploaded_asset(
                identity_scope,
                metadata,
                client_operation_id,
                raw_content_object=content,
                raw_content_hash=content_hash,
                raw_producer=UPLOAD_PRODUCER,
                raw_producer_version=UPLOAD_PRODUCER_VERSION,
            )
            snapshot = await self._parse_service.parse_required_representation(
                identity_scope,
                receipt.handle,
            )
            return WorkspaceAssetUploadReceipt(
                handle=WorkspaceAssetHandle(asset_ref=receipt.handle.asset_ref, asset=snapshot),
                created=receipt.created,
            )


__all__ = ["WorkspaceAssetApplicationService"]
