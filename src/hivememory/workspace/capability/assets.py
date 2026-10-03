"""WorkspaceAsset 能力：上传用例（A2 §1.2，自 ``system/application`` 原样迁入）。

串行接收、原子注册与请求内解析交接；AssetStore 与解析交接位于 ``workspace.assets``，
文件格式解析器位于 ``infrastructure.attachments``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.components.serial_gate import KeyedSerialGate
from hivememory.config.attachments import AttachmentParserConfig
from hivememory.core.access import WorkspaceOperation
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models.identity import IdentityScope, WorkspaceIdentity
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
    from hivememory.workspace.access import WorkspaceAccessGuard


class WorkspaceAssetApplicationService:
    """只编排上传用例，输入规则与解析接纳由附件服务负责。

    接收 server 已冻结的 IdentityScope。同 key 的门覆盖完整请求，等待方
    只会在前次请求终态或错误收尾后进入 Store 的重放/冲突判定。

    访问边界（A1 访问边界返工第 4.3/4.5 节）：WorkspaceAsset 不属于
    Patchouli，在自己的公共入口调用同一共享行为检查——上传绑定
    ``management.asset``（``asset.acquire`` 只授权解析/获取，不自动授权
    上传），行为检查先于接收与注册副作用。``access`` 为统一认证网关签发
    的可信 context，必须提供；传入的 ``identity_scope`` 只作一致性校验，
    与 context 的可信坐标不一致时在副作用前拒绝（P-1 缺陷修复：跨
    Workspace、跨 owner 或替换 Actor 的 scope 冲突不再放行）。
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
        access: WorkspaceAccessContext,
    ) -> WorkspaceAssetUploadReceipt:
        """接收一个文件，注册资产并返回解析终态，保留首次创建/重放标记。"""
        # 行为检查先于接收与注册副作用：不允许未经许可的上传消耗解析与
        # 存储资源。
        authorized_scope = self._access_guard.authorize_operation(
            access, WorkspaceOperation.MANAGEMENT_ASSET
        )
        # 可信 context 是认证后操作范围的依据，另行传入的 scope 只能用于
        # 一致性校验；冲突在读取上传内容之前拒绝，不产生任何副作用。
        if identity_scope != authorized_scope:
            raise WorkspaceMismatchError(
                details={
                    "reason": "request_scope_mismatches_access_context",
                    "access_workspace_id": authorized_scope.workspace_identity.workspace_id,
                    "request_workspace_id": identity_scope.workspace_identity.workspace_id,
                }
            )
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


__all__ = ["WorkspaceAssetApplicationService"]
