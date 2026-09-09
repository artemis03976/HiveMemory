"""WorkspaceAsset 上传应用服务。

承接 HTTP 上传入口与 WorkspaceAssetStore 之间的边界：执行身份作用域下的
输入校验、受限分块读取、实际哈希计算，并调用上传专用的原子 Store 命令。
路由不直接编排多个 Store 命令；本服务也不解析身份、不产生解析内容
（文本提取属于 W1-B/W1-C 的 parser 链路）。
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Protocol

from hivememory.core.models.identity import IdentityScope
from hivememory.core.models.workspace_asset import (
    AssetRepresentationKind,
    WorkspaceAssetMetadata,
    WorkspaceAssetUploadReceipt,
)
from hivememory.system.config import AttachmentsConfig
from hivememory.system.runtime.workspace.ports import WorkspaceAssetCommandPort
from hivememory.system.services.attachments import resolve_attachment_format

#: 首轮全部批准格式统一登记的资产 kind（计划 7.1 节）。
DOCUMENT_ASSET_KIND = "document"

#: RAW representation 的稳定 producer 身份（上传字节未被任何解析器加工）。
UPLOAD_PRODUCER = "upload"
UPLOAD_PRODUCER_VERSION = "1"

#: 分块读取的窗口大小；读取期间内存占用与该值加已累计字节成正比。
_READ_CHUNK_SIZE = 64 * 1024


class AttachmentUploadError(Exception):
    """上传请求错误基类，携带可直接展示的安全文案。

    ``message`` 不包含临时路径、原始字节或内部异常栈；HTTP 层按子类
    映射到稳定的局部状态码。
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class EmptyAttachmentError(AttachmentUploadError):
    """上传了 0 字节的空文件。"""


class InvalidAttachmentNameError(AttachmentUploadError):
    """文件名在规范化后为空、只剩控制字符或超过长度上限。"""


class AttachmentTooLargeError(AttachmentUploadError):
    """文件实际字节数超过配置的 RAW 输入硬上限。"""


class SupportsAsyncRead(Protocol):
    """受限读取所需的最小文件协议（兼容 FastAPI ``UploadFile``）。"""

    async def read(self, size: int = -1) -> bytes: ...


class WorkspaceAssetApplicationService:
    """附件上传用例的系统应用服务。

    身份入口约定（v0.6.2 收敛）：本服务接收 server 边界已经冻结的
    ``IdentityScope``，不再从文件字段或 body 推导 Workspace。所有不会
    产生副作用的校验（空文件、文件名、媒体类型、大小上限）都在创建
    asset 之前完成；读取完成后只向 Store 提交不可变 bytes、规范化
    metadata 与实际内容哈希。
    """

    def __init__(
        self,
        store: WorkspaceAssetCommandPort,
        config: AttachmentsConfig,
    ) -> None:
        self._store = store
        self._config = config

    async def upload_asset(
        self,
        *,
        identity_scope: IdentityScope,
        file_name: str,
        declared_media_type: str | None,
        source: SupportsAsyncRead,
        client_operation_id: str,
    ) -> WorkspaceAssetUploadReceipt:
        """接收一个文件并在当前 Workspace 内原子注册资产与 RAW 表示。

        同一 ``(workspace_identity, client_operation_id)`` 重放且
        fingerprint（metadata + 内容哈希）相同时返回既有资产的回执
        （``created=False``）；携带不同内容时由 Store 拒绝为操作冲突。
        """
        display_name = self._normalize_display_name(file_name)
        format_ = resolve_attachment_format(display_name, declared_media_type)
        content, content_hash = await self._receive_bounded(source)
        metadata = WorkspaceAssetMetadata(
            kind=DOCUMENT_ASSET_KIND,
            display_name=display_name,
            media_type=format_.canonical_media_type,
            size_bytes=len(content),
            required_representation_kind=AssetRepresentationKind.EXTRACTED_TEXT,
        )
        return self._store.register_uploaded_asset(
            identity_scope,
            metadata,
            client_operation_id,
            raw_content_object=content,
            raw_content_hash=content_hash,
            raw_producer=UPLOAD_PRODUCER,
            raw_producer_version=UPLOAD_PRODUCER_VERSION,
        )

    def _normalize_display_name(self, file_name: str) -> str:
        """把客户端文件名规范化为仅作展示用途的 ``display_name``。

        文件名不参与 Store 寻址或物理路径：NFKC 规范化后去除路径分隔符
        与控制字符，再拒绝规范化后为空或超过长度上限的名字。
        """
        normalized = unicodedata.normalize("NFKC", file_name)
        cleaned = "".join(
            character
            for character in normalized
            if character not in ("/", "\\") and unicodedata.category(character) not in {"Cc", "Cf"}
        ).strip()
        if not cleaned:
            raise InvalidAttachmentNameError("文件名无效，请重命名后重新上传")
        if len(cleaned) > self._config.max_display_name_length:
            raise InvalidAttachmentNameError(
                f"文件名过长，请控制在 {self._config.max_display_name_length} 个字符以内"
            )
        return cleaned

    async def _receive_bounded(self, source: SupportsAsyncRead) -> tuple[bytes, str]:
        """分块读取上传流，计算实际字节数与 SHA-256，并强制 RAW 上限。

        以实际读取字节数为准，不信任 ``Content-Length``；读取超限时立即
        中止，不产生无内容的孤儿 asset。
        """
        digest = hashlib.sha256()
        buffer = bytearray()
        while True:
            chunk = await source.read(_READ_CHUNK_SIZE)
            if not chunk:
                break
            buffer.extend(chunk)
            if len(buffer) > self._config.max_raw_bytes:
                raise AttachmentTooLargeError(
                    f"文件超过大小上限（{self._config.max_raw_bytes} 字节）"
                )
            digest.update(chunk)
        if not buffer:
            raise EmptyAttachmentError("不能上传空文件")
        return bytes(buffer), digest.hexdigest()


__all__ = [
    "DOCUMENT_ASSET_KIND",
    "AttachmentTooLargeError",
    "AttachmentUploadError",
    "EmptyAttachmentError",
    "InvalidAttachmentNameError",
    "UPLOAD_PRODUCER",
    "UPLOAD_PRODUCER_VERSION",
    "SupportsAsyncRead",
    "WorkspaceAssetApplicationService",
]
