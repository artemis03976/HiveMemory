"""附件上传输入校验、受限接收与 RAW 元数据构造。"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Protocol

from hivememory.core.models.workspace_asset import (
    AssetRepresentationKind,
    WorkspaceAssetMetadata,
)
from hivememory.system.config import AttachmentParserConfig

from .errors import AttachmentTooLargeError, EmptyAttachmentError, InvalidAttachmentNameError
from .formats import resolve_attachment_format

UPLOAD_PRODUCER = "upload"
UPLOAD_PRODUCER_VERSION = "1"
_READ_CHUNK_SIZE = 64 * 1024
_MAX_DISPLAY_NAME_LENGTH = 200


class SupportsAsyncRead(Protocol):
    """受限读取所需的最小文件协议（兼容 FastAPI UploadFile）。"""

    async def read(self, size: int = -1) -> bytes: ...


async def receive_upload(
    *,
    file_name: str,
    declared_media_type: str | None,
    source: SupportsAsyncRead,
    config: AttachmentParserConfig,
) -> tuple[WorkspaceAssetMetadata, bytes, str]:
    """在资产注册前完成校验与读取，返回元数据、原始字节及实际摘要。

    不信任 Content-Length；流由 transport 所有者关闭。本模块不访问 Store，
    拒绝输入时不会留下无内容的孤儿资产。
    """
    display_name = _normalize_display_name(file_name)
    format_ = resolve_attachment_format(display_name, declared_media_type)
    content, content_hash = await _receive_bounded(source, config)
    metadata = WorkspaceAssetMetadata(
        kind="document",
        display_name=display_name,
        media_type=format_.canonical_media_type,
        size_bytes=len(content),
        required_representation_kind=AssetRepresentationKind.EXTRACTED_TEXT,
    )
    return metadata, content, content_hash


def _normalize_display_name(file_name: str) -> str:
    """NFKC 规范化后去除分隔符和控制字符；名称只用于展示，不参与寻址。"""
    normalized = unicodedata.normalize("NFKC", file_name)
    cleaned = "".join(
        character
        for character in normalized
        if character not in ("/", "\\") and unicodedata.category(character) not in {"Cc", "Cf"}
    ).strip()
    if not cleaned:
        raise InvalidAttachmentNameError("文件名无效，请重命名后重新上传")
    if len(cleaned) > _MAX_DISPLAY_NAME_LENGTH:
        raise InvalidAttachmentNameError(
            f"文件名过长，请控制在 {_MAX_DISPLAY_NAME_LENGTH} 个字符以内"
        )
    return cleaned


async def _receive_bounded(
    source: SupportsAsyncRead,
    config: AttachmentParserConfig,
) -> tuple[bytes, str]:
    """按实际字节数强制 RAW 上限，摘要覆盖全部读取块。"""
    digest = hashlib.sha256()
    buffer = bytearray()
    while True:
        chunk = await source.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        if len(buffer) + len(chunk) > config.max_raw_bytes:
            raise AttachmentTooLargeError(f"文件超过大小上限（{config.max_raw_bytes} 字节）")
        buffer.extend(chunk)
        digest.update(chunk)
    if not buffer:
        raise EmptyAttachmentError("不能上传空文件")
    return bytes(buffer), digest.hexdigest()


__all__ = ["UPLOAD_PRODUCER", "UPLOAD_PRODUCER_VERSION", "SupportsAsyncRead", "receive_upload"]
