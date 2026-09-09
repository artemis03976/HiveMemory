"""WorkspaceAsset 上传与请求内解析应用服务。

承接 HTTP 上传入口与 WorkspaceAssetStore 之间的边界：执行身份作用域下的
输入校验、受限分块读取、实际哈希计算，并调用上传专用的原子 Store 命令；
随后在**同一次请求内**完成 W1-B parser 的接纳（register/start/parse/
complete-or-fail），直到 required representation 进入 READY/FAILED 终态
才返回。路由不直接编排多个 Store 命令；本服务也不解析身份。

解析在 Store 锁外通过标准线程转交执行；同一
``(workspace_identity, client_operation_id)`` 的并发上传请求在应用服务
入口按 operation key 进程内串行化，任何上传响应只会是终态快照或稳定
错误，不会出现 PROCESSING 中间快照（计划 8.2 节）。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from hivememory.core.errors import StaleAssetResultError, WorkspaceDomainError
from hivememory.core.models.identity import IdentityScope, WorkspaceIdentity
from hivememory.core.models.workspace_asset import (
    AssetRepresentation,
    AssetRepresentationKind,
    AssetRepresentationState,
    AssetSafeError,
    WorkspaceAsset,
    WorkspaceAssetHandle,
    WorkspaceAssetMetadata,
    WorkspaceAssetState,
    WorkspaceAssetUploadReceipt,
)
from hivememory.system.config import AttachmentsConfig
from hivememory.system.runtime.workspace.ports import WorkspaceAssetCommandPort
from hivememory.system.services.attachments import (
    CONTENT_UNREADABLE,
    EXECUTION_FAILURE,
    RESOURCE_LIMIT,
    AttachmentParseError,
    AttachmentParser,
    ParsedAttachmentContent,
    ParseLimits,
    resolve_attachment_format,
    resolve_parser,
)

#: 首轮全部批准格式统一登记的资产 kind（计划 7.1 节）。
DOCUMENT_ASSET_KIND = "document"

#: RAW representation 的稳定 producer 身份（上传字节未被任何解析器加工）。
UPLOAD_PRODUCER = "upload"
UPLOAD_PRODUCER_VERSION = "1"

#: 解析失败的公共终态错误码；具体原因只作为文案选择依据（计划 8.4 节）。
ASSET_FAILED_CODE = "workspace.asset.failed"

#: execution_failure 与意外异常统一返回的通用安全文案。
_GENERIC_PARSE_FAILURE_MESSAGE = "附件解析失败，请重新上传"

#: 分块读取的窗口大小；读取期间内存占用与该值加已累计字节成正比。
_READ_CHUNK_SIZE = 64 * 1024

logger = logging.getLogger(__name__)


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


@dataclass
class _OperationGate:
    """同 key 上传请求的进程内互斥门。

    ``waiters`` 记录已登记的等待者数量；锁释放且无人等待时把门从注册表
    回收，避免 operation key 无限累积。
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.waiters = 0


class WorkspaceAssetApplicationService:
    """附件上传与请求内解析的系统应用服务。

    身份入口约定（v0.6.2 收敛）：本服务接收 server 边界已经冻结的
    ``IdentityScope``，不再从文件字段或 body 推导 Workspace。所有不会
    产生副作用的校验（空文件、文件名、媒体类型、大小上限）都在创建
    asset 之前完成；RAW 注册成功后发生的解析失败在同一响应中以
    ``asset.state=failed`` 与安全摘要返回，不改写为"上传失败"。
    """

    def __init__(
        self,
        store: WorkspaceAssetCommandPort,
        config: AttachmentsConfig,
        parser_factory: Callable[[str], AttachmentParser] | None = None,
    ) -> None:
        self._store = store
        self._config = config
        self._parser_factory = parser_factory or resolve_parser
        self._gates: dict[tuple[WorkspaceIdentity, str], _OperationGate] = {}
        self._gate_guard = asyncio.Lock()

    async def upload_asset(
        self,
        *,
        identity_scope: IdentityScope,
        file_name: str,
        declared_media_type: str | None,
        source: SupportsAsyncRead,
        client_operation_id: str,
    ) -> WorkspaceAssetUploadReceipt:
        """接收一个文件，原子注册资产并在请求内完成解析直到终态。

        同一 ``(workspace_identity, client_operation_id)`` 的并发请求在
        入口按 operation key 串行化：等待方在持有方到达终态或错误收尾后
        继续执行，随后命中既有重放或冲突路径；锁 key 只包含 Workspace
        与 operation identity，不包含文件内容。
        """
        key = (identity_scope.workspace_identity, client_operation_id)
        gate = await self._gate_for(key)
        try:
            async with gate.lock:
                return await self._upload_locked(
                    identity_scope,
                    file_name=file_name,
                    declared_media_type=declared_media_type,
                    source=source,
                    client_operation_id=client_operation_id,
                )
        finally:
            async with self._gate_guard:
                gate.waiters -= 1
                if gate.waiters <= 0 and not gate.lock.locked():
                    self._gates.pop(key, None)

    async def _gate_for(self, key: tuple[WorkspaceIdentity, str]) -> _OperationGate:
        """取（或建）operation key 对应的串行化门并登记等待。

        注册表临界区内没有任何 await，单事件循环内天然原子。
        """
        async with self._gate_guard:
            gate = self._gates.get(key)
            if gate is None:
                gate = _OperationGate()
                self._gates[key] = gate
            gate.waiters += 1
            return gate

    async def _upload_locked(
        self,
        identity_scope: IdentityScope,
        *,
        file_name: str,
        declared_media_type: str | None,
        source: SupportsAsyncRead,
        client_operation_id: str,
    ) -> WorkspaceAssetUploadReceipt:
        """在持有 operation 门期间执行校验、注册与请求内解析。"""
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
        receipt = self._store.register_uploaded_asset(
            identity_scope,
            metadata,
            client_operation_id,
            raw_content_object=content,
            raw_content_hash=content_hash,
            raw_producer=UPLOAD_PRODUCER,
            raw_producer_version=UPLOAD_PRODUCER_VERSION,
        )
        return await self._accept_required_representation(
            identity_scope,
            receipt,
            format_.canonical_media_type,
            received_raw_hash=content_hash,
        )

    # ------------------------------------------------------------------
    # 请求内解析交接（计划 8.2 节）
    # ------------------------------------------------------------------

    async def _accept_required_representation(
        self,
        identity_scope: IdentityScope,
        receipt: WorkspaceAssetUploadReceipt,
        media_type: str,
        received_raw_hash: str,
    ) -> WorkspaceAssetUploadReceipt:
        """把 PROCESSING 文档资产推进到 EXTRACTED_TEXT READY/FAILED 终态。

        终态重放直接返回当前快照，不重复 register/start/parse；解析输入
        来自 ``start_representation()`` 返回的冻结快照，不调用只接受
        READY asset 的 reader。
        """
        snapshot = receipt.handle.asset
        if snapshot.state in {WorkspaceAssetState.READY, WorkspaceAssetState.FAILED}:
            return receipt

        asset_ref = receipt.handle.asset_ref
        parser = self._parser_factory(media_type)
        pending = self._store.register_representation(
            identity_scope,
            asset_ref,
            kind=AssetRepresentationKind.EXTRACTED_TEXT,
            producer=parser.producer,
            producer_version=parser.producer_version,
        )
        target_id = next(
            representation.representation_id
            for representation in pending.representations
            if representation.kind == AssetRepresentationKind.EXTRACTED_TEXT
        )
        processing = self._store.start_representation(identity_scope, asset_ref, target_id)

        # token 在 raw/target 核对成功前不可用；核对失败的内部故障无法提交
        # 失败终态，只能原样上抛（Store 不变量保证正常路径不会走到）。
        target: AssetRepresentation | None = None
        try:
            raw, target = self._verify_raw_snapshot(processing, received_raw_hash)
            # Store 锁外执行 parser；CPU 密集时经标准线程转交避免阻塞事件循环。
            result = await asyncio.to_thread(
                parser.parse,
                raw.content_object,
                limits=self._parse_limits(),
                source_raw_revision=raw.revision,
                source_raw_hash=raw.content_hash,
            )
            self._verify_result_source(result, raw.revision, raw.content_hash, parser)
        except asyncio.CancelledError:
            # 外层等待被取消：在可执行的收尾点以原 token 提交安全失败；
            # 无法提交时保留 Store 的既有状态错误，不通过新任务补偿。
            if target is not None:
                self._submit_failure(
                    identity_scope,
                    asset_ref,
                    target_id,
                    target.parse_operation_id,
                    _GENERIC_PARSE_FAILURE_MESSAGE,
                )
            raise
        except AttachmentParseError as exc:
            if target is None:
                raise
            message = (
                exc.message
                if exc.category in {CONTENT_UNREADABLE, RESOURCE_LIMIT}
                else _GENERIC_PARSE_FAILURE_MESSAGE
            )
            return self._finish_as_failed(
                identity_scope,
                receipt,
                snapshot.asset_id,
                asset_ref,
                target_id,
                target.parse_operation_id,
                message,
            )
        except Exception as exc:
            # parser 异常或结果不符合契约：完整 cause 只进入受控日志。
            if target is None:
                raise
            logger.warning("附件解析发生内部错误：%r", exc)
            return self._finish_as_failed(
                identity_scope,
                receipt,
                snapshot.asset_id,
                asset_ref,
                target_id,
                target.parse_operation_id,
                _GENERIC_PARSE_FAILURE_MESSAGE,
            )

        ready_snapshot = self._store.complete_representation(
            identity_scope,
            asset_ref,
            target_id,
            target.parse_operation_id,
            content_object=result.content_object,
            content_hash=result.content_hash,
        )
        return self._receipt_with_snapshot(receipt, ready_snapshot)

    def _finish_as_failed(
        self,
        identity_scope: IdentityScope,
        receipt: WorkspaceAssetUploadReceipt,
        asset_id: str,
        asset_ref,
        representation_id: str,
        operation_token: str | None,
        message: str,
    ) -> WorkspaceAssetUploadReceipt:
        """提交解析失败终态；竞态拒绝按既有 Store 错误收尾。

        - ``removed``/Store 关闭：向上传播，由 HTTP 边界映射 410/503；
        - ``stale``：终态已由其他提交决定，以当前权威快照结束本次请求；
        - 不换 token 重试，也不把拒绝包装成文档损坏。
        """
        try:
            failed = self._store.fail_representation(
                identity_scope,
                asset_ref,
                representation_id,
                operation_token,
                safe_error=AssetSafeError(code=ASSET_FAILED_CODE, message=message),
            )
        except StaleAssetResultError:
            current = next(
                (
                    handle.asset
                    for handle in self._store.list_workspace_assets(identity_scope)
                    if handle.asset.asset_id == asset_id
                ),
                None,
            )
            return self._receipt_with_snapshot(receipt, current or receipt.handle.asset)
        return self._receipt_with_snapshot(receipt, failed)

    @staticmethod
    def _receipt_with_snapshot(
        receipt: WorkspaceAssetUploadReceipt,
        snapshot: WorkspaceAsset,
    ) -> WorkspaceAssetUploadReceipt:
        """以上传后最新权威快照重建回执，保持 created 标记不变。"""
        return WorkspaceAssetUploadReceipt(
            handle=WorkspaceAssetHandle(
                asset_ref=receipt.handle.asset_ref,
                asset=snapshot,
            ),
            created=receipt.created,
        )

    def _verify_raw_snapshot(
        self,
        snapshot: WorkspaceAsset,
        received_raw_hash: str,
    ) -> tuple[AssetRepresentation, AssetRepresentation]:
        """核对冻结快照的 RAW kind/state、bytes 类型、大小与内容哈希。

        实际读取的 RAW content hash 与接收阶段计算的摘要不符属于内部
        交接故障（计划 7.7 节），按 ``execution_failure`` 收尾。
        """
        raw = next(
            (
                representation
                for representation in snapshot.representations
                if representation.kind == AssetRepresentationKind.RAW
            ),
            None,
        )
        target = next(
            (
                representation
                for representation in snapshot.representations
                if representation.kind == AssetRepresentationKind.EXTRACTED_TEXT
            ),
            None,
        )
        if (
            raw is None
            or target is None
            or raw.state != AssetRepresentationState.READY
            or not isinstance(raw.content_object, bytes)
            or raw.content_hash != received_raw_hash
            or len(raw.content_object) != snapshot.size_bytes
        ):
            raise AttachmentParseError(
                EXECUTION_FAILURE,
                _GENERIC_PARSE_FAILURE_MESSAGE,
                params={"reason": "raw_snapshot_mismatch"},
            )
        return raw, target

    @staticmethod
    def _verify_result_source(
        result: ParsedAttachmentContent,
        raw_revision: int,
        raw_content_hash: str,
        parser: AttachmentParser,
    ) -> None:
        """核对解析结果确实来自本次冻结 RAW，且 producer/version 未漂移。"""
        if (
            result.source_raw != {"revision": raw_revision, "content_hash": raw_content_hash}
            or result.producer != parser.producer
            or result.producer_version != parser.producer_version
        ):
            raise AttachmentParseError(
                EXECUTION_FAILURE,
                _GENERIC_PARSE_FAILURE_MESSAGE,
                params={"reason": "result_source_mismatch"},
            )

    def _submit_failure(
        self,
        identity_scope: IdentityScope,
        asset_ref,
        representation_id: str,
        operation_token: str | None,
        message: str,
    ) -> None:
        """以原 token 提交 ``workspace.asset.failed`` 安全终态（取消收尾专用）。

        取消路径中 Store 拒绝（stale/removed/closed）时保留既有状态错误：
        资产终态已由其他提交决定或 Store 不再可用，不换 token 重试。
        """
        try:
            self._store.fail_representation(
                identity_scope,
                asset_ref,
                representation_id,
                operation_token,
                safe_error=AssetSafeError(code=ASSET_FAILED_CODE, message=message),
            )
        except WorkspaceDomainError as exc:
            logger.warning("解析失败提交被 Store 拒绝：%r", exc)

    def _parse_limits(self) -> ParseLimits:
        """把 System 配置映射为本次解析固定的资源限制。"""
        return ParseLimits(
            max_raw_bytes=self._config.max_raw_bytes,
            max_extracted_text_bytes=self._config.max_extracted_text_bytes,
            max_canonical_content_bytes=self._config.max_canonical_content_bytes,
            max_locator_count=self._config.max_locator_count,
            max_docx_members=self._config.max_docx_members,
            max_docx_member_uncompressed_bytes=self._config.max_docx_member_uncompressed_bytes,
            max_docx_package_uncompressed_bytes=self._config.max_docx_package_uncompressed_bytes,
            max_docx_compression_ratio=self._config.max_docx_compression_ratio,
            max_xml_depth=self._config.max_xml_depth,
            max_xml_nodes=self._config.max_xml_nodes,
            parse_budget_seconds=self._config.parse_budget_seconds,
        )

    # ------------------------------------------------------------------
    # W1-A 输入校验与受限接收
    # ------------------------------------------------------------------

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
    "ASSET_FAILED_CODE",
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
