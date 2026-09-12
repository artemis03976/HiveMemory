"""接纳附件 RAW，在请求内完成 required representation 的解析与提交。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable

from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models.identity import IdentityScope
from hivememory.core.models.workspace_asset import (
    AssetRepresentation,
    AssetRepresentationKind,
    AssetRepresentationState,
    AssetSafeError,
    WorkspaceAsset,
    WorkspaceAssetHandle,
    WorkspaceAssetRef,
    WorkspaceAssetState,
)
from hivememory.system.config import AttachmentParserConfig
from hivememory.system.runtime.workspace.ports import WorkspaceAssetCommandPort

from .errors import CONTENT_UNREADABLE, EXECUTION_FAILURE, RESOURCE_LIMIT, AttachmentParseError
from .models import ParsedAttachmentContent
from .parser import AttachmentParser, resolve_parser

ASSET_FAILED_CODE = "workspace.asset.failed"
_GENERIC_PARSE_FAILURE_MESSAGE = "附件解析失败，请重新上传"
logger = logging.getLogger(__name__)


class AttachmentParseService:
    """拥有解析交接逻辑，状态与 token 的有效性仍由 Store 原子校验。

    输入是注册后的 handle；解析只读取 Store 命令返回的冻结快照，不依赖
    HTTP 回执、上传流或只接受 READY 资产的 reader。调用方负责同上传操作
    的串行化，本服务不另存运行记录。
    """

    def __init__(
        self,
        store: WorkspaceAssetCommandPort,
        config: AttachmentParserConfig,
        parser_factory: Callable[[str], AttachmentParser] | None = None,
    ) -> None:
        self._store = store
        self._config = config
        self._parser_factory = parser_factory or resolve_parser

    async def parse_required_representation(
        self,
        identity_scope: IdentityScope,
        handle: WorkspaceAssetHandle,
    ) -> WorkspaceAsset:
        """返回 READY/FAILED 快照，或传播 Store 冲突、移除、关闭等错误。"""
        if handle.asset.state in {WorkspaceAssetState.READY, WorkspaceAssetState.FAILED}:
            return handle.asset

        asset_ref = handle.asset_ref
        parser = self._parser_factory(handle.asset.media_type)
        pending = self._store.register_representation(
            identity_scope,
            asset_ref,
            kind=AssetRepresentationKind.EXTRACTED_TEXT,
            producer=parser.producer,
            producer_version=parser.producer_version,
        )
        target_id = next(
            item.representation_id
            for item in pending.representations
            if item.kind == AssetRepresentationKind.EXTRACTED_TEXT
        )
        processing = self._store.start_representation(identity_scope, asset_ref, target_id)
        target = next(
            item for item in processing.representations if item.representation_id == target_id
        )
        try:
            raw = _verify_raw_snapshot(processing)
            # 在线程中运行以让出事件循环；不持有 Store 锁，不引入后台任务队列。
            result = await asyncio.to_thread(
                parser.parse,
                raw.content_object,
                config=self._config,
                source_raw_revision=raw.revision,
                source_raw_hash=raw.content_hash,
            )
            _verify_result_source(result, raw, parser)
        except asyncio.CancelledError:
            # 原 token 尽力收尾；Store 拒绝不能替换请求取消，也不另起任务重试。
            try:
                self._fail(identity_scope, asset_ref, target, _GENERIC_PARSE_FAILURE_MESSAGE)
            except WorkspaceDomainError as exc:
                logger.warning("解析取消收尾被 Store 拒绝：%r", exc)
            raise
        except AttachmentParseError as exc:
            message = (
                exc.message
                if exc.category in {CONTENT_UNREADABLE, RESOURCE_LIMIT}
                else _GENERIC_PARSE_FAILURE_MESSAGE
            )
            return self._fail(identity_scope, asset_ref, target, message)
        except Exception as exc:
            logger.warning("附件解析发生内部错误：%r", exc)
            return self._fail(identity_scope, asset_ref, target, _GENERIC_PARSE_FAILURE_MESSAGE)

        # Store 提交错误必须原样传播，不能被转换成解析失败或旧快照。
        return self._store.complete_representation(
            identity_scope,
            asset_ref,
            target.representation_id,
            target.parse_operation_id,
            content_object=result.content_object,
            content_hash=result.content_hash,
        )

    def _fail(
        self,
        identity_scope: IdentityScope,
        asset_ref: WorkspaceAssetRef,
        target: AssetRepresentation,
        message: str,
    ) -> WorkspaceAsset:
        """以原目标 token 提交安全失败；过期、移除或关闭均由 Store 拒绝。"""
        return self._store.fail_representation(
            identity_scope,
            asset_ref,
            target.representation_id,
            target.parse_operation_id,
            safe_error=AssetSafeError(code=ASSET_FAILED_CODE, message=message),
        )


def _verify_raw_snapshot(snapshot: WorkspaceAsset) -> AssetRepresentation:
    """直接校验 Store RAW 的实际字节、摘要和大小，不再另传上传摘要副本。"""
    raw = next(
        (item for item in snapshot.representations if item.kind == AssetRepresentationKind.RAW),
        None,
    )
    if (
        raw is None
        or raw.state != AssetRepresentationState.READY
        or not isinstance(raw.content_object, bytes)
        or len(raw.content_object) != snapshot.size_bytes
        or hashlib.sha256(raw.content_object).hexdigest() != raw.content_hash
    ):
        raise AttachmentParseError(
            EXECUTION_FAILURE,
            _GENERIC_PARSE_FAILURE_MESSAGE,
            params={"reason": "raw_snapshot_mismatch"},
        )
    return raw


def _verify_result_source(
    result: ParsedAttachmentContent,
    raw: AssetRepresentation,
    parser: AttachmentParser,
) -> None:
    """结果必须来自本次 RAW 与 parser，不能接纳漂移的来源。"""
    if (
        result.source_raw != {"revision": raw.revision, "content_hash": raw.content_hash}
        or result.producer != parser.producer
        or result.producer_version != parser.producer_version
    ):
        raise AttachmentParseError(
            EXECUTION_FAILURE,
            _GENERIC_PARSE_FAILURE_MESSAGE,
            params={"reason": "result_source_mismatch"},
        )


__all__ = ["ASSET_FAILED_CODE", "AttachmentParseService"]
