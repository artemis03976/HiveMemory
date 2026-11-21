"""记忆生成的来源 Artifact：结算时的交互原文与话题附件的文档快照。

两者都在生成任务里写下，但写的是交互与文档 Artifact，不写记忆；记忆的唯一写入者仍是
``MemoryGenerationFamiliar``。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models import WorkspaceIdentity
from hivememory.core.models.artifact import ArtifactRef, DocumentSourceType
from hivememory.core.models.workspace_asset import TopicAssetBinding
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.patchouli.control.memory_generation.models import InteractionArtifactInput

if TYPE_CHECKING:
    from hivememory.engines.artifacts.engine import ArtifactEngine

logger = logging.getLogger(__name__)


class SourceArtifactRecorder:
    """为一次生成保存它的来源：话题交互的原文快照，以及话题附件提升成的文档 Artifact。"""

    def __init__(
        self,
        *,
        artifact_engine: ArtifactEngine,
        asset_reader: WorkspaceAssetReaderPort | None,
    ) -> None:
        self._artifact_engine = artifact_engine
        # 附件 promotion 的只读 reader（assembler 经 PatchouliRuntime
        # 注入进程级唯一 Store）；ref 失效或 Store 关闭时 best-effort 降级。
        self._asset_reader = asset_reader

    async def capture_interaction(
        self,
        interaction_input: InteractionArtifactInput | None,
        belong_to: WorkspaceIdentity,
    ) -> ArtifactRef | None:
        """
        构建原始交互 artifact：结算时覆盖话题的全部轮次与折叠记录。

        在生成之前捕获，失败只记录 warning、生成照常进行；提取失败、没有产出
        或结果为 TOUCH 时，原文仍保存在 Artifact 中，只是没有记忆引用它。
        """
        if interaction_input is None:
            return None
        if not interaction_input.blocks:
            return None
        try:
            return await self._artifact_engine.interaction.build_and_store(
                topic_id=interaction_input.topic_id,
                topic_title=interaction_input.topic_title,
                topic_summary=interaction_input.topic_summary,
                blocks=interaction_input.blocks,
                folds=interaction_input.folds,
                belong_to=belong_to,
            )
        except Exception:
            logger.warning("构建交互 artifact 失败", exc_info=True)
            return None

    async def promote_attachments(
        self,
        bindings: tuple[TopicAssetBinding, ...],
        *,
        belong_to: WorkspaceIdentity,
    ) -> None:
        """沿 binding.asset_ref 提升附件 DocumentArtifact（best-effort）。

        每个 binding 固定为：acquire READY representation → 构建
        DocumentArtifact → 释放 lease。ref 已 remove、Store 已关闭或写入
        失败时跳过该 binding 并记录结构化 warning；已提交的 binding 保持
        不变，不回滚 Interaction/Memory 结果。promotion
        retry 复用现有 generation operation identity 与同一 binding payload。
        """
        for binding in bindings:
            try:
                await self._promote_attachment(binding, belong_to=belong_to)
            except WorkspaceDomainError as exc:
                logger.warning(
                    "附件 promotion 跳过: asset_id=%s, code=%s",
                    binding.asset_id,
                    exc.code,
                )
            except Exception as exc:
                logger.warning(
                    "附件 promotion 写入失败: asset_id=%s, error=%s",
                    binding.asset_id,
                    type(exc).__name__,
                    exc_info=True,
                )

    async def _promote_attachment(
        self,
        binding: TopicAssetBinding,
        *,
        belong_to: WorkspaceIdentity,
    ) -> None:
        if self._asset_reader is None:
            return
        lease = self._asset_reader.acquire_ready_representation(
            belong_to,
            binding.asset_ref,
        )
        try:
            representation = lease.representation
            content_format = ""
            if isinstance(representation.content_object, Mapping):
                content_format = str(representation.content_object.get("format") or "")
            source_type: DocumentSourceType = "markdown" if content_format == "markdown" else "file"
            mime_type = "text/markdown" if content_format == "markdown" else "text/plain"
            # 冻结映射：source asset/representation 标识 + revision +
            # producer/version 全部钉进 source_uri，content_hash 单列，
            # 使提升产物自身锁定来源版本：消费方读到的始终是被冻结的那个版本。
            source_uri = (
                f"attachment://{binding.asset_id}"
                f"#{representation.representation_id}"
                f"?revision={representation.revision}"
                f"&producer={representation.producer}"
                f"&producer_version={representation.producer_version}"
            )
            await self._artifact_engine.document.build_and_store(
                source_type=source_type,
                source_uri=source_uri,
                content_hash=representation.content_hash,
                retrieved_at=datetime.now(UTC),
                workspace_identity=belong_to,
                mime_type=mime_type,
                title=f"attachment:{binding.asset_id}",
            )
        finally:
            self._asset_reader.release_representation_lease(lease.lease_id)


__all__ = ["SourceArtifactRecorder"]
