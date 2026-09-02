"""Patchouli 记忆生成任务的规范化与提交协调。"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    LogicalBlock,
    IdentityScope,
    require_identity_scope,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, UpdateFocus, WriteFocus
from hivememory.engines.generation.models import GenerationRequest
from hivememory.engines.perception.models import TopicMaterializeTask
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.models import (
    InteractionArtifactInput,
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskSpec,
)
from hivememory.patchouli.control.pending_atom_settler import PendingAtomSettler
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.prompts.transcript import GenerationTranscriptBuilder

logger = logging.getLogger(__name__)


class SpecBuildError(RuntimeError):
    """Raised when one active generation spec cannot be built."""


class MemoryGenerationCoordinator:
    """将原始生成请求归一化为 MemoryGenerationTaskSpec。"""

    def __init__(
        self,
        *,
        bus: PatchouliBus,
        pending_atom_settler: PendingAtomSettler | None = None,
    ) -> None:
        self._bus = bus
        self._pending_atom_settler = pending_atom_settler or PendingAtomSettler(bus)
        self._transcript_builder = GenerationTranscriptBuilder()

    async def submit_settlement(self, payload: TopicMaterializeTask) -> MemoryGenerationTask | None:
        """将感知层 TopicMaterializeTask 转为 SETTLEMENT 任务规范。"""
        gen_context = self._transcript_builder.build_context(
            payload.blocks,
            state_summary=payload.state_summary,
        )
        if not gen_context.turns:
            logger.warning("空对话轮次，跳过被动生成")
            return None

        spec = MemoryGenerationTaskSpec(
            identity_scope=payload.identity_scope,
            topic_id=payload.topic_id,
            label=payload.topic_id,
            source=MemoryGenerationSource.SETTLE,
            request=GenerationRequest(
                context=gen_context,
            ),
            interaction_input=self._build_interaction_input(
                topic_id=payload.topic_id,
                topic_title=payload.topic_title,
                topic_summary=payload.topic_summary,
                blocks=payload.blocks,
                asset_bindings=payload.asset_bindings,
            ),
        )
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION,
            spec,
        )

    async def submit_active(
        self,
        tasks: list[PendingAtomMaterializeTask],
        topic_id: str,
        *,
        identity_scope: IdentityScope,
    ) -> list[MemoryGenerationTask]:
        """将 MTP WRITE/UPDATE 请求转为主动生成任务规范。"""
        if not tasks:
            return []
        identity_scope = require_identity_scope(identity_scope)

        topic_data = await self._bus.request(
            PatchouliLocalRoutes.TOPIC_GET,
            topic_id,
            identity_scope=identity_scope,
        )
        blocks = topic_data.recent_blocks(5) if topic_data is not None else []
        state_summary = topic_data.state_summary if topic_data is not None else ""
        gen_context = self._transcript_builder.build_context(
            blocks,
            state_summary=state_summary,
        )
        interaction_input = self._build_interaction_input(
            topic_id=topic_id,
            topic_title=topic_data.topic_title if topic_data is not None else "",
            topic_summary=topic_data.topic_summary if topic_data is not None else "",
            blocks=blocks,
        )

        # 并行构建任务规范，防止 UPDATE 任务的 IO 操作阻塞
        raw_specs = await asyncio.gather(
            *[
                self._try_build_active_spec(
                    task,
                    topic_id=topic_id,
                    gen_context=gen_context,
                    interaction_input=interaction_input,
                    identity_scope=identity_scope,
                )
                for task in tasks
            ]
        )
        # 过滤构建失败的任务
        specs = [spec for spec in raw_specs if spec is not None]
        if not specs:
            return []

        try:
            accepted = await self._bus.request(
                PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
                specs,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # 批量入口会在 Controller 内逐项隔离 admission；若连批量响应本身都
            # 未返回，则无法判断哪些 intent 已被接纳，必须保持 PendingAtom 非终态。
            logger.warning(
                "Active generation batch admission outcome unknown: intents=%s",
                [spec.intent_id for spec in specs],
                exc_info=True,
            )
            return []

        if not isinstance(accepted, list) or not all(
            isinstance(task, MemoryGenerationTask) for task in accepted
        ):
            logger.warning("Active generation batch admission returned invalid tasks")
            return []
        return accepted

    async def _try_build_active_spec(
        self,
        task: PendingAtomMaterializeTask,
        *,
        topic_id: str,
        gen_context,
        interaction_input: InteractionArtifactInput | None,
        identity_scope: IdentityScope,
    ) -> MemoryGenerationTaskSpec | None:
        try:
            return await self._build_active_spec(
                task,
                topic_id=topic_id,
                gen_context=gen_context,
                interaction_input=interaction_input,
                identity_scope=identity_scope,
            )
        except SpecBuildError as exc:
            logger.error(
                f"Active spec build failed, skipping task: pending_alias={task.pending_alias}, err={exc}",
            )
            await self._pending_atom_settler.failed(task.pending_alias)
            return None
        except Exception:
            logger.exception(
                "Active spec build outcome unknown, keeping intent pending: pending_alias=%s",
                task.pending_alias,
            )
            # 例如 MEMORY_GET 的存储异常无法确定 UPDATE 的业务前提，不属于
            # 确定性输入拒绝；保留 intent，下一次 dispatch 使用同一 intent_id 重试。
            return None

    async def _build_active_spec(
        self,
        task: PendingAtomMaterializeTask,
        *,
        topic_id: str,
        gen_context,
        interaction_input: InteractionArtifactInput | None,
        identity_scope: IdentityScope,
    ) -> MemoryGenerationTaskSpec:
        if task.identity_scope.workspace_identity != identity_scope.workspace_identity:
            raise WorkspaceMismatchError(details={"pending_alias": task.pending_alias})
        source = MemoryGenerationSource(task.source_verb)
        focus = task.focus
        if source == MemoryGenerationSource.WRITE:
            if not isinstance(focus, WriteFocus):
                raise SpecBuildError(f"WRITE focus must be WriteFocus, got {type(focus)}")
            request = GenerationRequest(
                context=gen_context,
                write_focus=focus,
            )
        elif source == MemoryGenerationSource.UPDATE:
            if not isinstance(focus, UpdateFocus):
                raise SpecBuildError(f"UPDATE focus must be UpdateFocus, got {type(focus)}")
            try:
                base_uuid = UUID(focus.base_uuid)
            except ValueError as exc:
                raise SpecBuildError(
                    f"UPDATE target memory UUID is invalid: {focus.base_uuid}"
                ) from exc

            existing = await self._bus.request(
                PatchouliLocalRoutes.MEMORY_GET,
                base_uuid,
                identity_scope=identity_scope,
            )
            if existing is None:
                logger.error(f"UPDATE target memory not found: {focus.base_uuid}")
                raise SpecBuildError(f"UPDATE target memory not found: {focus.base_uuid}")
            request = GenerationRequest(
                context=gen_context,
                update_focus=focus,
                existing_memory=existing,
            )
        else:
            raise ValueError(f"Unsupported active generation source: {source}")

        return MemoryGenerationTaskSpec(
            identity_scope=task.identity_scope,
            topic_id=topic_id,
            label=task.pending_alias,
            source=source,
            request=request,
            interaction_input=interaction_input,
            intent_id=task.intent_id,
            pending_alias=task.pending_alias,
        )

    def _build_interaction_input(
        self,
        *,
        topic_id: str,
        topic_title: str,
        topic_summary: str,
        blocks: list[LogicalBlock],
        asset_bindings: tuple = (),
    ) -> InteractionArtifactInput | None:
        """将原始交互数据冻结为生成数据平面的交互输入。"""
        if not blocks:
            return None
        return InteractionArtifactInput(
            topic_id=topic_id,
            topic_title=topic_title,
            topic_summary=topic_summary,
            blocks=tuple(blocks),
            asset_bindings=tuple(asset_bindings),
        )

__all__ = ["MemoryGenerationCoordinator"]
