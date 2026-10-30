"""Patchouli 记忆生成任务的规范化与提交协调。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from uuid import UUID

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    LogicalBlock,
    WorkspaceIdentity,
    system_actor_for_workspace,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, UpdateFocus, WriteFocus
from hivememory.engines.generation.models import GenerationRequest
from hivememory.engines.perception.models import TopicMaterializeTask
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationAdmission
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
    """某个生效的生成规范无法构建时抛出。"""


class MemoryGenerationCoordinator:
    """将原始生成请求归一化为 MemoryGenerationTaskSpec。"""

    def __init__(
        self,
        *,
        bus: PatchouliBus,
        recent_block_limit: int,
        pending_atom_settler: PendingAtomSettler | None = None,
        transcript_config: GenerationTranscriptConfig | None = None,
    ) -> None:
        if recent_block_limit < 0:
            raise ValueError("recent_block_limit 不能为负数")
        self._bus = bus
        # 主动生成只取话题最近的若干个 block 作为材料（含 state_summary）。
        self._recent_block_limit = recent_block_limit
        self._pending_atom_settler = pending_atom_settler or PendingAtomSettler(bus)
        # 生成材料的过程条目按配置的上限截断，感知的 token 统计使用同一组上限。
        self._transcript_builder = GenerationTranscriptBuilder(transcript_config)

    async def submit_settlement(self, payload: TopicMaterializeTask) -> MemoryGenerationTask | None:
        """将话题结算转为 SETTLE 任务，统一以归属对应的 system 为发起者。"""
        gen_context = self._transcript_builder.build_context(
            payload.blocks,
            state_summary=payload.state_summary,
        )
        if not gen_context.turns:
            logger.warning("空对话轮次，跳过被动生成")
            return None

        spec = MemoryGenerationTaskSpec(
            belong_to=payload.belong_to,
            from_actor=system_actor_for_workspace(payload.belong_to),
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
        task: PendingAtomMaterializeTask,
        topic_id: str,
        *,
        belong_to: WorkspaceIdentity,
    ) -> MemoryGenerationAdmission:
        """将一个 WRITE/UPDATE 意图转为主动生成任务，返回它的接纳结果。

        材料是话题的 ``state_summary`` 与最近 ``recent_block_limit`` 个 block；
        话题不存在时按空材料生成。来源记录由 Patchouli 生成链内部维护；任务
        独立携带归属与发起者。
        """
        try:
            topic_data = await self._bus.request(
                PatchouliLocalRoutes.TOPIC_GET,
                topic_id,
                belong_to=belong_to,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # 读取材料失败时意图没有进入生成队列：确定性地结算为失败，
            # 不留下永远等不到结算的句柄。
            logger.warning(
                "Active generation topic read failed: topic_id=%s, intent_id=%s",
                topic_id,
                task.intent_id,
                exc_info=True,
            )
            await self._pending_atom_settler.failed(task.pending_alias, task.intent_id)
            return MemoryGenerationAdmission.rejected()

        blocks = (
            topic_data.recent_blocks(self._recent_block_limit) if topic_data is not None else ()
        )
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
        built = await self._try_build_active_spec(
            task,
            topic_id=topic_id,
            gen_context=gen_context,
            interaction_input=interaction_input,
            belong_to=belong_to,
        )
        if isinstance(built, MemoryGenerationAdmission):
            return built
        return await self._admit_spec(built)

    async def _admit_spec(self, spec: MemoryGenerationTaskSpec) -> MemoryGenerationAdmission:
        """提交规范；接纳请求本身失败时无法判断是否已接纳，按结果未知处理。"""
        try:
            return await self._bus.request(
                PatchouliLocalRoutes.MEMORY_TASK_ADMIT_GENERATION,
                spec,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Controller 已在接纳入口内分类拒绝与未知；连接纳结果都没有返回时，
            # 无法判断 intent 是否已被接纳，不能发布结算事件。
            logger.warning(
                "Active generation admission outcome unknown: intent_id=%s",
                spec.intent_id,
                exc_info=True,
            )
            return MemoryGenerationAdmission.unknown()

    async def _try_build_active_spec(
        self,
        task: PendingAtomMaterializeTask,
        *,
        topic_id: str,
        gen_context,
        interaction_input: InteractionArtifactInput | None,
        belong_to: WorkspaceIdentity,
    ) -> MemoryGenerationTaskSpec | MemoryGenerationAdmission:
        """构建规范；确定性失败结算为 FAILED 并返回拒绝，不确定的失败返回结果未知。"""
        try:
            return await self._build_active_spec(
                task,
                topic_id=topic_id,
                gen_context=gen_context,
                interaction_input=interaction_input,
                belong_to=belong_to,
            )
        except SpecBuildError as exc:
            logger.error(
                f"Active spec build failed, skipping task: pending_alias={task.pending_alias}, err={exc}",
            )
            await self._pending_atom_settler.failed(task.pending_alias, task.intent_id)
            return MemoryGenerationAdmission.rejected()
        except Exception:
            logger.exception(
                "Active spec build outcome unknown, keeping intent in flight: pending_alias=%s",
                task.pending_alias,
            )
            # 例如 MEMORY_GET 的存储异常无法确定 UPDATE 的业务前提，不属于
            # 确定性输入拒绝；不发布结算事件，由提交方按结果未知处理。
            return MemoryGenerationAdmission.unknown()

    async def _build_active_spec(
        self,
        task: PendingAtomMaterializeTask,
        *,
        topic_id: str,
        gen_context,
        interaction_input: InteractionArtifactInput | None,
        belong_to: WorkspaceIdentity,
    ) -> MemoryGenerationTaskSpec:
        if task.belong_to != belong_to:
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
                belong_to=belong_to,
                from_actor=task.from_actor,
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
            belong_to=task.belong_to,
            from_actor=task.from_actor,
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
        blocks: Sequence[LogicalBlock],
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
