"""Patchouli 记忆意图提交 application 用例（memory_intent.submit）。

承接中立的记忆意图参数并转换为内部生成任务：由 controller admission 与
generation engine 决定生成、更新、合并或丢弃；**不映射为管理 CRUD，也
不由调用方构造内部任务投影**。Alice 的 settlement coordinator（WRX-5）
与计划 B 的外部 adapter 都经本用例提交。

本用例不在 A1 第 6 节兼容清单内：缺少经统一认证网关签发的 access 一律
拒绝，不进入裸 scope 受信适配。

幂等语义：``MemoryIntent.intent_id`` 是幂等键——``pending_alias`` 由
``intent_id`` 确定性派生，同一 intent 携带相同载荷重试命中 controller
幂等复用并返回原任务；载荷变化按 intent 冲突确定性拒绝
（at-most-one-canonical，不产生第二个 canonical Memory）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from hivememory.core.models.pending import (
    PendingAtomMaterializeTask,
    UpdateFocus,
    WriteFocus,
)
from hivememory.patchouli.application.access_consumption import required_scope
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.workspace.access import WorkspaceOperation

if TYPE_CHECKING:
    from hivememory.core.models import IdentityScope
    from hivememory.patchouli.runtime.bus import PatchouliBus
    from hivememory.system.access import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard


def _slugify(text: str, max_len: int = 30) -> str:
    """将文本转为 alias 友好的 slug 片段（与 PendingAtom alias 风格一致）。"""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s_]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len].rstrip("_")


def _intent_token(intent_id: str) -> str:
    """从 intent_id 派生确定性短 token，保证重试生成同一 pending_alias。"""
    return hashlib.sha1(intent_id.encode("utf-8")).hexdigest()[:4]


@dataclass(frozen=True)
class MemoryIntent:
    """中立的记忆意图参数：由 Patchouli 决定生成、更新、合并或丢弃。

    ``kind="write"`` 需要 ``content``；``kind="update"`` 需要 ``instruction``
    以及 ``base_alias`` + ``base_uuid``（UPDATE 的业务前提是调用方已持有
    当前 base 的坐标）。``intent_id`` 是幂等键，缺省生成。
    """

    kind: Literal["write", "update"]
    topic_id: str
    content: str | None = None
    title: str | None = None
    reason: str | None = None
    instruction: str | None = None
    base_alias: str | None = None
    base_uuid: str | None = None
    intent_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "write":
            if not self.content:
                raise ValueError("write 意图需要 content")
            if self.instruction or self.base_alias or self.base_uuid:
                raise ValueError("write 意图不得携带 update 字段")
        elif self.kind == "update":
            if not self.instruction:
                raise ValueError("update 意图需要 instruction")
            if not self.base_alias or not self.base_uuid:
                raise ValueError("update 意图需要 base_alias 与 base_uuid")
        else:
            raise ValueError(f"不支持的意图类型: {self.kind!r}")
        if not isinstance(self.topic_id, str) or not self.topic_id.strip():
            raise ValueError("topic_id 不能为空")


@dataclass(frozen=True)
class MemoryIntentSubmissionResult:
    """意图提交回执，说明"是否被接纳"；接纳不等于已应用。

    ``accepted=False`` 表示生成 admission 未接纳（确定性拒绝或批量响应
    缺失）；应用进度经任务观察用例以同一 task_id 查询。
    """

    accepted: bool
    task_id: str | None = None
    task_status: str | None = None
    detail: str | None = None


class MemoryIntentSubmissionService:
    """经 Patchouli 生成提交链的公开意图提交用例（``memory_intent.submit``）。"""

    def __init__(self, *, bus: PatchouliBus, access_guard: WorkspaceAccessGuard) -> None:
        self._bus = bus
        self._access_guard = access_guard

    async def submit_memory_intent(
        self,
        *,
        access: WorkspaceAccessContext,
        intent: MemoryIntent,
        identity_scope: IdentityScope | None = None,
    ) -> MemoryIntentSubmissionResult:
        """提交记忆意图；结果由 Patchouli 生成链决定并经任务观察用例查询。"""
        scope = required_scope(
            access, WorkspaceOperation.MEMORY_INTENT_SUBMIT, identity_scope,
            access_guard=self._access_guard,
        )

        task = self._build_materialize_task(intent, scope)
        accepted = await self._bus.request(
            PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
            tasks=[task],
            topic_id=intent.topic_id,
            identity_scope=scope,
            submitted_by=access.principal.principal_id,
        )
        tasks = list(accepted or [])
        if not tasks:
            # 空响应表示确定性拒绝（intent 冲突 / spec 构建失败，settler 已
            # 发布失败投影）或批量结果缺失；两种情况都不能声称"已接纳"。
            return MemoryIntentSubmissionResult(
                accepted=False,
                detail="generation admission did not accept the intent",
            )
        admitted = tasks[0]
        return MemoryIntentSubmissionResult(
            accepted=True,
            task_id=admitted.task_id,
            task_status=admitted.status.value,
        )

    # ---- 内部辅助 ----

    @staticmethod
    def _build_materialize_task(
        intent: MemoryIntent,
        scope: IdentityScope,
    ) -> PendingAtomMaterializeTask:
        """把中立意图转换为内部生成任务投影（本用例的内部实现细节）。"""
        resolved_intent_id = intent.intent_id or f"intent_{uuid4().hex[:12]}"
        token = _intent_token(resolved_intent_id)
        if intent.kind == "write":
            slug = _slugify(intent.title or intent.content or "") or "untitled"
            pending_alias = f"draft_{slug}_{token}"
            focus = WriteFocus(
                content=intent.content or "",
                reason=intent.reason,
                title=intent.title,
            )
            source_verb = "WRITE"
        else:
            slug = _slugify(intent.base_alias or "") or "base"
            pending_alias = f"rev_{slug}_{token}"
            focus = UpdateFocus(
                instruction=intent.instruction or "",
                content=intent.content,
                base_uuid=intent.base_uuid or "",
                base_alias=intent.base_alias or "",
            )
            source_verb = "UPDATE"
        return PendingAtomMaterializeTask(
            pending_alias=pending_alias,
            intent_id=resolved_intent_id,
            source_verb=source_verb,
            identity_scope=scope,
            focus=focus,
        )


__all__ = [
    "MemoryIntent",
    "MemoryIntentSubmissionResult",
    "MemoryIntentSubmissionService",
]
