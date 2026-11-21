"""非 Workspace 专项测试使用的 schema 2.1 Memory 构造辅助，以及修订用的漂移拦截替身。"""

import math
from typing import Any

from hivememory.core.models import (
    ActorIdentity,
    IdentityScope,
    MemoryAccessPolicy,
    MemoryLifecycleState,
    MemoryProvenance,
    MemoryVisibility,
    MetaData,
    WorkspaceIdentity,
)
from hivememory.engines.generation.edits import IndexDriftGuard
from hivememory.infrastructure.embedding.base import BaseEmbeddingService
from hivememory.utils.time import utc_now

# 旧平铺测试参数 → meta.lifecycle 聚合字段的透传集合。
_LIFECYCLE_KEYS = frozenset(
    {
        "access_count",
        "last_accessed_at",
        "event_vitality_boost",
        "vitality_score",
        "confidence_score",
        "verification_status",
        "decay_anchor_at",
    }
)

# schema 2.1 已删除、测试夹具不再接受的历史字段。
_REMOVED_KEYS = frozenset({"session_id", "history_summary"})


def make_memory_metadata(
    *,
    source_agent_id: str,
    user_id: str,
    team_id: str | None = None,
    visibility: MemoryVisibility | str = MemoryVisibility.PUBLIC,
    workspace_id: str = "main_workspace",
    access_policy: MemoryAccessPolicy | None = None,
    contributing_agent_ids: tuple[str, ...] | list[str] | None = None,
    lifecycle: MemoryLifecycleState | dict[str, Any] | None = None,
    **values: Any,
) -> MetaData:
    """把旧测试夹具的显式语义转换为 canonical 2.1 metadata。

    旧平铺动态字段（``access_count``/``vitality_score`` 等）透传进
    ``meta.lifecycle`` 聚合；``decay_anchor_at`` 未提供时取 ``created_at``（
    保持"创建时等于创建时间"）或当前 UTC 时间。
    """
    normalized = visibility.value if hasattr(visibility, "value") else str(visibility)
    if access_policy is None:
        if normalized == "PUBLIC":
            access_policy = MemoryAccessPolicy.public()
        elif normalized == "PRIVATE":
            access_policy = MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id=source_agent_id,
            )
        elif normalized in {"TEAM", "WORKSPACE"}:
            if not team_id:
                raise ValueError("TEAM 测试 metadata 必须提供 team_id")
            access_policy = MemoryAccessPolicy(
                visibility=MemoryVisibility.TEAM,
                target_team_id=team_id,
            )
        else:
            raise ValueError(f"不支持的测试 visibility: {normalized}")

    for removed in _REMOVED_KEYS:
        values.pop(removed, None)

    if lifecycle is None:
        lifecycle_kwargs = {k: values.pop(k) for k in list(values) if k in _LIFECYCLE_KEYS}
        if "decay_anchor_at" not in lifecycle_kwargs:
            lifecycle_kwargs["decay_anchor_at"] = values.get("created_at") or utc_now()
        lifecycle = MemoryLifecycleState(**lifecycle_kwargs)
    elif isinstance(lifecycle, dict):
        lifecycle = MemoryLifecycleState(**lifecycle)

    provenance = MemoryProvenance(
        source_agent_id=source_agent_id,
        source_team_id=team_id,
        contributing_agent_ids=tuple(contributing_agent_ids or ()),
    )

    return MetaData(
        workspace_identity=WorkspaceIdentity(
            owner_user_id=user_id,
            workspace_key=workspace_id,
            workspace_id=workspace_id,
        ),
        provenance=provenance,
        access_policy=access_policy,
        lifecycle=lifecycle,
        **values,
    )


def make_memory_identity_scope(
    *,
    user_id: str = "u1",
    agent_id: str = "a1",
    team_id: str | None = None,
    workspace_id: str = "main_workspace",
) -> IdentityScope:
    """构造不读取进程状态的显式生成 scope。"""
    return IdentityScope(
        actor_identity=ActorIdentity(
            user_id=user_id,
            agent_id=agent_id,
            team_id=team_id,
        ),
        workspace_identity=WorkspaceIdentity(
            owner_user_id=user_id,
            workspace_key=workspace_id,
            workspace_id=workspace_id,
        ),
    )


class FixedSimilarityEmbedding(BaseEmbeddingService):
    """嵌入服务的替身：批量编码修改前后两段索引文本时，返回余弦相似度为 ``similarity`` 的
    两条向量；``similarity`` 为 None 时任何编码请求都失败，用于断言不发生编码。"""

    def __init__(self, similarity: float | None = 1.0) -> None:
        self.similarity = similarity

    async def encode(self, dense_texts=None, sparse_texts=None, **kwargs: Any):
        if self.similarity is None:
            raise AssertionError("不应发生编码")
        assert isinstance(dense_texts, list) and len(dense_texts) == 2
        return [
            [1.0, 0.0],
            [self.similarity, math.sqrt(max(0.0, 1.0 - self.similarity**2))],
        ]


def make_drift_guard(
    *, similarity: float | None = 1.0, min_similarity: float = 0.85
) -> IndexDriftGuard:
    """以固定相似度的替身嵌入服务构造漂移拦截；缺省时一切修订都通过。"""
    return IndexDriftGuard(FixedSimilarityEmbedding(similarity), min_similarity=min_similarity)


__all__ = [
    "FixedSimilarityEmbedding",
    "make_drift_guard",
    "make_memory_identity_scope",
    "make_memory_metadata",
]
