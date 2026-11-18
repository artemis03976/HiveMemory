"""
帕秋莉·检索使魔 (The Retrieval Familiar of Patchouli)

定位：服务员与执行者
职责：
    - 混合检索 (Dense + Sparse + RRF)
    - 重排序 (Reranking)
    - 访问统计更新
"""

import logging
from typing import Any
from uuid import UUID

from hivememory.core.errors import InvalidAgentProfileError, ResourceNotFoundError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AgentProfile,
    MemoryAtom,
    MemoryType,
    TopicData,
    TopicSnapshot,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.models.query import QueryFilters
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.models import RetrievalQuery
from hivememory.memory_library.library import MemoryLibrary
from hivememory.patchouli.contracts.errors import ProfileResolutionError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_management import MemoryListPage
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.utils.time import utc_now

logger = logging.getLogger(__name__)


class RetrievalFamiliar:
    """
    帕秋莉·检索使魔 (The Retrieval Familiar of Patchouli)

    当"真理之眼"确认需要查书时，帕秋莉会召唤使魔去书架取书。

    特性：
        - 原生异步 I/O
        - 高并发
        - 本地计算密集

    职责：
        1. 接收公共边界转换后的内部检索查询 (RetrievalQuery)
        2. 根据 user_id 创建过滤条件 (乐观检索策略)
        3. 调用 RetrievalEngine 进行数据检索
        4. 处理副作用 (如统计更新)

    检索结果只包含记忆原子与元信息，Agent 可读文本由调用方通过 MemoryCompiler 编译。
    """

    def __init__(
        self,
        engine: RetrievalEngine,
        memory_library: MemoryLibrary,
        local_bus: PatchouliBus | None = None,
    ):
        """
        初始化检索使魔

        Args:
            engine: 检索引擎实例
            memory_library: 三级记忆书库，用于短/中/长期读入口
            local_bus: 本地总线，用于与其他服务通信
        """
        self.engine = engine
        self._memory_library = memory_library
        self._local_bus = local_bus

        logger.info("RetrievalFamiliar (检索使魔) 初始化完成")

    # ========== 短期记忆查询 ==========

    def get_topic(
        self,
        topic_id: str,
        *,
        belong_to: WorkspaceIdentity,
    ) -> TopicData | None:
        """
        读取短期话题上下文（纯读，无访问追踪副作用）。
        """
        return self._memory_library.short_term.get(
            belong_to,
            topic_id,
        )

    def list_active_topics(
        self,
        *,
        belong_to: WorkspaceIdentity,
        include_empty: bool = False,
        sort_by_recency: bool = True,
    ) -> tuple[TopicSnapshot, ...]:
        """
        列出指定用户的话题快照（短期检索入口）。

        默认排除空话题，供 Gateway 路由决策使用；include_empty=True
        时可承接前端话题池展示。按 ``last_update``（最近写入）倒序排列。
        """
        topics = self._memory_library.short_term.list_by_workspace(
            belong_to,
            include_empty=include_empty,
        )
        if not include_empty:
            topics = [topic for topic in topics if not topic.is_empty]
        if sort_by_recency:
            topics = sorted(topics, key=lambda t: t.last_update, reverse=True)
        return tuple(topic.to_topic_snapshot() for topic in topics)

    # ========== 中期记忆查询 ==========

    async def get_memory(
        self,
        memory_id: UUID | str,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        """
        根据记忆 ID 读取中期记忆原子。

        ``enforce_actor_visibility=False`` 供 owner 管理入口使用：管理读取要看到
        Workspace 内的全部记忆，ownership hard boundary 仍生效，只跳过 actor
        可见性过滤；Agent retrieval 不得使用该开关。
        """
        normalized_id = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        return await self._memory_library.mid_term.get(
            belong_to,
            normalized_id,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def list_memories(
        self,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        exclude_types: list[str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
        enforce_actor_visibility: bool = True,
    ) -> MemoryListPage:
        """
        根据查询和过滤条件列出一页中期记忆原子。

        没有 ``query`` 时按存储顺序翻页，``cursor`` 取上一页的 ``next_cursor``，
        ``total`` 按同一过滤条件计数；有 ``query`` 时按相关度返回至多 ``limit``
        条，不翻页。``exclude_types`` 在存储侧排除，每页条数与计数都不受影响。

        ``enforce_actor_visibility=False`` 供 owner 管理入口使用：管理读取要看到
        Workspace 内的全部记忆，ownership hard boundary 仍生效，只跳过 actor
        可见性过滤；Agent retrieval 不得使用该开关。
        """
        query_filters = self._build_business_filters(filters, exclude_types)
        mid_term = self._memory_library.mid_term
        if query:
            results = await mid_term.search(
                belong_to,
                query=query,
                top_k=limit,
                filters=query_filters,
                from_actor=from_actor,
                enforce_actor_visibility=enforce_actor_visibility,
            )
            found = tuple(result["memory"] for result in results if "memory" in result)
            return MemoryListPage(memories=found, next_cursor=None, total=len(found))
        memories, next_cursor = await mid_term.scroll(
            belong_to,
            filters=query_filters,
            limit=limit,
            cursor=cursor,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )
        total = await mid_term.count(
            belong_to,
            filters=query_filters,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )
        return MemoryListPage(memories=tuple(memories), next_cursor=next_cursor, total=total)

    async def get_agent_profile(self, agent_alias: str | None) -> AgentProfile:
        """Profile 解析的唯一实现：内置 Profile、按 agent_id 读取文件与三种失败原因。

        只有未指定 agent_id 或明确选择内置 ``default`` / ``omni_doll`` 时才返回
        Omni-Doll。其余 agent_id 读取 AgentProfileStore（全局，不按 Workspace 区分）：没有文件时抛出
        ``ResourceNotFoundError``，文件不合格（``invalid_config``）或已停用
        （``profile_disabled``）时抛出 ``ProfileResolutionError``，都不降级为内置
        Profile。每次都直接读文件，文件的改动在下一次解析时生效。
        """
        normalized_alias = agent_alias.strip() if agent_alias else ""
        if not normalized_alias or normalized_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE.model_copy(deep=True)

        try:
            stored = self._memory_library.agent_profiles.get(normalized_alias)
        except InvalidAgentProfileError as exc:
            raise ProfileResolutionError(
                agent_alias=normalized_alias, reason="invalid_config"
            ) from exc
        if stored is None:
            raise ResourceNotFoundError(
                details={"resource": "agent_profile", "agent_alias": normalized_alias}
            )
        if not stored.profile.enabled:
            raise ProfileResolutionError(agent_alias=normalized_alias, reason="profile_disabled")
        return stored.profile

    async def retrieve(self, query: RetrievalQuery) -> list[MemoryAtom]:
        """
        语义检索相关记忆，按领域排序返回完整原子列表；条数由检索器配置决定。

        检索失败（存储不可用、引擎异常）按原错误传播，不伪装为空列表；
        耗时等诊断信息只进入日志，由调用侧 adapter 自行测量。
        """
        engine_result = await self.engine.retrieve(query=query)

        logger.debug(
            "检索完成: query_length=%d, filters=%s, 使魔取回了 %s 条记忆, latency=%.1fms",
            len(query.semantic_query),
            query.filters,
            engine_result.memories_count,
            engine_result.latency_ms,
        )
        return list(engine_result.memories)

    async def retrieve_async(self, query: RetrievalQuery) -> list[MemoryAtom]:
        """
        异步总线入口：只执行检索与活跃度刷新。
        """
        memories = await self.retrieve(query)
        await self._refresh_vitality_for_memories(memories)
        return memories

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        belong_to: WorkspaceIdentity,
        *,
        from_actor: ActorIdentity,
    ) -> list[MemoryAtom]:
        """
        精确按 alias 取回实际可读的完整原子。

        alias 先去首尾空白并按首次出现去重，结果保持请求顺序；缺失或对当前
        Actor 不可见的 alias 不出现在结果中（不以下标表达逐项状态）。存储
        不可用与 alias 多义（``MemoryAliasConflictError``）按原错误传播，
        不伪装为空列表。
        """
        memories: list[MemoryAtom] = []
        seen_aliases: set[str] = set()
        for alias in aliases:
            normalized = alias.strip() if alias else ""
            if not normalized or normalized in seen_aliases:
                continue
            seen_aliases.add(normalized)

            atom = await self._memory_library.mid_term.get_by_alias(
                belong_to,
                normalized,
                from_actor=from_actor,
            )
            if atom is None:
                logger.debug("按 alias 检索未命中: %s", normalized)
                continue
            memories.append(atom)
        return memories

    async def retrieve_by_aliases_async(
        self,
        aliases: list[str],
        belong_to: WorkspaceIdentity,
        *,
        from_actor: ActorIdentity,
    ) -> list[MemoryAtom]:
        """
        精确别名检索的异步总线入口。
        """
        memories = await self.retrieve_by_aliases(aliases, belong_to, from_actor=from_actor)
        await self._refresh_vitality_for_memories(memories)
        return memories

    async def update_access_stats(
        self,
        belong_to: WorkspaceIdentity,
        memories: list[MemoryAtom],
    ) -> None:
        """
        更新被引用记忆的访问统计

        当记忆被成功使用时调用，增加访问计数
        """
        now = utc_now()
        for memory in memories:
            try:
                # 受限局部更新：只推进访问计数与最近访问时间。
                await self._memory_library.mid_term.patch_payload(
                    WorkspaceMemoryKey(
                        workspace_identity=belong_to,
                        memory_id=memory.id,
                    ),
                    {
                        "meta.lifecycle.access_count": memory.meta.lifecycle.access_count + 1,
                        "meta.lifecycle.last_accessed_at": now,
                    },
                )
            except Exception as e:
                logger.warning("更新访问统计失败: memory_id=%s: %s", memory.id, e)

    # ========== 长期记忆查询 ==========

    async def is_archived(self, memory_id) -> bool:
        """
        检查记忆是否已进入长期冷存储。
        """
        return await self._memory_library.long_term.is_archived(memory_id)

    # ========== 内部辅助方法 ==========

    async def _refresh_vitality_for_memories(self, memories: list[MemoryAtom]) -> None:
        if not memories or self._local_bus is None:
            return
        try:
            await self._local_bus.request(
                PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY,
                memories,
                persist=False,
            )
        except Exception as e:
            logger.warning("刷新检索结果的生命力分数失败: %s", e)

    @staticmethod
    def _build_business_filters(
        filters: dict[str, Any] | None, exclude_types: list[str] | None = None
    ) -> QueryFilters:
        """只接纳业务维度，拒绝调用方用裸字典覆盖 Workspace hard boundary。"""
        excluded = [MemoryType(t) for t in exclude_types or []]
        if not filters:
            return QueryFilters(exclude_memory_types=excluded)
        allowed = {"index.memory_type", "meta.lifecycle.confidence_score"}
        unsupported = set(filters) - allowed
        if unsupported:
            raise ValueError(f"不支持的 Memory 过滤字段: {sorted(unsupported)}")
        memory_type = filters.get("index.memory_type")
        confidence = filters.get("meta.lifecycle.confidence_score", 0.0)
        if isinstance(confidence, dict):
            confidence = confidence.get("gte", 0.0)
        return QueryFilters(
            memory_type=MemoryType(memory_type) if memory_type else None,
            exclude_memory_types=excluded,
            min_confidence=float(confidence),
        )


__all__ = [
    "RetrievalFamiliar",
]
