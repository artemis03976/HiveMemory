"""
帕秋莉·检索使魔 (The Retrieval Familiar of Patchouli)

定位：服务员与执行者
职责：
    - 混合检索 (Dense + Sparse + RRF)
    - 重排序 (Reranking)
    - 访问统计更新

版本: 3.0 (Phase C — 编译解耦)
"""

import logging
from typing import Any
from uuid import UUID

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    AgentProfile,
    MemoryAtom,
    MemoryType,
    ResolvedAgentProfile,
    TopicData,
    TopicSnapshot,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.models.query import QueryFilters
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
    MemoryTypeMismatchError,
)
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.models import RetrievalQuery
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.memory_library.library import MemoryLibrary
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
        local_bus: Any | None = None,
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

        ``enforce_actor_visibility=False`` 供 owner-management 管理入口使用
        （D4）：ownership hard boundary 仍生效，跳过 Workspace 内 actor
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
        limit: int = 20,
        enforce_actor_visibility: bool = True,
    ) -> list[MemoryAtom]:
        """
        根据查询和过滤条件列出中期记忆原子。

        ``enforce_actor_visibility=False`` 供 owner-management 管理入口使用
        （D4）：ownership hard boundary 仍生效，跳过 Workspace 内 actor
        可见性过滤；Agent retrieval 不得使用该开关。
        """
        query_filters = self._build_business_filters(filters)
        if query:
            results = await self._memory_library.mid_term.search(
                belong_to,
                query=query,
                top_k=limit,
                filters=query_filters,
                from_actor=from_actor,
                enforce_actor_visibility=enforce_actor_visibility,
            )
            return [result["memory"] for result in results if "memory" in result]
        return await self._memory_library.mid_term.scroll(
            belong_to,
            filters=query_filters,
            from_actor=from_actor,
            limit=limit,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
    ) -> ResolvedAgentProfile:
        """
        Profile 解析的唯一实现：builtin/alias 查找/可见性校验/类型校验/解析
        只维护在本方法，返回 ``ResolvedAgentProfile``（A2 §8 D-3）。

        只有未指定 alias 或明确选择内置 ``default`` / ``omni_doll`` 时才返回
        Omni-Doll（无源原子）。任何自定义 alias 的缺失、越权、类型错误或配置
        损坏都会显式失败，不降级为默认配置。源原子的读取策略与 UUID/版本随
        结果返回，供 workspace Profile 解析缓存做命中授权与失效对账；可见性
        校验在此处独立成立（纵深防御）。
        """
        normalized_alias = agent_alias.strip() if agent_alias else ""
        if not normalized_alias or normalized_alias in ("default", "omni_doll"):
            return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE.model_copy(deep=True))

        atom = await self._memory_library.mid_term.get_by_alias(
            belong_to,
            normalized_alias,
            from_actor=from_actor,
        )
        if atom is None:
            raise AliasNotFoundError(
                message_key="mtp.call.profile_not_found",
                params={"agent_alias": normalized_alias},
            )
        if atom.index.memory_type != MemoryType.AGENT_PROFILE:
            raise MemoryTypeMismatchError(
                message_key="mtp.call.profile_type_mismatch",
                params={"agent_alias": normalized_alias},
            )

        profile = AgentProfile.from_atom(atom)
        if profile is None:
            raise InvalidArgumentError(
                message_key="mtp.call.profile_invalid",
                params={"agent_alias": normalized_alias},
            )
        return ResolvedAgentProfile(
            profile=profile.model_copy(deep=True),
            access_policy=atom.meta.access_policy.model_copy(deep=True),
            source_memory_id=atom.id,
            source_version=atom.meta.version,
        )

    async def retrieve(self, query: RetrievalQuery, top_k: int = 10) -> list[MemoryAtom]:
        """
        语义检索相关记忆，按领域排序返回完整原子列表（A2 §2.1）。

        检索失败（存储不可用、引擎异常）按原错误传播，不伪装为空列表；
        耗时等诊断信息只进入日志，由调用侧 adapter 自行测量。
        """
        engine_result = await self.engine.retrieve(
            query=query,
            top_k=top_k,
        )

        logger.info(
            f"检索完成: query='{query.semantic_query[:20]}...', "
            f"filters={query.filters}, "
            f"使魔取回了 {engine_result.memories_count} 条记忆, "
            f"latency={engine_result.latency_ms:.1f}ms"
        )
        return list(engine_result.memories)

    async def retrieve_async(self, query: RetrievalQuery, top_k: int = 10) -> list[MemoryAtom]:
        """
        异步总线入口：只执行检索与活跃度刷新。
        """
        memories = await self.retrieve(query, top_k=top_k)
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
        精确按 alias 取回实际可读的完整原子（A2 §2.1）。

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
                logger.debug(f"Alias not found during alias retrieval: {normalized}")
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
                # 受限局部更新：只推进访问计数与最近访问时间（A2-P §4.1）。
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
                logger.warning(f"更新访问统计失败: {memory.id} - {e}")

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
            logger.warning(f"Failed to refresh retrieval vitality scores: {e}")

    @staticmethod
    def _build_business_filters(filters: dict[str, Any] | None) -> QueryFilters:
        """只接纳业务维度，拒绝调用方用裸字典覆盖 Workspace hard boundary。"""
        if not filters:
            return QueryFilters()
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
            min_confidence=float(confidence),
        )


__all__ = [
    "RetrievalFamiliar",
]
