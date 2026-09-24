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
import time
from typing import Any
from uuid import UUID

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    IdentityScope,
    MemoryAtom,
    MemoryType,
    ProfileSnapshot,
    TopicData,
    TopicSnapshot,
    WorkspaceMemoryKey,
    require_identity_scope,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
    MemoryTypeMismatchError,
    StorageOfflineError,
    StorageReadError,
)
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.models import QueryFilters, RetrievalQuery
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
        1. 接收业务请求 (RetrievalRequest)
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
        identity_scope: IdentityScope,
    ) -> TopicData | None:
        """
        读取短期话题上下文（纯读，无访问追踪副作用）。
        """
        require_identity_scope(identity_scope)
        return self._memory_library.short_term.get(
            identity_scope,
            topic_id,
        )

    def list_active_topics(
        self,
        *,
        identity_scope: IdentityScope,
        include_empty: bool = False,
        sort_by_recency: bool = True,
    ) -> tuple[TopicSnapshot, ...]:
        """
        列出指定用户的话题快照（短期检索入口）。

        默认排除空话题，供 Gateway 路由决策使用；include_empty=True
        时可承接前端话题池展示。按 ``last_update``（最近写入）倒序排列。
        """
        identity_scope = require_identity_scope(identity_scope)
        topics = self._memory_library.short_term.list_by_workspace(
            identity_scope,
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
        identity_scope: IdentityScope,
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
            require_identity_scope(identity_scope),
            normalized_id,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def list_memories(
        self,
        *,
        identity_scope: IdentityScope,
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
        identity_scope = require_identity_scope(identity_scope)
        query_filters = self._build_business_filters(filters)
        if query:
            results = await self._memory_library.mid_term.search(
                identity_scope,
                query=query,
                top_k=limit,
                filters=query_filters,
                enforce_actor_visibility=enforce_actor_visibility,
            )
            return [result["memory"] for result in results if "memory" in result]
        return await self._memory_library.mid_term.scroll(
            identity_scope,
            filters=query_filters,
            limit=limit,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        identity_scope: IdentityScope,
    ) -> AgentProfile:
        """
        根据 Agent 别名读取配置，并由 Profile 所有者 Patchouli 执行可见性校验。

        只有未指定 alias 或明确选择内置 ``default`` / ``omni_doll`` 时才返回
        Omni-Doll。任何自定义 alias 的缺失、越权、类型错误或配置损坏都会显式失败。
        """
        snapshot = await self.get_agent_profile_snapshot(
            agent_alias,
            identity_scope=identity_scope,
        )
        return snapshot.profile

    async def get_agent_profile_snapshot(
        self,
        agent_alias: str | None,
        *,
        identity_scope: IdentityScope,
    ) -> ProfileSnapshot:
        """
        Profile 解析的唯一实现（父计划 5.2 节）：builtin/alias 查找/类型校验/
        profile 解析只维护在本方法，返回携带 source atom UUID/revision 的
        不可变快照；``get_agent_profile`` 是其裸 Profile 兼容投影。
        """
        identity_scope = require_identity_scope(identity_scope)
        normalized_alias = agent_alias.strip() if agent_alias else ""
        if not normalized_alias or normalized_alias in ("default", "omni_doll"):
            return ProfileSnapshot(
                agent_alias=None,
                profile=OMNI_DOLL_PROFILE.model_copy(deep=True),
                source_kind="builtin",
            )

        atom = await self._memory_library.mid_term.get_by_alias(
            identity_scope,
            normalized_alias,
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
        return ProfileSnapshot(
            agent_alias=normalized_alias,
            profile=profile.model_copy(deep=True),
            source_kind="atom",
            source_atom_uuid=str(atom.id),
            source_revision=atom.meta.version,
        )

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        检索相关记忆，返回原子与元信息
        """
        start_time = time.time()

        response = RetrievalResponse()

        try:
            query_filters = QueryFilters()

            # Step 2: 合并 MTP filter (如果有)
            if request.filters is not None:
                if request.filters.memory_type is not None:
                    query_filters.memory_type = request.filters.memory_type
                if request.filters.tags:
                    query_filters.tags = request.filters.tags
                if request.filters.min_confidence > 0:
                    query_filters.min_confidence = request.filters.min_confidence

            # Step 3: 构建 RetrievalQuery
            query = RetrievalQuery(
                semantic_query=request.semantic_query,
                keywords=request.keywords or [],
                filters=query_filters,
                identity_scope=request.identity_scope,
            )

            engine_result = await self.engine.retrieve(
                query=query,
                top_k=request.top_k,
            )

            response.memories = engine_result.memories
            response.memories_count = engine_result.memories_count
            response.latency_ms = engine_result.latency_ms

            logger.info(
                f"检索完成: query='{request.semantic_query[:20]}...', "
                f"filters={query_filters}, "
                f"使魔取回了 {response.memories_count} 条记忆, "
                f"latency={response.latency_ms:.1f}ms"
            )

        except (StorageOfflineError, StorageReadError):
            raise
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            response.latency_ms = (time.time() - start_time) * 1000

        return response

    async def retrieve_async(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        异步总线入口：只执行检索与活跃度刷新。
        """
        response = await self.retrieve(request)
        await self._refresh_vitality_for_memories(response.memories)
        return response

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        identity_scope: IdentityScope,
    ) -> RetrievalResponse:
        """
        精确按 alias 取回记忆。
        """
        start_time = time.time()
        response = RetrievalResponse()
        identity_scope = require_identity_scope(identity_scope)

        try:
            memories: list[MemoryAtom] = []
            seen_aliases: set[str] = set()
            for alias in aliases:
                normalized = alias.strip() if alias else ""
                if not normalized or normalized in seen_aliases:
                    continue
                seen_aliases.add(normalized)

                atom = await self._memory_library.mid_term.get_by_alias(
                    identity_scope,
                    normalized,
                )
                if atom is None:
                    logger.warning(f"Alias not found during alias retrieval: {normalized}")
                    continue
                memories.append(atom)

            response.memories = memories
            response.memories_count = len(memories)
            response.latency_ms = (time.time() - start_time) * 1000

        except (StorageOfflineError, StorageReadError):
            raise
        except Exception as e:
            logger.error(f"Alias retrieval failed: {e}", exc_info=True)
            response.latency_ms = (time.time() - start_time) * 1000

        return response

    async def retrieve_by_aliases_async(
        self,
        aliases: list[str],
        identity_scope: IdentityScope,
    ) -> RetrievalResponse:
        """
        精确别名检索的异步总线入口。
        """
        response = await self.retrieve_by_aliases(aliases, identity_scope)
        await self._refresh_vitality_for_memories(response.memories)
        return response

    async def update_access_stats(
        self,
        identity_scope: IdentityScope,
        memories: list[MemoryAtom],
    ) -> None:
        """
        更新被引用记忆的访问统计

        当记忆被成功使用时调用，增加访问计数
        """
        identity_scope = require_identity_scope(identity_scope)
        now = utc_now()
        for memory in memories:
            try:
                # 受限局部更新：只推进访问计数与最近访问时间（A2-P §4.1）。
                await self._memory_library.mid_term.patch_payload(
                    WorkspaceMemoryKey(
                        workspace_identity=identity_scope.workspace_identity,
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
