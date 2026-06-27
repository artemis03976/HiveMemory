"""
帕秋莉·检索使魔 (The Retrieval Familiar of Patchouli)

定位：服务员与执行者
职责：
    - 混合检索 (Dense + Sparse + RRF)
    - 重排序 (Reranking)
    - 访问统计更新

版本: 3.0 (Phase C — renderer 解耦)
"""

from typing import Any, List, Optional
from uuid import UUID
import time
import logging

from hivememory.core.models import (
    AgentProfile,
    Identity,
    MemoryAtom,
    MemoryType,
    OMNI_DOLL_PROFILE,
    TopicSnapshot,
)
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.models import RetrievalQuery, QueryFilters
from hivememory.core.mtp.exceptions import StorageOfflineError, StorageReadError
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.models import TopicData

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

    rendered_context 由调用方填充 (Phase D)。
    """

    def __init__(
        self,
        engine: RetrievalEngine,
        memory_library: MemoryLibrary,
        local_bus: Optional[Any] = None,
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
        touch: bool = True,
        deep_copy: bool = True,
    ) -> Optional[TopicData]:
        """
        读取短期话题上下文。
        """
        return self._memory_library.short_term.get_topic_data(
            topic_id,
            touch=touch,
            deep_copy=deep_copy,
        )

    def list_active_topics(
        self,
        identity: Identity,
        *,
        include_empty: bool = False,
        sort_by_access: bool = True,
    ) -> List[TopicSnapshot]:
        """
        列出指定用户的话题快照（短期检索入口）。

        默认排除空话题，供 TheEye 路由决策使用；include_empty=True
        时可承接前端话题池展示。
        """
        topics = self._memory_library.short_term.list_topic_data(
            user_id=identity.user_id,
            include_empty=include_empty,
            deep_copy=False,
        )
        if not include_empty:
            topics = [topic for topic in topics if not topic.is_empty]
        if sort_by_access:
            topics = sorted(topics, key=lambda t: t.last_accessed_at, reverse=True)
        return [topic.to_topic_snapshot() for topic in topics]

    # ========== 中期记忆查询 ========== 

    async def get_memory(self, memory_id: UUID | str) -> Optional[MemoryAtom]:
        """
        根据记忆 ID 读取中期记忆原子。
        """
        normalized_id = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        return await self._memory_library.mid_term.get(normalized_id)

    async def list_memories(
        self,
        *,
        query: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[MemoryAtom]:
        """
        根据查询和过滤条件列出中期记忆原子。
        """
        if query:
            results = await self._memory_library.mid_term.search(
                query=query,
                top_k=limit,
                filters=filters,
            )
            return [result["memory"] for result in results if "memory" in result]
        return await self._memory_library.mid_term.scroll(
            filters=filters,
            limit=limit,
        )

    async def get_agent_profile(self, agent_alias: str) -> AgentProfile:
        """
        根据Agent别名读取Agent配置文件。
        """
        if not agent_alias or agent_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        try:
            atom = await self._memory_library.mid_term.get_by_alias(agent_alias)
            if atom is not None and atom.index.memory_type == MemoryType.AGENT_PROFILE:
                profile = AgentProfile.from_atom(atom)
                if profile is not None:
                    return profile
        except Exception as e:
            logger.warning(
                f"Failed to load agent profile '{agent_alias}' from memory library: {e}"
            )

        logger.info(
            f"Agent profile '{agent_alias}' not found, falling back to OMNI_DOLL_PROFILE."
        )
        return OMNI_DOLL_PROFILE

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        检索相关记忆，不填充 rendered_context（由调用方编译，Phase D）。
        """
        start_time = time.time()

        response = RetrievalResponse()

        try:
            # Step 1: 基础过滤条件 (identity 安全基线，不可被 MTP filter 覆盖)
            # 实现 MutiAgentSystem.md §3.3.1 Visibility Scope Filtering
            query_filters = QueryFilters(identity=request.identity)

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
            )

            engine_result = await self.engine.retrieve(query=query)

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
        """Async bus entrypoint — search + vitality refresh only."""
        response = await self.retrieve(request)
        await self._refresh_vitality_for_memories(response.memories)
        return response

    async def retrieve_by_aliases(
        self,
        aliases: List[str],
        identity: Optional[Identity] = None,
    ) -> RetrievalResponse:
        """
        精确按 alias 取回记忆。rendered_context 由调用方填充 (Phase D)。
        """
        start_time = time.time()
        response = RetrievalResponse()

        try:
            memories: List[MemoryAtom] = []
            seen_aliases: set[str] = set()
            user_id = identity.user_id if identity is not None else None

            for alias in aliases:
                normalized = alias.strip() if alias else ""
                if not normalized or normalized in seen_aliases:
                    continue
                seen_aliases.add(normalized)

                atom = await self._memory_library.mid_term.get_by_alias(normalized, user_id)
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
        aliases: List[str],
        identity: Optional[Identity] = None,
    ) -> RetrievalResponse:
        """Async bus entrypoint for exact alias retrieval."""
        response = await self.retrieve_by_aliases(aliases, identity)
        await self._refresh_vitality_for_memories(response.memories)
        return response

    async def update_access_stats(self, memories: List[MemoryAtom]) -> None:
        """
        更新被引用记忆的访问统计

        当记忆被成功使用时调用，增加访问计数
        """
        for memory in memories:
            try:
                await self._memory_library.mid_term.update_access_info(memory.id)
            except Exception as e:
                logger.warning(f"更新访问统计失败: {memory.id} - {e}")

    # ========== 长期记忆查询 ========== 

    async def query_archive(
        self,
        *,
        limit: int = 100,
        vitality_threshold: Optional[float] = None,
    ):
        """
        查询长期冷存储归档记录。
        """
        return await self._memory_library.long_term.query(
            limit=limit,
            vitality_threshold=vitality_threshold,
        )

    async def is_archived(self, memory_id) -> bool:
        """
        检查记忆是否已进入长期冷存储。
        """
        return await self._memory_library.long_term.is_archived(memory_id)

    # ========== 内部辅助方法 ========== 

    async def _refresh_vitality_for_memories(self, memories: List[MemoryAtom]) -> None:
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


__all__ = [
    "RetrievalFamiliar",
]
