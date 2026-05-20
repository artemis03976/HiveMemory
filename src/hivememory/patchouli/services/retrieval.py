"""
帕秋莉·检索使魔 (The Retrieval Familiar of Patchouli)

定位：服务员与执行者
职责：
    - 混合检索 (Dense + Sparse + RRF)
    - 重排序 (Reranking)
    - 上下文渲染
    - 访问统计更新

基于原 MemoryRetrievalEngine (engines/retrieval/engine.py) 改造

作者: HiveMemory Team
版本: 2.2 (乐观检索策略)
"""

from typing import List, Optional, TYPE_CHECKING
import time
import logging

if TYPE_CHECKING:
    from hivememory.system.config import MemoryRetrievalConfig

from hivememory.core.models import Identity, MemoryAtom
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.interfaces import BaseContextRenderer
from hivememory.engines.retrieval.models import RetrievalQuery, QueryFilters
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.core.mtp.exceptions import StorageOfflineError, StorageReadError
from hivememory.core.protocol.models import RetrievalRequest, RetrievalResponse

logger = logging.getLogger(__name__)

class RetrievalFamiliar:
    """
    帕秋莉·检索使魔 (The Retrieval Familiar of Patchouli)

    当"真理之眼"确认需要查书时，帕秋莉会召唤使魔去书架取书。
    这是一个即时响应的动作（Hot Path），没有复杂的思考，只有精准的执行。

    特性：
        - 同步阻塞
        - 高并发
        - 本地计算密集

    职责：
        1. 接收业务请求 (RetrievalRequest)
        2. 根据 user_id 创建过滤条件 (乐观检索策略)
        3. 调用 RetrievalEngine 进行数据检索
        4. 处理副作用 (如统计更新)

    乐观检索策略：
        - 不再从 Gateway 接收 filters
        - 仅根据 user_id 创建过滤条件
        - 让 Reranker 来排序所有相关记忆

    使用示例:
        ```python
        from hivememory.patchouli.services.retrieval import RetrievalFamiliar
        from hivememory.engines.retrieval.engine import RetrievalEngine
        # ...
        engine = RetrievalEngine(retriever=..., renderer=...)
        familiar = RetrievalFamiliar(engine=engine, storage=...)

        result = familiar.retrieve(request)
        ```
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        engine: RetrievalEngine,
        passive_renderer: Optional[BaseContextRenderer] = None,
    ):
        """
        初始化检索使魔

        Args:
            storage: QdrantMemoryStore 实例 (用于更新统计)
            engine: 检索引擎实例
            passive_renderer: 被动模式渲染器 (FullContextRenderer)，
                              用于 Passive Observer Mode 的上下文降级渲染
        """
        self.storage = storage
        self.engine = engine
        self._passive_renderer = passive_renderer

        logger.info("RetrievalFamiliar (检索使魔) 初始化完成")

    def retrieve(self, request: RetrievalRequest, mode: str = "active") -> RetrievalResponse:
        """
        检索相关记忆

        完整流程:
        1. 根据 user_id 创建基础过滤条件 (安全基线)
        2. 合并 MTP filter 传入的额外过滤维度 (如 type:CODE)
        3. 构建查询对象 (RetrievalQuery)
        4. 调用 Engine 执行检索
        5. 根据 mode 选择渲染策略:
           - active: 使用 Engine 内置 renderer (CompactContext/Cascade)
           - passive: 使用 FullContextRenderer 降级渲染 (Passive.md §5.2)

        Args:
            request: 检索请求协议消息
            mode: 运行模式 ("active" | "passive")

        Returns:
            RetrievalResponse 对象
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

            engine_result = self.engine.retrieve(query=query)

            response.memories = engine_result.memories
            response.memories_count = engine_result.memories_count
            response.latency_ms = engine_result.latency_ms

            # Step 5: 渲染策略分流 (Passive.md §5.2)
            if (
                mode == "passive"
                and self._passive_renderer is not None
                and engine_result.search_results
                and not engine_result.search_results.is_empty()
            ):
                # 被动模式: 使用 FullContextRenderer 降级渲染
                # 外部 Agent 不懂 MTP，无法使用 READ 获取详情，
                # 因此直接注入完整 Payload 文本
                response.rendered_context = self._passive_renderer.render(
                    engine_result.search_results.results
                )
            else:
                response.rendered_context = engine_result.rendered_context

            logger.info(
                f"检索完成: query='{request.semantic_query[:20]}...', "
                f"mode={mode}, filters={query_filters}, "
                f"使魔取回了 {response.memories_count} 条记忆, "
                f"latency={response.latency_ms:.1f}ms"
            )

        except (StorageOfflineError, StorageReadError):
            raise  # Must propagate to Koakuma's _route_and_execute
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            response.latency_ms = (time.time() - start_time) * 1000

        return response

    def retrieve_by_aliases(
        self,
        aliases: List[str],
        identity: Optional[Identity] = None,
        mode: str = "active",
    ) -> RetrievalResponse:
        """
        精确按 alias 取回记忆并复用 Retrieval renderer 渲染上下文。

        该入口与 retrieve() 并列：retrieve() 负责语义检索，retrieve_by_aliases()
        负责已知 alias 的精确取回。二者最终都返回 RetrievalResponse.rendered_context。
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

                atom = self.storage.get_memory_by_alias(normalized, user_id)
                if atom is None:
                    logger.warning(f"Alias not found during alias retrieval: {normalized}")
                    continue
                memories.append(atom)

            response.memories = memories
            response.memories_count = len(memories)

            if memories:
                if mode == "passive" and self._passive_renderer is not None:
                    response.rendered_context = self._passive_renderer.render(memories)
                else:
                    response.rendered_context = self.engine.render_memories(memories)

            response.latency_ms = (time.time() - start_time) * 1000

        except (StorageOfflineError, StorageReadError):
            raise
        except Exception as e:
            logger.error(f"Alias retrieval failed: {e}", exc_info=True)
            response.latency_ms = (time.time() - start_time) * 1000

        return response

    def update_access_stats(self, memories: List[MemoryAtom]) -> None:
        """
        更新被引用记忆的访问统计

        当记忆被成功使用时调用，增加访问计数
        """
        for memory in memories:
            try:
                self.storage.update_access_info(memory.id)
            except Exception as e:
                logger.warning(f"更新访问统计失败: {memory.id} - {e}")


__all__ = [
    "RetrievalFamiliar",
]
