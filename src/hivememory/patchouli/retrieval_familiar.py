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
版本: 2.1
"""

from typing import List, TYPE_CHECKING
import time
import logging

if TYPE_CHECKING:
    from hivememory.patchouli.config import MemoryRetrievalConfig

from hivememory.core.models import MemoryAtom
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.models import RetrievalQuery
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.patchouli.protocol.models import RetrievalRequest, RetrievalResponse

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
        2. 调用 RetrievalEngine 进行数据检索
        3. 处理副作用 (如统计更新)

    使用示例:
        ```python
        from hivememory.patchouli.retrieval_familiar import RetrievalFamiliar
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
    ):
        """
        初始化检索使魔

        Args:
            storage: QdrantMemoryStore 实例 (用于更新统计)
            engine: 检索引擎实例
        """
        self.storage = storage
        self.engine = engine

        logger.info("RetrievalFamiliar (检索使魔) 初始化完成")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        检索相关记忆

        完整流程:
        1. 构建查询对象 (RetrievalQuery)
        2. 调用 Engine 执行检索
        3. Engine 内部完成上下文渲染

        Args:
            request: 检索请求协议消息

        Returns:
            RetrievalResponse 对象
        """
        start_time = time.time()

        response = RetrievalResponse()

        try:
            # Step 1: 构建查询对象
            query_filters = request.filters.model_copy()
            if request.user_id and not query_filters.user_id:
                query_filters.user_id = request.user_id

            # 构建 RetrievalQuery
            query = RetrievalQuery(
                semantic_query=request.semantic_query,
                keywords=request.keywords or [],
                filters=query_filters,
            )

            engine_result = self.engine.retrieve(query=query)

            response.memories = engine_result.memories
            response.memories_count = engine_result.memories_count
            response.rendered_context = engine_result.rendered_context
            response.latency_ms = engine_result.latency_ms

            logger.info(
                f"检索完成: query='{request.semantic_query[:20]}...', "
                f"使魔取回了 {response.memories_count} 条记忆, "
                f"latency={response.latency_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
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
