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

from typing import List, Optional, TYPE_CHECKING
import time
import logging

if TYPE_CHECKING:
    from hivememory.patchouli.config import MemoryRetrievalConfig

from hivememory.core.models import MemoryAtom
from hivememory.engines.retrieval.models import RetrievalQuery, QueryFilters
from hivememory.engines.retrieval.retriever import create_default_retriever
from hivememory.engines.retrieval.renderer import create_default_renderer
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.infrastructure.rerank import BaseRerankService
from hivememory.patchouli.protocol.models import RetrievalRequest, RetrievalResult

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
        1. 并行召回 (Parallel Recall)：
           - Dense: 使用 Rewritten Query 进行向量检索
           - Sparse: 使用 Keywords 进行 BM25 检索
           - Filter: 应用 Type 或 Source 过滤
        2. 融合 (Fusion)：使用 RRF (Reciprocal Rank Fusion) 合并多路结果
        3. 精排 (Reranking)：引入 Cross-Encoder Reranker 对 Top-N 结果进行语义重打分
        4. 上下文渲染：将 JSON 转换为 XML/Markdown 格式

    使用示例:
        ```python
        from hivememory.patchouli.retrieval_familiar import RetrievalFamiliar
        from hivememory.infrastructure.storage import QdrantMemoryStore
        from hivememory.patchouli.protocol.models import RetrievalRequest

        storage = QdrantMemoryStore()
        familiar = RetrievalFamiliar(storage=storage)

        request = RetrievalRequest(
            semantic_query="我之前设置的 API Key 是什么？",
            user_id="user_123"
        )
        result = familiar.retrieve(request)

        if not result.is_empty():
            print(result.rendered_context)
        ```
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        reranker_service: Optional[BaseRerankService] = None,
        config: Optional["MemoryRetrievalConfig"] = None,
    ):
        """
        初始化检索使魔

        Args:
            storage: QdrantMemoryStore 实例
            reranker_service: Rerank 服务实例
            config: 记忆检索配置（可选，用于创建组件）

        Examples:
            >>> # 使用默认配置
            >>> familiar = RetrievalFamiliar(storage=storage)
            >>>
            >>> # 使用自定义配置
            >>> from hivememory.patchouli.config import MemoryRetrievalConfig
            >>> config = MemoryRetrievalConfig()
            >>> familiar = RetrievalFamiliar(storage=storage, config=config)
        """
        self.storage = storage

        # 使用传入的配置或加载默认配置
        if config:
            self.config = config
        else:
            from hivememory.patchouli.config import MemoryRetrievalConfig
            self.config = MemoryRetrievalConfig()

        # 初始化检索器和渲染器
        # 注意: 路由器和查询处理器已废弃，由 GlobalGateway 接管
        self.retriever = create_default_retriever(
            storage, 
            self.config.retriever, 
            reranker_service=reranker_service
        )
        self.renderer = create_default_renderer(self.config.renderer)

        logger.info("RetrievalFamiliar 初始化完成")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """
        检索相关记忆

        完整流程:
        1. 构建查询对象 (RetrievalQuery)
        2. 混合检索（向量 + 元数据）
        3. 上下文渲染（XML/Markdown）

        Args:
            request: 检索请求协议消息

        Returns:
            RetrievalResult 对象
        """
        start_time = time.time()

        result = RetrievalResult()

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

            # Step 2: 执行检索
            retriever_results = self.retriever.retrieve(
                query=query,
            )
  
            result.memories = retriever_results.get_memories()
            result.memories_count = len(result.memories)

            # Step 3: 渲染上下文
            if not retriever_results.is_empty():
                result.rendered_context = self.renderer.render(retriever_results.results)

            result.latency_ms = (time.time() - start_time) * 1000

            logger.info(
                f"检索完成: query='{request.semantic_query[:20]}...', "
                f"使魔取回了 {result.memories_count} 条记忆, "
                f"latency={result.latency_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            result.latency_ms = (time.time() - start_time) * 1000

        return result

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
