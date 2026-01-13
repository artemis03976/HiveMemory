"""
记忆检索引擎门面

职责:
    整合所有检索组件，提供统一的检索入口

组件集成:
    - RetrievalRouter: 判断是否需要检索
    - QueryProcessor: 查询预处理
    - HybridRetriever: 混合检索
    - ContextRenderer: 上下文渲染

对应设计文档: PROJECT.md 第 5 章
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime
import time
import logging

if TYPE_CHECKING:
    from hivememory.core.config import MemoryRetrievalConfig

from hivememory.core.models import MemoryAtom
from hivememory.generation import ConversationMessage
from hivememory.retrieval.models import RetrievalResult, ProcessedQuery, SearchResults, RenderFormat
from hivememory.retrieval.query import QueryProcessor, create_default_processor
from hivememory.retrieval.router import RetrievalRouter, create_default_router
from hivememory.retrieval.retriever import HybridRetriever, create_default_retriever
from hivememory.retrieval.renderer import ContextRenderer, create_default_renderer
from hivememory.retrieval.interfaces import RetrievalEngine
from hivememory.memory.storage import QdrantMemoryStore

logger = logging.getLogger(__name__)


class MemoryRetrievalEngine(RetrievalEngine):
    """
    记忆检索引擎

    统一的检索入口，整合路由、查询处理、检索、渲染四大组件

    使用示例:
        ```python
        engine = MemoryRetrievalEngine(storage=my_storage)
        result = engine.retrieve_context(
            query="我之前设置的 API Key 是什么？",
            user_id="user_123"
        )

        if not result.is_empty():
            print(result.rendered_context)
        ```
    """
    
    def __init__(
        self,
        storage: QdrantMemoryStore,
        router: Optional[RetrievalRouter] = None,
        processor: Optional[QueryProcessor] = None,
        retriever: Optional[HybridRetriever] = None,
        renderer: Optional[ContextRenderer] = None,
        config: Optional["MemoryRetrievalConfig"] = None,
    ):
        """
        初始化检索引擎

        Args:
            storage: QdrantMemoryStore 实例
            router: 检索路由器（可选，使用配置）
            processor: 查询处理器（可选，使用配置）
            retriever: 混合检索器（可选，使用配置）
            renderer: 上下文渲染器（可选，使用配置）
            config: 记忆检索配置（可选，用于创建组件）

        Examples:
            >>> # 使用默认配置
            >>> engine = MemoryRetrievalEngine(storage=storage)
            >>>
            >>> # 使用自定义配置
            >>> from hivememory.core.config import MemoryRetrievalConfig
            >>> config = MemoryRetrievalConfig()
            >>> engine = MemoryRetrievalEngine(storage=storage, config=config)
        """
        self.storage = storage

        # 使用传入的配置或加载默认配置
        if config is None:
            from hivememory.core.config import MemoryRetrievalConfig
            config = MemoryRetrievalConfig()

        self.enable_routing = config.enable_routing

        if router is None:
            self.router = create_default_router(config.router)
        else:
            self.router = router

        if processor is None:
            self.processor = create_default_processor(config.processor)
        else:
            self.processor = processor

        if retriever is None:
            self.retriever = create_default_retriever(storage, config.retriever)
        else:
            self.retriever = retriever

        if renderer is None:
            self.renderer = create_default_renderer(config.renderer)
        else:
            self.renderer = renderer

        logger.info("MemoryRetrievalEngine 初始化完成")
    
    def retrieve_context(
        self,
        query: str,
        user_id: str,
        context: Optional[List[ConversationMessage]] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        force_retrieve: bool = False
    ) -> RetrievalResult:
        """
        检索相关记忆并渲染上下文
        
        完整流程:
        1. 路由判断（是否需要检索）
        2. 查询预处理（提取过滤条件）
        3. 混合检索（向量 + 元数据）
        4. 上下文渲染（XML/Markdown）
        
        Args:
            query: 用户查询
            user_id: 用户 ID（用于过滤）
            context: 对话上下文（可选）
            top_k: 返回数量（可选）
            score_threshold: 相似度阈值（可选）
            force_retrieve: 强制检索（跳过路由判断）
            
        Returns:
            RetrievalResult 对象
        """
        start_time = time.time()

        result = RetrievalResult()
        
        try:
            # Step 1: 路由判断
            if self.enable_routing and not force_retrieve:
                should_retrieve = self.router.should_retrieve(query, context)
                result.router_decision = should_retrieve
                
                if not should_retrieve:
                    result.should_retrieve = False
                    result.latency_ms = (time.time() - start_time) * 1000
                    logger.debug(f"路由器判断无需检索: '{query[:30]}...'")
                    return result
            
            # Step 2: 查询预处理
            processed_query = self.processor.process(
                query=query,
                context=context,
                user_id=user_id
            )
            result.processor_output = processed_query
            result.query_used = processed_query.semantic_query
            
            # Step 3: 执行检索
            retriever_results = self.retriever.retrieve(
                query=processed_query,
                top_k=top_k,
                score_threshold=score_threshold
            )
            result.retriever_results = retriever_results
            result.memories = retriever_results.get_memories()
            result.memories_count = len(result.memories)
            
            # Step 4: 渲染上下文
            if not retriever_results.is_empty():
                result.rendered_context = self.renderer.render(retriever_results.results)
            
            result.latency_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"检索完成: query='{query[:20]}...', "
                f"memories={result.memories_count}, "
                f"latency={result.latency_ms:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            result.latency_ms = (time.time() - start_time) * 1000
        
        return result
    
    def retrieve_memories(
        self,
        query_text: str,
        user_id: str,
        top_k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[MemoryAtom]:
        """
        简化的记忆搜索接口
        
        跳过路由和渲染，直接返回记忆列表
        
        Args:
            query_text: 查询文本
            user_id: 用户 ID
            top_k: 返回数量
            memory_type: 记忆类型过滤
            
        Returns:
            MemoryAtom 列表
        """
        # 使用处理器处理查询
        processed_query = self.processor.process(
            query=query_text,
            user_id=user_id
        )
        
        # 如果有记忆类型，覆盖过滤条件
        if memory_type:
            try:
                from hivememory.core.models import MemoryType
                processed_query.filters.memory_type = MemoryType(memory_type)
            except ValueError:
                logger.warning(f"无效的记忆类型: {memory_type}")
        
        # 执行搜索
        results = self.retriever.retrieve(
            query=processed_query,
            top_k=top_k
        )
        return results.get_memories()
    
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
    "MemoryRetrievalEngine",
]
