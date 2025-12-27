"""
检索引擎门面

职责:
    整合所有检索组件，提供统一的检索入口

组件集成:
    - RetrievalRouter: 判断是否需要检索
    - QueryProcessor: 查询预处理
    - HybridSearcher: 混合检索
    - ContextRenderer: 上下文渲染

对应设计文档: PROJECT.md 第 5 章
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import logging

from hivememory.core.models import ConversationMessage, MemoryAtom
from hivememory.retrieval.query import ProcessedQuery, QueryProcessor
from hivememory.retrieval.router import SimpleRouter, RetrievalRouter
from hivememory.retrieval.searcher import HybridSearcher, SearchResults
from hivememory.retrieval.renderer import ContextRenderer, RenderFormat
from hivememory.retrieval.interfaces import RetrievalEngine as RetrievalEngineInterface

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    检索结果封装
    
    包含完整的检索信息和渲染后的上下文
    """
    should_retrieve: bool = True  # 路由器判断结果
    memories: List[MemoryAtom] = field(default_factory=list)  # 检索到的记忆
    rendered_context: str = ""  # 渲染后的上下文字符串
    
    # 元信息
    latency_ms: float = 0.0  # 总耗时
    query_used: str = ""  # 实际使用的查询
    memories_count: int = 0  # 检索到的数量
    
    # 调试信息
    router_decision: bool = False
    processor_output: Optional[ProcessedQuery] = None
    search_results: Optional[SearchResults] = None
    
    def is_empty(self) -> bool:
        """检查是否没有检索到任何记忆"""
        return len(self.memories) == 0
    
    def get_context_for_prompt(self) -> str:
        """获取可直接注入 System Prompt 的上下文"""
        if self.is_empty():
            return ""
        return self.rendered_context


class RetrievalEngine(RetrievalEngineInterface):
    """
    记忆检索引擎

    统一的检索入口，整合路由、查询处理、检索、渲染四大组件

    使用示例:
        ```python
        engine = RetrievalEngine(storage=my_storage)
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
        storage,  # QdrantMemoryStore
        router: Optional[RetrievalRouter] = None,
        processor: Optional[QueryProcessor] = None,
        searcher: Optional[HybridSearcher] = None,
        renderer: Optional[ContextRenderer] = None,
        # 默认配置
        enable_routing: bool = True,
        default_top_k: int = 5,
        default_threshold: float = 0.3,
        render_format: RenderFormat = RenderFormat.XML,
        max_context_tokens: int = 2000
    ):
        """
        初始化检索引擎
        
        Args:
            storage: QdrantMemoryStore 实例
            router: 检索路由器（可选，默认 SimpleRouter）
            processor: 查询处理器（可选，默认 QueryProcessor）
            searcher: 混合检索器（可选，自动创建）
            renderer: 上下文渲染器（可选，自动创建）
            enable_routing: 是否启用路由判断
            default_top_k: 默认返回数量
            default_threshold: 默认相似度阈值
            render_format: 渲染格式
            max_context_tokens: 最大上下文长度
        """
        self.storage = storage
        self.enable_routing = enable_routing
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold
        
        # 初始化组件
        self.router = router or SimpleRouter()
        self.processor = processor or QueryProcessor()
        self.searcher = searcher or HybridSearcher(
            storage=storage,
            default_top_k=default_top_k,
            default_threshold=default_threshold
        )
        self.renderer = renderer or ContextRenderer(
            render_format=render_format,
            max_tokens=max_context_tokens
        )
        
        logger.info("RetrievalEngine 初始化完成")
    
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
        
        top_k = top_k or self.default_top_k
        score_threshold = score_threshold or self.default_threshold
        
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
            search_results = self.searcher.search(
                query=processed_query,
                top_k=top_k,
                score_threshold=score_threshold
            )
            result.search_results = search_results
            result.memories = search_results.get_memories()
            result.memories_count = len(result.memories)
            
            # Step 4: 渲染上下文
            if not search_results.is_empty():
                result.rendered_context = self.renderer.render(search_results.results)
            
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
    
    def search_memories(
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
        results = self.searcher.search_by_text(
            query_text=query_text,
            user_id=user_id,
            top_k=top_k,
            memory_type=memory_type
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


# 便捷函数

def create_retrieval_engine(
    storage,
    enable_routing: bool = True,
    top_k: int = 5,
    threshold: float = 0.6,
    render_format: str = "xml"
) -> RetrievalEngine:
    """
    创建检索引擎的便捷函数
    
    Args:
        storage: QdrantMemoryStore 实例
        enable_routing: 是否启用路由判断
        top_k: 默认返回数量
        threshold: 默认相似度阈值
        render_format: 渲染格式（"xml" 或 "markdown"）
        
    Returns:
        RetrievalEngine 实例
    """
    fmt = RenderFormat.XML if render_format.lower() == "xml" else RenderFormat.MARKDOWN
    
    return RetrievalEngine(
        storage=storage,
        enable_routing=enable_routing,
        default_top_k=top_k,
        default_threshold=threshold,
        render_format=fmt
    )


__all__ = [
    "RetrievalResult",
    "RetrievalEngine",
    "create_retrieval_engine",
]
