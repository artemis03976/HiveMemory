"""
HiveMemory - Retrieval 模块接口抽象层

定义了记忆检索模块的所有核心接口，遵循依赖倒置原则，便于扩展和测试。

接口列表:
- RetrievalRouter: 检索路由器接口
- QueryProcessor: 查询预处理器接口
- MemorySearcher: 记忆检索器接口
- ContextRenderer: 上下文渲染器接口
- RetrievalEngine: 检索引擎接口

作者: HiveMemory Team
版本: 0.1.0
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum

from hivememory.core.models import MemoryAtom
from hivememory.generation.models import ConversationMessage


# ========== 接口定义 ==========

class RetrievalRouter(ABC):
    """
    检索路由器接口

    职责:
        判断用户查询是否需要检索记忆。

    实现策略:
        - SimpleRouter: 基于规则的关键词匹配
        - LLMRouter: 基于轻量级 LLM 的智能分类
    """

    @abstractmethod
    def should_retrieve(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None
    ) -> bool:
        """
        判断是否需要检索记忆

        Args:
            query: 用户查询
            context: 对话上下文

        Returns:
            bool: True 表示需要检索

        Examples:
            >>> router = SimpleRouter()
            >>> router.should_retrieve("我之前设置的API Key是什么？")
            True
        """
        pass


class QueryProcessor(ABC):
    """
    查询预处理器接口

    职责:
        重写和扩展用户查询，提取过滤条件。

    实现策略:
        - 解析时间表达式（"昨天"、"最近3天"）
        - 识别记忆类型（代码、事实、URL等）
        - 提取结构化过滤条件
        - 可选的 LLM 查询改写
    """

    @abstractmethod
    def process(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None,
        user_id: Optional[str] = None
    ) -> "ProcessedQuery":
        """
        处理查询

        Args:
            query: 原始查询
            context: 对话上下文（可选）
            user_id: 用户 ID（用于过滤）

        Returns:
            ProcessedQuery: 处理后的查询对象

        Examples:
            >>> processor = QueryProcessor()
            >>> processed = processor.process("昨天的代码", user_id="user123")
            >>> print(processed.semantic_query)
            "昨天的代码"
        """
        pass

    @abstractmethod
    def has_context_reference(self, query: str) -> bool:
        """
        检查查询是否包含上下文引用

        Args:
            query: 查询文本

        Returns:
            bool: True 表示包含历史上下文引用
        """
        pass


class MemorySearcher(ABC):
    """
    记忆检索器接口

    职责:
        执行混合检索（向量 + BM25 + 结构化过滤）。

    实现策略:
        - 向量检索 (Qdrant Dense Vector)
        - 元数据过滤 (类型、标签、时间、置信度)
        - 结果排序和打分
        - 可选的 RRF 融合和 Reranking
    """

    @abstractmethod
    def search(
        self,
        query: "ProcessedQuery",
        top_k: int = 5,
        score_threshold: float = 0.75
    ) -> "SearchResults":
        """
        检索记忆

        Args:
            query: 处理后的查询
            top_k: 返回数量
            score_threshold: 相似度阈值

        Returns:
            SearchResults: 检索结果集合

        Examples:
            >>> searcher = HybridSearcher(storage)
            >>> results = searcher.search(processed_query, top_k=5)
            >>> for result in results:
            ...     print(result.memory.index.title)
        """
        pass


class RenderFormat(str, Enum):
    """
    渲染格式枚举

    Attributes:
        XML: XML 标签格式
        MARKDOWN: Markdown 格式
    """
    XML = "xml"
    MARKDOWN = "markdown"


class ContextRenderer(ABC):
    """
    上下文渲染器接口

    职责:
        将检索到的记忆渲染为适合注入 LLM Context 的格式。

    实现策略:
        - XML 标签格式（Claude/GPT-4 推荐）
        - Markdown 格式（通用）
        - 极简格式（最小化 Token）
    """

    @abstractmethod
    def render(
        self,
        results: List,
        render_format: Optional["RenderFormat"] = None
    ) -> str:
        """
        渲染记忆列表为上下文字符串

        Args:
            results: SearchResult 列表或 MemoryAtom 列表
            render_format: 输出格式（可选，覆盖默认）

        Returns:
            str: 渲染后的上下文字符串

        Examples:
            >>> renderer = ContextRenderer(format=RenderFormat.XML)
            >>> context = renderer.render(search_results.results)
            >>> print(context)
            <system_memory_context>...
        """
        pass


class RetrievalEngine(ABC):
    """
    检索引擎接口

    职责:
        整合所有检索组件，提供统一的检索入口。

    实现策略:
        - 路由判断
        - 查询预处理
        - 混合检索
        - 上下文渲染
    """

    @abstractmethod
    def retrieve_context(
        self,
        query: str,
        user_id: str,
        context: Optional[List[ConversationMessage]] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        force_retrieve: bool = False
    ) -> "RetrievalResult":
        """
        检索相关记忆并渲染上下文

        Args:
            query: 用户查询
            user_id: 用户 ID
            context: 对话上下文
            top_k: 返回数量
            score_threshold: 相似度阈值
            force_retrieve: 强制检索（跳过路由判断）

        Returns:
            RetrievalResult: 完整的检索结果

        Examples:
            >>> engine = RetrievalEngine(storage)
            >>> result = engine.retrieve_context(
            ...     query="我之前设置的 API Key 是什么？",
            ...     user_id="user_123"
            ... )
            >>> if not result.is_empty():
            ...     print(result.rendered_context)
        """
        pass

    @abstractmethod
    def search_memories(
        self,
        query_text: str,
        user_id: str,
        top_k: int = 5,
        memory_type: Optional[str] = None
    ) -> List[MemoryAtom]:
        """
        简化的记忆搜索接口

        Args:
            query_text: 查询文本
            user_id: 用户 ID
            top_k: 返回数量
            memory_type: 记忆类型过滤

        Returns:
            List[MemoryAtom]: 记忆原子列表
        """
        pass


# ========== 异常定义 ==========

class RetrievalError(Exception):
    """检索异常基类"""
    pass


class QueryProcessingError(RetrievalError):
    """查询处理失败异常"""
    pass


class SearchError(RetrievalError):
    """检索失败异常"""
    pass


class RenderError(RetrievalError):
    """渲染失败异常"""
    pass


# ========== 导出列表 ==========

__all__ = [
    # 接口
    "RetrievalRouter",
    "QueryProcessor",
    "MemorySearcher",
    "ContextRenderer",
    "RetrievalEngine",

    # 枚举
    "RenderFormat",

    # 异常
    "RetrievalError",
    "QueryProcessingError",
    "SearchError",
    "RenderError",
]
