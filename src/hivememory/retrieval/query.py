"""
查询预处理模块

职责:
1. 解析用户查询，提取语义查询文本
2. 识别结构化过滤条件（时间、类型、Agent等）
3. 可选的 LLM 驱动查询改写

对应设计文档: PROJECT.md 5.1 节
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import re
import logging

from hivememory.core.models import MemoryType
from hivememory.generation.models import ConversationMessage
from hivememory.retrieval.interfaces import QueryProcessor as QueryProcessorInterface

from hivememory.retrieval.models import QueryFilters, ProcessedQuery

logger = logging.getLogger(__name__)

# ========== 时间表达式解析 ==========

class TimeExpressionParser:
    """
    时间表达式解析器
    
    支持中英文时间表达式：
    - "昨天"、"yesterday"
    - "上周"、"last week"
    - "最近3天"、"recent 3 days"
    """
    
    # 中文时间模式
    CN_PATTERNS = {
        r"今天|今日": lambda: (
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            datetime.now()
        ),
        r"昨天|昨日": lambda: (
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1),
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ),
        r"前天": lambda: (
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2),
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        ),
        r"这周|本周": lambda: (
            datetime.now() - timedelta(days=datetime.now().weekday()),
            datetime.now()
        ),
        r"上周": lambda: (
            datetime.now() - timedelta(days=datetime.now().weekday() + 7),
            datetime.now() - timedelta(days=datetime.now().weekday())
        ),
        r"这个月|本月": lambda: (
            datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            datetime.now()
        ),
        r"上个月": lambda: (
            (datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1),
            datetime.now().replace(day=1) - timedelta(days=1)
        ),
        r"最近(\d+)天": lambda m: (
            datetime.now() - timedelta(days=int(m.group(1))),
            datetime.now()
        ),
        r"最近(\d+)周": lambda m: (
            datetime.now() - timedelta(weeks=int(m.group(1))),
            datetime.now()
        ),
        r"(\d+)天前": lambda m: (
            datetime.now() - timedelta(days=int(m.group(1)) + 1),
            datetime.now() - timedelta(days=int(m.group(1)))
        ),
    }
    
    # 英文时间模式
    EN_PATTERNS = {
        r"today": lambda: (
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            datetime.now()
        ),
        r"yesterday": lambda: (
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1),
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ),
        r"this week": lambda: (
            datetime.now() - timedelta(days=datetime.now().weekday()),
            datetime.now()
        ),
        r"last week": lambda: (
            datetime.now() - timedelta(days=datetime.now().weekday() + 7),
            datetime.now() - timedelta(days=datetime.now().weekday())
        ),
        r"recent(?:ly)?\s*(\d+)\s*days?": lambda m: (
            datetime.now() - timedelta(days=int(m.group(1))),
            datetime.now()
        ),
        r"(\d+)\s*days?\s*ago": lambda m: (
            datetime.now() - timedelta(days=int(m.group(1)) + 1),
            datetime.now() - timedelta(days=int(m.group(1)))
        ),
    }
    
    @classmethod
    def parse(cls, text: str) -> Optional[Tuple[datetime, datetime]]:
        """
        解析文本中的时间表达式
        
        Args:
            text: 输入文本
            
        Returns:
            时间范围元组 (start, end)，如果未识别到则返回 None
        """
        text_lower = text.lower()
        
        # 先尝试中文模式
        for pattern, resolver in cls.CN_PATTERNS.items():
            match = re.search(pattern, text)
            if match:
                try:
                    if callable(resolver):
                        # 检查是否需要传入 match 对象
                        import inspect
                        sig = inspect.signature(resolver)
                        if len(sig.parameters) > 0:
                            return resolver(match)
                        return resolver()
                except Exception as e:
                    logger.debug(f"时间解析失败: {pattern} -> {e}")
                    continue
        
        # 再尝试英文模式
        for pattern, resolver in cls.EN_PATTERNS.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    import inspect
                    sig = inspect.signature(resolver)
                    if len(sig.parameters) > 0:
                        return resolver(match)
                    return resolver()
                except Exception as e:
                    logger.debug(f"时间解析失败: {pattern} -> {e}")
                    continue
        
        return None


# ========== 类型识别 ==========

class MemoryTypeDetector:
    """
    记忆类型检测器
    
    基于关键词识别用户想要查找的记忆类型
    """
    
    # 类型关键词映射
    TYPE_KEYWORDS = {
        MemoryType.CODE_SNIPPET: [
            "代码", "函数", "方法", "类", "code", "function", "class", "method",
            "实现", "算法", "脚本", "snippet", "implementation"
        ],
        MemoryType.FACT: [
            "事实", "规则", "定义", "参数", "配置", "fact", "rule", "definition",
            "parameter", "config", "设置", "setting"
        ],
        MemoryType.URL_RESOURCE: [
            "链接", "网址", "文档", "资源", "url", "link", "document", "resource",
            "网页", "webpage", "文章", "article"
        ],
        MemoryType.REFLECTION: [
            "总结", "经验", "反思", "教训", "summary", "reflection", "lesson",
            "学到", "learned", "结论", "conclusion"
        ],
        MemoryType.USER_PROFILE: [
            "偏好", "习惯", "喜欢", "preference", "habit", "profile",
            "个人", "personal", "用户", "user"
        ],
    }
    
    @classmethod
    def detect(cls, text: str) -> Optional[MemoryType]:
        """
        检测文本中暗示的记忆类型
        
        Args:
            text: 输入文本
            
        Returns:
            检测到的类型，未识别返回 None
        """
        text_lower = text.lower()
        
        for memory_type, keywords in cls.TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return memory_type
        
        return None


# ========== 查询处理器 ==========

class QueryProcessor(QueryProcessorInterface):
    """
    查询预处理器

    职责:
    1. 解析时间表达式
    2. 识别记忆类型
    3. 提取关键词
    4. 可选的 LLM 查询改写
    """

    # 需要移除的停用词
    STOPWORDS = {
        "的", "了", "是", "在", "我", "你", "他", "她", "它", "们",
        "这", "那", "有", "和", "与", "或", "但", "因为", "所以",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "can", "may", "might", "must",
        "what", "which", "who", "when", "where", "why", "how",
        "找", "查", "搜索", "检索", "请", "帮我", "帮忙",
    }

    # 上下文引用关键词（表示需要历史记忆）
    CONTEXT_KEYWORDS = [
        "之前", "刚才", "上次", "昨天", "前面", "earlier", "before",
        "previous", "last", "那个", "那段", "项目里", "系统中",
        "记得", "remember", "提到", "mentioned", "说过", "told",
    ]
    
    def __init__(self, enable_llm_rewrite: bool = False, llm_config: Optional[Dict] = None):
        """
        初始化查询处理器
        
        Args:
            enable_llm_rewrite: 是否启用 LLM 查询改写
            llm_config: LLM 配置（如果启用改写）
        """
        self.enable_llm_rewrite = enable_llm_rewrite
        self.llm_config = llm_config or {}
    
    def process(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None,
        user_id: Optional[str] = None
    ) -> ProcessedQuery:
        """
        处理用户查询
        
        Args:
            query: 原始查询文本
            context: 对话上下文（可选）
            user_id: 用户 ID（用于过滤）
            
        Returns:
            处理后的结构化查询
        """
        # 1. 初始化过滤条件
        filters = QueryFilters(user_id=user_id)
        
        # 2. 解析时间表达式
        time_range = TimeExpressionParser.parse(query)
        if time_range:
            filters.time_range = time_range
            logger.debug(f"检测到时间范围: {time_range}")
        
        # 3. 识别记忆类型
        memory_type = MemoryTypeDetector.detect(query)
        if memory_type:
            filters.memory_type = memory_type
            logger.debug(f"检测到记忆类型: {memory_type}")
        
        # 4. 提取关键词
        keywords = self._extract_keywords(query)
        
        # 5. 构建语义查询
        semantic_query = self._build_semantic_query(query, context)
        
        # 6. 可选的 LLM 改写
        rewritten = False
        if self.enable_llm_rewrite:
            try:
                semantic_query = self._llm_rewrite(semantic_query, context)
                rewritten = True
            except Exception as e:
                logger.warning(f"LLM 改写失败，使用原始查询: {e}")
        
        return ProcessedQuery(
            semantic_query=semantic_query,
            original_query=query,
            keywords=keywords,
            filters=filters,
            rewritten=rewritten
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词
        
        简单实现：分词后过滤停用词
        """
        # 简单的分词（按空格和标点）
        tokens = re.split(r'[\s\,\.\?\!，。？！、]+', text)
        
        # 过滤停用词和短词
        keywords = [
            t.strip() for t in tokens
            if t.strip() and len(t.strip()) > 1 and t.lower() not in self.STOPWORDS
        ]
        
        return keywords[:10]  # 最多保留10个关键词
    
    def _build_semantic_query(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None
    ) -> str:
        """
        构建语义查询
        
        如果有上下文，尝试补全指代
        """
        # 简单实现：直接返回原始查询
        # TODO: 高级实现可以使用 LLM 补全指代
        
        # 如果查询太短且有上下文，尝试从上下文补充
        if len(query) < 10 and context and len(context) > 0:
            # 取最近一条用户消息作为补充
            for msg in reversed(context[-3:]):
                if msg.role == "user" and len(msg.content) > len(query):
                    return f"{msg.content} {query}"
        
        return query
    
    def _llm_rewrite(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None
    ) -> str:
        """
        使用 LLM 改写查询
        
        将口语化的查询改写为更适合检索的形式
        """
        # TODO: 实现 LLM 改写
        # 示例 Prompt:
        # "请将以下口语化的查询改写为更适合语义检索的形式：
        #  原始查询: {query}
        #  对话上下文: {context}
        #  改写后的查询:"
        
        logger.debug("LLM 改写未实现，返回原始查询")
        return query
    
    def has_context_reference(self, query: str) -> bool:
        """
        检查查询是否包含上下文引用
        
        这表明用户可能需要历史记忆
        """
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.CONTEXT_KEYWORDS)


__all__ = [
    "QueryFilters",
    "ProcessedQuery",
    "QueryProcessor",
    "TimeExpressionParser",
    "MemoryTypeDetector",
]
