"""
检索路由模块

职责:
    判断用户查询是否需要检索记忆。

实现策略:
    - SimpleRouter: 基于规则的关键词匹配
    - LLMRouter: 基于轻量级 LLM 的智能分类

对应设计文档: PROJECT.md 5.0.2 节
"""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
import re
import logging

from hivememory.generation import ConversationMessage
from hivememory.retrieval.interfaces import RetrievalRouter

logger = logging.getLogger(__name__)


class SimpleRouter(RetrievalRouter):
    """
    基于规则的简单路由器
    
    使用关键词匹配判断是否需要检索历史记忆
    """
    
    # 强烈暗示需要检索的关键词
    RETRIEVAL_KEYWORDS = [
        # 时间引用
        "之前", "刚才", "上次", "昨天", "前面", "earlier", "before",
        "previous", "last time", "上周", "last week",
        
        # 上下文引用
        "那个", "那段", "那个代码", "那个函数", "the code", "the function",
        "项目里", "系统中", "in the project", "in the system",
        
        # 记忆引用
        "记得", "remember", "提到", "mentioned", "说过", "told",
        "定义过", "defined", "设置过", "configured",
        
        # 具体查询 - 用户设置
        "我的", "my ", "我设置", "我配置",
        "api", "key", "密钥", "token",
        "我说过", "I said", "我们讨论", "we discussed",
        "历史", "history", "记录", "record",
        
        # 项目相关
        "代码", "code", "函数", "function", "配置", "config",
        "规则", "rule", "偏好", "preference",
        
        # 查询类问句 - 询问已有信息
        "是什么", "是多少", "有哪些", "哪个", "用的",
        "what is", "what's", "which", "how many",
        
        # 版本和环境
        "版本", "version", "环境", "environment",
        "python", "node", "java",
    ]
    
    # 明确不需要检索的模式（闲聊、问候等）
    NO_RETRIEVAL_PATTERNS = [
        r"^(你好|hi|hello|hey|嗨|哈喽)[\s\!\?\。\？\！]*$",
        r"^(谢谢|thanks|thank you|感谢)[\s\!\?\。\？\！]*$",
        r"^(再见|bye|goodbye|拜拜)[\s\!\?\。\？\！]*$",
        r"^(好的|ok|okay|是的|yes|no|不是)[\s\!\?\。\？\！]*$",
        r"^(你是谁|who are you|你叫什么)[\s\?\？]*$",
        r"^(帮我|help me|请问)?\s*(写|生成|创建)\s*(一个|a|an)?\s*",  # 新建任务
    ]
    
    def __init__(
        self,
        additional_keywords: Optional[List[str]] = None,
        min_query_length: int = 3
    ):
        """
        初始化路由器
        
        Args:
            additional_keywords: 额外的检索触发关键词
            min_query_length: 最小查询长度（过短的查询不触发检索）
        """
        self.keywords = self.RETRIEVAL_KEYWORDS.copy()
        if additional_keywords:
            self.keywords.extend(additional_keywords)
        self.min_query_length = min_query_length
    
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
        """
        # 过滤过短的查询
        if len(query.strip()) < self.min_query_length:
            logger.debug(f"查询过短，跳过检索: '{query}'")
            return False
        
        # 检查是否匹配不需要检索的模式
        query_lower = query.lower().strip()
        for pattern in self.NO_RETRIEVAL_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                logger.debug(f"匹配到不需要检索的模式: '{query}'")
                return False
        
        # 检查是否包含检索关键词
        for keyword in self.keywords:
            if keyword.lower() in query_lower:
                logger.debug(f"命中检索关键词 '{keyword}': '{query}'")
                return True
        
        # 检查上下文中是否有引用（可选）
        if context and len(context) > 0:
            # 如果查询很短但有上下文，可能是后续问题
            if len(query) < 20 and self._is_followup_question(query):
                logger.debug(f"检测到后续问题，触发检索: '{query}'")
                return True
        
        # 默认策略：对于较长的查询，启用检索
        # 这是一个保守的策略，可以调整
        if len(query) > 50:
            logger.debug(f"查询较长，触发检索: '{query[:30]}...'")
            return True
        
        logger.debug(f"未触发检索: '{query}'")
        return False
    
    def _is_followup_question(self, query: str) -> bool:
        """
        检查是否是后续问题
        
        后续问题通常很短且包含指代词
        """
        followup_patterns = [
            r"^(这|那|它|这个|那个|this|that|it)\s*",
            r"^(怎么|如何|怎样|how|what)\s*",
            r"^(为什么|why)\s*",
            r"^(还有|另外|also|and)\s*",
        ]
        
        query_lower = query.lower().strip()
        return any(re.match(p, query_lower) for p in followup_patterns)


class LLMRouter(RetrievalRouter):
    """
    基于 LLM 的智能路由器
    
    使用轻量级 LLM（如 GPT-4o-mini）判断是否需要检索
    """
    
    SYSTEM_PROMPT = """You are a routing classifier for a memory retrieval system.
Your job is to determine if a user query requires searching historical memory.

Answer ONLY with "YES" or "NO".

Say YES if the query:
- References past conversations or information ("what did I say", "the code we wrote")
- Asks about previously defined configurations, preferences, or settings
- Mentions specific past events, dates, or sessions
- Requires context from previous interactions

Say NO if the query:
- Is a simple greeting or farewell
- Asks for new content generation without referencing history
- Is a general knowledge question
- Is self-contained and doesn't need historical context
"""
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        fallback_router: Optional[RetrievalRouter] = None
    ):
        """
        初始化 LLM 路由器
        
        Args:
            llm_config: LLM 配置（model, api_key 等）
            fallback_router: 备用路由器（LLM 失败时使用）
        """
        self.llm_config = llm_config or {}
        self.fallback_router = fallback_router or SimpleRouter()
        self._client = None
    
    def should_retrieve(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None
    ) -> bool:
        """
        使用 LLM 判断是否需要检索
        """
        try:
            import litellm
            
            # 构建消息
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}"}
            ]
            
            # 如果有上下文，添加最近几条
            if context and len(context) > 0:
                context_str = "\n".join([
                    f"{m.role}: {m.content[:100]}"
                    for m in context[-3:]
                ])
                messages[1]["content"] += f"\n\nRecent context:\n{context_str}"
            
            # 调用 LLM
            response = litellm.completion(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                messages=messages,
                api_key=self.llm_config.get("api_key"),
                api_base=self.llm_config.get("api_base"),
                max_tokens=5,
                temperature=0,
            )
            
            answer = response.choices[0].message.content.strip().upper()
            result = answer.startswith("YES")
            
            logger.debug(f"LLM Router 判断: '{query[:30]}...' -> {result}")
            return result
            
        except Exception as e:
            logger.warning(f"LLM Router 失败，使用备用路由器: {e}")
            return self.fallback_router.should_retrieve(query, context)


class AlwaysRetrieveRouter(RetrievalRouter):
    """
    总是触发检索的路由器
    
    用于调试或特定场景
    """
    
    def should_retrieve(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None
    ) -> bool:
        return True


class NeverRetrieveRouter(RetrievalRouter):
    """
    从不触发检索的路由器
    
    用于禁用检索功能
    """
    
    def should_retrieve(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None
    ) -> bool:
        return False


def create_default_router(config: Optional["RouterConfig"] = None) -> RetrievalRouter:
    """
    创建默认路由器

    Args:
        config: 检索路由配置

    Returns:
        RetrievalRouter 实例
    """
    if config is None:
        from hivememory.core.config import RouterConfig
        config = RouterConfig()
    
    if config.router_type == "llm":
        # 如果启用了 LLM 路由
        return LLMRouter(
            llm_config=config.llm_config.model_dump() if config.llm_config else None,
            fallback_router=SimpleRouter(
                additional_keywords=config.additional_keywords,
                min_query_length=config.min_query_length
            )
        )
    else:
        # 默认使用简单路由
        return SimpleRouter(
            additional_keywords=config.additional_keywords,
            min_query_length=config.min_query_length
        )


__all__ = [
    "SimpleRouter",
    "LLMRouter",
    "AlwaysRetrieveRouter",
    "NeverRetrieveRouter",
    "create_default_router",
]
