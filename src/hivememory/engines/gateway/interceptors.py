"""
L1: 规则拦截器 - 快速路径

零开销拦截系统指令和无效文本，可拦截约 20-30% 的无效 LLM 调用。

作者: HiveMemory Team
版本: 2.0
"""

import logging
import re
from typing import List, Optional

from hivememory.engines.gateway.interfaces import BaseInterceptor
from hivememory.engines.gateway.models import GatewayIntent, InterceptorResult

logger = logging.getLogger(__name__)


class RuleInterceptor(BaseInterceptor):
    """
    规则拦截器

    基于正则和字符串匹配的零开销拦截器，用于快速处理：
    1. 系统指令 (/clear, /reset, etc.)
    2. 简单寒暄 (Hi, 你好, 谢谢, etc.)

    拦截策略：
    - 系统指令 -> Intent: SYSTEM
    - 简单寒暄 -> Intent: CHAT (且 worth_saving=False)
    - 其他 -> 不拦截，进入 L2 语义分析

    示例:
        >>> interceptor = RuleInterceptor()
        >>> result = interceptor.intercept("/clear")
        >>> assert result.intent == GatewayIntent.SYSTEM
        >>> result = interceptor.intercept("你好")
        >>> assert result.intent == GatewayIntent.CHAT
        >>> result = interceptor.intercept("如何部署贪吃蛇游戏")
        >>> assert result is None  # 不拦截
    """

    #: 系统指令模式 (正则列表)
    # TODO: 系统指令应从全局获取
    SYSTEM_PATTERNS: List[str] = [
        r"^/clear$",
        r"^/reset$",
        r"^/start$",
        r"^/help$",
        r"^/restart$",
    ]

    #: 无效闲聊模式 (正则列表)
    CHAT_PATTERNS: List[str] = [
        # 中英文问候
        r"^(你好|hi|hello|hey|嗨|哈喽)[\s\!\?。\?\！]*$",
        # 中英文感谢
        r"^(谢谢|thanks|thank you|感谢)[\s\!\?。\?\！]*$",
        # 中英文再见
        r"^(再见|bye|goodbye|拜拜|88)[\s\!\?。\?\！]*$",
        # 简单确认
        r"^(好的|ok|okay|o?k)[\s\!\?。\?\！]*$",
        r"^(是|是的|对|yes|yeah)[\s\!\?。\?\！]*$",
        r"^(不|不是|no|nope)[\s\!\?。\?\！]*$",
        # 极短文本（< 3 字符，可能是误输入或无关紧要）
        r"^.{0,2}$",
    ]

    def __init__(
        self,
        enable_system: bool = True,
        enable_chat: bool = True,
        custom_system_patterns: Optional[List[str]] = None,
        custom_chat_patterns: Optional[List[str]] = None,
    ):
        """
        初始化规则拦截器

        Args:
            enable_system: 是否启用系统指令拦截
            enable_chat: 是否启用闲聊拦截
            custom_system_patterns: 自定义系统指令模式
            custom_chat_patterns: 自定义闲聊模式
        """
        self.enable_system = enable_system
        self.enable_chat = enable_chat

        # 编译正则表达式
        system_patterns = custom_system_patterns or self.SYSTEM_PATTERNS
        chat_patterns = custom_chat_patterns or self.CHAT_PATTERNS

        self._system_regex = [re.compile(p, re.IGNORECASE) for p in system_patterns]
        self._chat_regex = [re.compile(p, re.IGNORECASE) for p in chat_patterns]

        logger.debug(
            f"RuleInterceptor initialized: "
            f"system={len(self._system_regex)} patterns, "
            f"chat={len(self._chat_regex)} patterns"
        )

    def intercept(self, query: str) -> Optional[InterceptorResult]:
        """
        执行拦截

        Args:
            query: 用户查询

        Returns:
            InterceptorResult if intercepted, None otherwise
        """
        query_stripped = query.strip()

        # 跳过空查询
        if not query_stripped:
            return InterceptorResult(
                intent=GatewayIntent.CHAT,
                reason="空查询",
                hit=True,
            )

        # 检查系统指令
        if self.enable_system:
            for pattern in self._system_regex:
                if pattern.match(query_stripped):
                    logger.debug(f"L1 拦截: 系统指令 '{query_stripped}'")
                    return InterceptorResult(
                        intent=GatewayIntent.SYSTEM,
                        reason=f"系统指令: {query_stripped}",
                        hit=True,
                    )

        # 检查无效闲聊
        if self.enable_chat:
            for pattern in self._chat_regex:
                if pattern.match(query_stripped):
                    logger.debug(f"L1 拦截: 简单寒暄 '{query_stripped}'")
                    return InterceptorResult(
                        intent=GatewayIntent.CHAT,
                        reason="简单寒暄",
                        hit=True,
                    )

        # 不拦截，进入 L2 语义分析
        return None

    def add_chat_pattern(self, pattern: str) -> None:
        """
        动态添加闲聊模式

        Args:
            pattern: 正则表达式字符串
        """
        self._chat_regex.append(re.compile(pattern, re.IGNORECASE))
        logger.debug(f"Added chat pattern: {pattern}")

    def add_system_pattern(self, pattern: str) -> None:
        """
        动态添加系统指令模式

        Args:
            pattern: 正则表达式字符串
        """
        self._system_regex.append(re.compile(pattern, re.IGNORECASE))
        logger.debug(f"Added system pattern: {pattern}")


__all__ = [
    "RuleInterceptor",
]
