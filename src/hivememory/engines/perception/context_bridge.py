"""
HiveMemory 上下文桥接器 (Context Bridge)

负责从 SemanticBuffer 中提取上一轮对话的上下文，
并将其与当前查询组合，构建增强的语义锚点。

核心功能：
    1. 从 Buffer 的最后一个 Block 提取上下文
    2. 构建 anchor_text = f"Context: {last_context}\nQuery: {rewritten_query}"

参考: docs/mod/DiscourseContinuity.md

作者: HiveMemory Team
版本: 2.0.0
"""

import logging
from typing import Optional

from hivememory.patchouli.config import ContextBridgeConfig
from hivememory.engines.perception.models import SemanticBuffer

logger = logging.getLogger(__name__)


# 默认的停用词列表（用于检测非信息性短文本）
DEFAULT_STOP_WORDS = {
    # 中文停用词
    "不对", "报错了", "错了", "错误", "嗯", "哦", "啊",
    "好", "行", "可以", "继续", "next", "好的",
    # 英文停用词
    "ok", "okay", "yes", "no", "yeah", "yep", "nope",
    "continue", "go on", "sure", "alright",
    # 符号/表情
    "...", "..", ".", "!", "!", "！", "～", "~",
}


class ContextBridge:
    """
    上下文桥接器

    职责：
        - 从 SemanticBuffer 提取上一轮上下文摘要
        - 将上下文与 rewritten_query 组合成增强的语义锚点

    设计思路：
        为了解决跨域任务流的连续性问题（如"写代码" → "部署服务器"），
        单纯的语义相似度计算不足以捕捉任务流的逻辑关系。
        通过引入上一轮的上下文，可以让模型理解任务之间的因果/递进关系。

    Examples:
        >>> bridge = ContextBridge(context_max_length=200)
        >>> anchor = bridge.build_anchor_text("部署服务器", buffer)
        >>> print(anchor)
        Context: 用户在写贪吃蛇游戏的代码...
        Query: 部署服务器
    """

    def __init__(
        self,
        config: ContextBridgeConfig,
    ):
        """
        初始化上下文桥接器

        Args:
            context_max_length: 上下文最大长度（字符数）
            context_source: 上下文来源
                - "response": 使用助手回复作为上下文（默认）
                - "user": 使用用户查询作为上下文
                - "auto": 自动选择（优先 response，回退 user）
        """
        self.config = config

        self.context_max_length = config.context_max_length
        self.context_source = config.context_source

        logger.debug(
            f"ContextBridge 初始化: max_length={config.context_max_length}, "
            f"source={config.context_source}"
        )

    def extract_last_context(self, buffer: SemanticBuffer) -> str:
        """
        从 Buffer 提取上一轮的上下文

        优先级顺序：
            1. 最后一个 block 的 response_block.content（助手回复）
            2. 最后一个 block 的 user_block.content（用户查询）
            3. Buffer 的 relay_summary（接力摘要，如果存在）
            4. 空字符串（无上下文可用）

        Args:
            buffer: 当前语义缓冲区

        Returns:
            str: 上一轮的上下文摘要（已截断到 max_length）
        """
        # 1. 尝试从最后一块 block 提取上下文
        if buffer.blocks:
            last_block = buffer.blocks[-1]

            # 尝试从助手回复提取
            if self.context_source in ("response", "auto"):
                if last_block.response_block and last_block.response_block.content:
                    context = last_block.response_block.content
                    logger.debug(f"从助手回复提取上下文: {context[:50]}...")
                    return self._truncate_context(context)

            # 尝试从用户查询提取
            if self.context_source in ("user", "auto"):
                if last_block.user_block and last_block.user_block.content:
                    context = last_block.user_block.content
                    logger.debug(f"从用户查询提取上下文: {context[:50]}...")
                    return self._truncate_context(context)

        # 2. 尝试从 relay_summary 提取
        if buffer.relay_summary:
            logger.debug(f"从 relay_summary 提取上下文: {buffer.relay_summary[:50]}...")
            return self._truncate_context(buffer.relay_summary)

        # 3. 无法提取上下文
        logger.debug("无法提取上下文，返回空字符串")
        return ""

    def _truncate_context(self, context: str) -> str:
        """
        截断上下文到最大长度

        Args:
            context: 原始上下文

        Returns:
            str: 截断后的上下文（带省略号）
        """
        if len(context) <= self.context_max_length:
            return context

        # 截断并添加省略号
        truncated = context[: self.context_max_length - 3] + "..."
        logger.debug(f"上下文已截断: {len(context)} -> {len(truncated)}")
        return truncated

    def build_anchor_text(
        self,
        rewritten_query: str,
        buffer: SemanticBuffer,
    ) -> str:
        """
        构建增强的语义锚点文本

        格式：
            Context: {last_context}
            Query: {rewritten_query}

        如果没有上下文，直接返回 query。

        Args:
            rewritten_query: Gateway 重写后的查询（指代消解后的完整查询）
            buffer: 当前语义缓冲区

        Returns:
            str: 增强的语义锚点文本

        Examples:
            >>> bridge = ContextBridge()
            >>> # 假设 buffer 的最后一条回复是"代码已写完"
            >>> anchor = bridge.build_anchor_text("部署服务器", buffer)
            >>> # 返回: "Context: 代码已写完\nQuery: 部署服务器"
        """
        # 如果没有 rewritten_query，使用空字符串（上层会处理）
        if not rewritten_query:
            logger.debug("rewritten_query 为空，返回空字符串")
            return ""

        # 提取上一轮上下文
        last_context = self.extract_last_context(buffer)

        # 如果没有上下文，直接返回 query
        if not last_context:
            logger.debug("无上下文，直接返回 query")
            return rewritten_query

        # 构建增强的锚点文本
        anchor_text = f"Context: {last_context}\nQuery: {rewritten_query}"
        logger.debug(f"构建锚点文本:\n{anchor_text}")

        return anchor_text

    def is_stop_word(self, text: str) -> bool:
        """
        检测文本是否为停用词（非信息性短文本）

        Args:
            text: 待检测的文本

        Returns:
            bool: 是否为停用词

        Examples:
            >>> bridge = ContextBridge()
            >>> bridge.is_stop_word("ok")
            True
            >>> bridge.is_stop_word("帮我写个函数")
            False
        """
        if not text:
            return True

        # 去除首尾空格并转换为小写
        normalized = text.strip().lower()

        # 直接匹配停用词列表
        if normalized in DEFAULT_STOP_WORDS:
            logger.debug(f"检测到停用词: {text}")
            return True

        return False


__all__ = [
    "ContextBridge",
]
