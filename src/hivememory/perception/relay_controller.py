"""
HiveMemory Token 溢出接力控制器

处理长任务导致的 Token 溢出问题，生成中间态摘要
以维持跨 Block 的上下文连贯性。

参考: PROJECT.md 4.1.3 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
from typing import List, Optional, Any

from hivememory.perception.interfaces import RelayController
from hivememory.perception.models import LogicalBlock, SemanticBuffer

logger = logging.getLogger(__name__)


class TokenOverflowRelayController(RelayController):
    """
    Token 溢出接力控制器

    职责：
        - 检测即将溢出的 Buffer
        - 生成中间态摘要
        - 维护跨 Block 的上下文连贯性

    Examples:
        >>> controller = TokenOverflowRelayController()
        >>> if controller.should_trigger_relay(buffer, new_block):
        ...     summary = controller.generate_summary(buffer.blocks)
        ...     print(f"接力摘要: {summary}")
    """

    def __init__(
        self,
        max_processing_tokens: int = 8192,
        summary_llm: Optional[Any] = None,
        enable_smart_summary: bool = False,
    ):
        """
        初始化接力控制器

        Args:
            max_processing_tokens: 单次处理的最大 Token 数
            summary_llm: 用于生成摘要的 LLM（可选，未实现）
            enable_smart_summary: 是否启用智能摘要（使用 LLM）
        """
        self.max_processing_tokens = max_processing_tokens
        self.summary_llm = summary_llm
        self.enable_smart_summary = enable_smart_summary

        logger.info(
            f"TokenOverflowRelayController 初始化: "
            f"max_tokens={max_processing_tokens}"
        )

    def should_trigger_relay(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> bool:
        """
        检测是否需要接力（Token 溢出）

        Args:
            buffer: 当前语义缓冲区
            new_block: 新的 LogicalBlock

        Returns:
            bool: 是否需要触发接力
        """
        projected_tokens = buffer.total_tokens + new_block.total_tokens

        should_relay = projected_tokens > self.max_processing_tokens

        if should_relay:
            logger.debug(
                f"Token 即将溢出: {projected_tokens} > {self.max_processing_tokens}"
            )

        return should_relay

    def generate_summary(self, blocks: List[LogicalBlock]) -> str:
        """
        生成中间态摘要

        策略：
            1. 提取所有 User Query
            2. 总结执行链的关键结果
            3. 生成简洁的状态描述

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 生成的摘要文本

        Examples:
            >>> blocks = [block1, block2]
            >>> summary = controller.generate_summary(blocks)
            >>> print(summary)  # "处理了 2 个用户请求；使用了工具: search, code_exec"
        """
        if not blocks:
            return ""

        if self.enable_smart_summary and self.summary_llm:
            return self._generate_llm_summary(blocks)

        return self._generate_simple_summary(blocks)

    def _generate_simple_summary(self, blocks: List[LogicalBlock]) -> str:
        """
        生成简单摘要（基于规则）

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 摘要文本
        """
        summary_parts = []

        # 1. 统计用户请求
        user_queries = [b.anchor_text for b in blocks if b.anchor_text]
        if user_queries:
            summary_parts.append(f"处理了 {len(user_queries)} 个用户请求")

        # 2. 提取使用的工具
        tool_names = set()
        for b in blocks:
            for t in b.execution_chain:
                if t.tool_name:
                    tool_names.add(t.tool_name)

        if tool_names:
            tools_str = ", ".join(sorted(tool_names))
            summary_parts.append(f"使用了工具: {tools_str}")

        # 3. 统计 Token
        total_tokens = sum(b.total_tokens for b in blocks)
        summary_parts.append(f"共 {total_tokens} tokens")

        # 4. 最后一个用户查询（作为上下文参考）
        if user_queries:
            last_query = user_queries[-1][:50]
            if len(user_queries[-1]) > 50:
                last_query += "..."
            summary_parts.append(f"最近: {last_query}")

        return "；".join(summary_parts)

    def _generate_llm_summary(self, blocks: List[LogicalBlock]) -> str:
        """
        使用 LLM 生成更智能的摘要（预留接口）

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 摘要文本
        """
        # TODO: 实现 LLM 调用
        # 目前回退到简单摘要
        logger.warning("LLM 摘要功能尚未实现，使用简单摘要")
        return self._generate_simple_summary(blocks)

    def create_relay_context(self, summary: str) -> str:
        """
        创建接力上下文文本

        将摘要转换为可注入下一个 Buffer 的上下文格式。

        Args:
            summary: 摘要文本

        Returns:
            str: 上下文文本

        Examples:
            >>> summary = "处理了 2 个用户请求"
            >>> context = controller.create_relay_context(summary)
            >>> print(context)  # "[接力摘要] 处理了 2 个用户请求..."
        """
        if not summary:
            return ""

        return f"[接力摘要] {summary}"

    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的 Token 数量

        Args:
            text: 输入文本

        Returns:
            int: 估算的 Token 数量
        """
        # 粗略估算：1 token ≈ 4 字符（英文）或 2 字符（中文）
        # 这里使用简单估算：4 字符 ≈ 1 token
        return max(1, len(text) // 4)

    def format_block_summary(self, block: LogicalBlock) -> str:
        """
        格式化单个 Block 的摘要

        Args:
            block: LogicalBlock

        Returns:
            str: 格式化的摘要
        """
        parts = []

        if block.user_block:
            user_content = block.user_block.content[:30]
            if len(block.user_block.content) > 30:
                user_content += "..."
            parts.append(f"用户: {user_content}")

        if block.execution_chain:
            tools = [t.tool_name for t in block.execution_chain if t.tool_name]
            if tools:
                parts.append(f"工具: {', '.join(set(tools))}")

        if block.response_block:
            response_preview = block.response_block.content[:30]
            if len(block.response_block.content) > 30:
                response_preview += "..."
            parts.append(f"响应: {response_preview}")

        return " | ".join(parts)

    def get_buffer_status(self, buffer: SemanticBuffer) -> dict:
        """
        获取缓冲区状态信息

        Args:
            buffer: 语义缓冲区

        Returns:
            dict: 状态信息字典
        """
        return {
            "buffer_id": buffer.buffer_id,
            "block_count": len(buffer.blocks),
            "total_tokens": buffer.total_tokens,
            "max_tokens": buffer.max_tokens,
            "utilization": buffer.total_tokens / buffer.max_tokens,
            "has_current_block": buffer.current_block is not None,
            "has_relay_summary": buffer.relay_summary is not None,
        }


__all__ = [
    "TokenOverflowRelayController",
]
