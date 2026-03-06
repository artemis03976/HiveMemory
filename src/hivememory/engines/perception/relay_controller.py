"""
HiveMemory Token 溢出接力控制器 / Page Folding 摘要生成器

无状态服务，职责：
    - 检测即将溢出的 Buffer (should_relay)
    - 为 Page Folding 生成 state_summary (generate_summary)
    - 生成中间态摘要以维持跨 Block 的上下文连贯性

参考: ShortTermMemory.md §4.2, PROJECT.md 4.1.3 节

作者: HiveMemory Team
版本: 4.6.0
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Any, TYPE_CHECKING
from hivememory.engines.perception.models import FlushEvent, LogicalBlock, SemanticBuffer, FlushReason

if TYPE_CHECKING:
    from hivememory.patchouli.config import SimpleRelayConfig, LLMRelayConfig

logger = logging.getLogger(__name__)


class BaseRelayController(ABC):
    """
    Token 溢出接力控制器基类 (v4.6)

    无状态服务，职责：
        - 检测即将溢出的 Buffer (should_relay)
        - 为 Page Folding 生成 state_summary (generate_summary)
        - 返回统一的 FlushEvent

    子类需实现 generate_summary 方法以提供不同的摘要生成策略。
    """

    def __init__(self, max_processing_tokens: int = 8192):
        """
        初始化接力控制器基类

        Args:
            max_processing_tokens: 单次处理的最大 Token 数
        """
        self.max_processing_tokens = max_processing_tokens

        logger.info(
            f"{self.__class__.__name__} 初始化: "
            f"max_tokens={max_processing_tokens}"
        )

    def should_relay(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> Optional[FlushEvent]:
        """
        检测是否需要接力（Token 溢出）

        Args:
            buffer: 当前语义缓冲区（只读）
            new_block: 新的 LogicalBlock

        Returns:
            None: 不需要接力
            FlushEvent: 需要接力，包含 flush 原因、blocks 和 relay_summary
        """
        projected_tokens = buffer.total_tokens + new_block.total_tokens

        if projected_tokens <= self.max_processing_tokens:
            return None

        logger.debug(
            f"Token 即将溢出: {projected_tokens} > {self.max_processing_tokens}"
        )

        # 生成接力摘要（不包含 previous_summary，因为这是 Token Overflow 场景）
        summary = self.generate_summary(buffer.blocks)

        return FlushEvent(
            flush_reason=FlushReason.TOKEN_OVERFLOW,
            blocks_to_flush=buffer.blocks.copy(),
            relay_summary=summary,
            triggered_by_block=new_block,
        )

    @abstractmethod
    def generate_summary(
        self,
        blocks_to_fold: List[LogicalBlock],
        previous_summary: Optional[str] = None
    ) -> str:
        """
        生成摘要（抽象方法）

        Args:
            blocks_to_fold: 需要折叠的 LogicalBlock 列表
            previous_summary: 之前的 state_summary（如果有）

        Returns:
            str: 生成的摘要文本（已合并 previous_summary）

        Examples:
            >>> blocks = [block1, block2]
            >>> summary = controller.generate_summary(blocks, previous_summary="旧摘要")
            >>> print(summary)  # "旧摘要\n---\n处理了 2 个用户请求；使用了工具: search"
        """
        pass

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


class SimpleRelayController(BaseRelayController):
    """
    简单接力控制器（基于规则的摘要生成）

    使用统计信息生成摘要：
        - 用户请求数量
        - 使用的工具列表
        - Token 总数
        - 最近的查询片段
    """

    def generate_summary(
        self,
        blocks_to_fold: List[LogicalBlock],
        previous_summary: Optional[str] = None
    ) -> str:
        """
        生成简单摘要并合并之前的摘要

        Args:
            blocks_to_fold: 需要折叠的 LogicalBlock 列表
            previous_summary: 之前的 state_summary（如果有）

        Returns:
            str: 合并后的摘要文本
        """
        if not blocks_to_fold:
            return previous_summary or ""

        # 生成新摘要
        new_summary = self._generate_simple_summary(blocks_to_fold)

        # 合并之前的摘要
        if previous_summary:
            return f"{previous_summary}\n---\n{new_summary}"
        else:
            return new_summary

    def _generate_simple_summary(self, blocks: List[LogicalBlock]) -> str:
        """
        生成简单摘要（基于规则）

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 摘要文本
        """
        summary_parts = []

        # 1. 统计用户请求（兼容 v3.0 和 legacy 模式）
        user_queries = []
        for b in blocks:
            # v3.0 模式：优先使用 rewritten_query 或 user_query
            if b.rewritten_query or b.user_query:
                user_queries.append(b.rewritten_query or b.user_query)
            # Legacy 模式：回退到 user_block.content
            elif b.user_block:
                user_queries.append(b.user_block.content)

        if user_queries:
            summary_parts.append(f"处理了 {len(user_queries)} 个用户请求")

        # 2. 提取使用的工具 (兼容 legacy execution_chain + v3.0 semantic_traces)
        tool_names = set()
        for b in blocks:
            # Legacy path: Triplet.tool_name
            for t in b.execution_chain:
                if t.tool_name:
                    tool_names.add(t.tool_name)
            # v3.0 path: TraceItem.tool (RUN) / TraceItem.action (READ/SEARCH)
            for t in b.semantic_traces:
                if t.tool:
                    tool_names.add(t.tool)
                elif t.action in ("READ", "SEARCH"):
                    tool_names.add(t.action.lower())

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


class LLMRelayController(BaseRelayController):
    """
    LLM 接力控制器（基于 LLM 的智能摘要生成）

    使用 LLM 生成更智能、更语义化的摘要。
    """

    def __init__(
        self,
        max_processing_tokens: int = 8192,
        summary_llm: Optional[Any] = None
    ):
        """
        初始化 LLM 接力控制器

        Args:
            max_processing_tokens: 单次处理的最大 Token 数
            summary_llm: 用于生成摘要的 LLM 服务（可选）
        """
        super().__init__(max_processing_tokens)
        self.summary_llm = summary_llm

    def generate_summary(
        self,
        blocks_to_fold: List[LogicalBlock],
        previous_summary: Optional[str] = None
    ) -> str:
        """
        使用 LLM 生成智能摘要并合并之前的摘要

        Args:
            blocks_to_fold: 需要折叠的 LogicalBlock 列表
            previous_summary: 之前的 state_summary（如果有）

        Returns:
            str: 合并后的摘要文本
        """
        if not blocks_to_fold:
            return previous_summary or ""

        # 生成新摘要
        new_summary = self._generate_llm_summary(blocks_to_fold)

        # 合并之前的摘要
        if previous_summary:
            return f"{previous_summary}\n---\n{new_summary}"
        else:
            return new_summary

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
        # 创建临时 SimpleRelayController 实例来生成简单摘要
        simple_controller = SimpleRelayController(self.max_processing_tokens)
        return simple_controller._generate_simple_summary(blocks)


# 向后兼容：保留 RelayController 作为 SimpleRelayController 的别名
RelayController = SimpleRelayController


# ========== 工厂函数 ==========

def create_relay_controller(
    config: "RelayControllerConfig",
    llm_service: Optional[Any] = None
) -> BaseRelayController:
    """
    创建 RelayController 实例（工厂函数）

    根据配置类型自动实例化对应的控制器实现。

    Args:
        config: RelayControllerConfig 配置对象
        llm_service: LLM 服务（可选，用于 LLMRelayController）

    Returns:
        BaseRelayController 实例

    Examples:
        >>> from hivememory.patchouli.config import RelayControllerConfig, SimpleRelayConfig
        >>> config = RelayControllerConfig(engine=SimpleRelayConfig(max_processing_tokens=4096))
        >>> controller = create_relay_controller(config)
        >>> isinstance(controller, SimpleRelayController)
        True
    """
    from hivememory.patchouli.config import SimpleRelayConfig, LLMRelayConfig

    impl_config = config.engine

    if isinstance(impl_config, SimpleRelayConfig):
        return SimpleRelayController(max_processing_tokens=impl_config.max_processing_tokens)

    elif isinstance(impl_config, LLMRelayConfig):
        return LLMRelayController(
            max_processing_tokens=impl_config.max_processing_tokens,
            summary_llm=llm_service
        )

    raise ValueError(f"未知的 RelayController 配置类型: {type(impl_config)}")


__all__ = [
    "BaseRelayController",
    "SimpleRelayController",
    "LLMRelayController",
    "RelayController",  # 向后兼容
    "create_relay_controller",  # 工厂函数
]
