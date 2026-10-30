"""
HiveMemory Token 溢出接力控制器 / Page Folding 摘要生成器

无状态服务，职责：
    - 检测即将溢出的 Buffer (should_relay)
    - 为 Page Folding 生成 state_summary (generate_summary)
    - 生成中间态摘要以维持跨 Block 的上下文连贯性
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from hivememory.core.errors import ModelProviderError
from hivememory.core.models import LogicalBlock
from hivememory.engines.perception.material import render_turn_text
from hivememory.i18n import get_relay_prompt_text

if TYPE_CHECKING:
    from hivememory.config.patchouli import GenerationTranscriptConfig, RelayControllerConfig
    from hivememory.infrastructure.llm.base import BaseLLMService, LLMRequestParams

logger = logging.getLogger(__name__)


class BaseRelayController(ABC):
    """Token 溢出接力控制器基类。

    无状态服务，职责：为 Page Folding 生成 ``state_summary``
    （``generate_summary``）。由 PerceptionFamiliar 在 lease 持有期间调用，
    不感知 Store 与话题生命周期。
    """

    @abstractmethod
    async def generate_summary(
        self,
        blocks_to_fold: list[Any],
        previous_summary: str | None = None,
    ) -> str:
        """生成摘要（抽象方法）"""

    def create_relay_context(self, summary: str) -> str:
        """创建接力上下文文本"""
        return f"[接力摘要] {summary}" if summary else ""


class SimpleRelayController(BaseRelayController):
    """
    简单接力控制器（基于规则的摘要生成）

    使用统计信息生成摘要：
        - 用户请求数量
        - 使用的工具列表：能力调用取动词的小写，外部工具取工具名
        - Token 总数
        - 最近的用户消息片段
    """

    async def generate_summary(
        self, blocks_to_fold: list[LogicalBlock], previous_summary: str | None = None
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

    def _generate_simple_summary(self, blocks: list[LogicalBlock]) -> str:
        """
        生成简单摘要（基于规则）

        Args:
            blocks: LogicalBlock 列表

        Returns:
            str: 摘要文本
        """
        summary_parts = []

        # 1. 统计用户请求
        user_messages = [b.user_message for b in blocks if b.user_message]

        if user_messages:
            summary_parts.append(f"处理了 {len(user_messages)} 个用户请求")

        # 2. 提取使用的工具；无法解析的 MTP 指令没有名字，不计入
        tool_names = set()
        for b in blocks:
            for event in b.turn_events:
                if event.kind != "tool_call":
                    continue
                if event.verb is not None:
                    tool_names.add(event.verb.lower())
                elif event.tool_name:
                    tool_names.add(event.tool_name)

        if tool_names:
            tools_str = ", ".join(sorted(tool_names))
            summary_parts.append(f"使用了工具: {tools_str}")

        # 3. 统计 Token
        total_tokens = sum(b.total_tokens for b in blocks)
        summary_parts.append(f"共 {total_tokens} tokens")

        # 4. 最后一条用户消息（作为上下文参考）
        if user_messages:
            last_message = user_messages[-1][:50]
            if len(user_messages[-1]) > 50:
                last_message += "..."
            summary_parts.append(f"最近: {last_message}")

        return "；".join(summary_parts)


class LLMRelayController(BaseRelayController):
    """
    LLM 接力控制器（基于 LLM 的智能摘要生成）

    使用 LLM 生成更智能、更语义化的摘要；没有可用 LLM、提供商失败或摘要调用
    超时时回退为简单摘要，程序错误与取消不降级。回退的简单摘要接在上一次摘要
    之后，已经折叠的历史不会丢失。超时回退使一次折叠的耗时有上限，并让话题
    回到折叠阈值以下，不会在之后每次写入时再触发一次慢折叠。
    """

    def __init__(
        self,
        summary_llm: "BaseLLMService | None" = None,
        llm_params: "LLMRequestParams | None" = None,
        *,
        summary_timeout_seconds: float,
        transcript_config: "GenerationTranscriptConfig",
    ):
        """
        初始化 LLM 接力控制器

        Args:
            summary_llm: 用于生成摘要的 LLM 服务（可选）
            llm_params: 由 Librarian 组件 LLM 配置转换的请求参数（可选）
            summary_timeout_seconds: 每次摘要调用的时限，超时回退为简单摘要
            transcript_config: 渲染近期事件的截断上限，与记忆生成视图相同
        """
        self.summary_llm = summary_llm
        self._llm_params = llm_params
        self._summary_timeout_seconds = summary_timeout_seconds
        self._transcript_config = transcript_config

    async def generate_summary(
        self, blocks_to_fold: list[LogicalBlock], previous_summary: str | None = None
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

        # 生成新摘要（传入 previous_summary 供 LLM 参考）
        new_summary = await self._generate_llm_summary(blocks_to_fold, previous_summary)

        return new_summary

    def _build_recent_events(self, blocks: list[LogicalBlock]) -> str:
        """构建 recent_events 文本：各块与记忆生成视图相同的材料，块之间空一行。

        被折叠的块在结算前只以摘要存在，近期事件因此保留思考与外部工具过程，
        不只有问答。
        """
        return "\n\n".join(
            render_turn_text(block.turn_events, self._transcript_config) for block in blocks
        )

    async def _generate_llm_summary(
        self, blocks: list[LogicalBlock], previous_summary: str | None = None
    ) -> str:
        """
        使用 LLM 生成智能摘要

        Args:
            blocks: LogicalBlock 列表
            previous_summary: 之前的 state_summary（供 LLM 合并）

        Returns:
            str: 摘要文本
        """
        # 无 LLM 服务时的回退
        if self.summary_llm is None or self._llm_params is None:
            logger.warning("summary_llm 未配置，回退到简单摘要")
            return await SimpleRelayController().generate_summary(blocks, previous_summary)

        # 构建近期事件文本
        recent_events = self._build_recent_events(blocks)

        # 构建用户提示词
        previous_summary_text = previous_summary or get_relay_prompt_text("previous_summary_empty")
        user_prompt = get_relay_prompt_text("user_prompt").format(
            previous_summary=previous_summary_text,
            recent_events=recent_events,
        )

        # 获取系统提示词（默认回退到全局 i18n 兜底文案）
        system_prompt = get_relay_prompt_text("system_prompt")

        # 调用 LLM；只有提供商失败与调用超时回退为简单摘要
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            async with asyncio.timeout(self._summary_timeout_seconds):
                completion = await self.summary_llm.complete(messages, self._llm_params)
        except (TimeoutError, ModelProviderError) as error:
            # TimeoutError 来自上面的时限：超过 summary_timeout_seconds 未返回。
            logger.error("LLM 摘要生成失败，回退到简单摘要: %r", error)
            return await SimpleRelayController().generate_summary(blocks, previous_summary)

        summary = completion.text
        logger.info(f"LLM 摘要生成成功，长度: {len(summary)} 字符")
        return summary


class NoOpRelayController(BaseRelayController):
    """RelayController 的禁用实现。"""

    async def generate_summary(
        self, blocks_to_fold: list[LogicalBlock], previous_summary: str | None = None
    ) -> str:
        return previous_summary or ""


# ========== 工厂函数 ==========


def create_relay_controller(
    config: "RelayControllerConfig",
    llm_service: "BaseLLMService | None" = None,
    llm_params: "LLMRequestParams | None" = None,
    *,
    transcript_config: "GenerationTranscriptConfig",
) -> BaseRelayController:
    """
    创建 RelayController 实例（工厂函数）

    根据配置类型自动实例化对应的控制器实现。

    Args:
        config: RelayControllerConfig 配置对象
        llm_service: LLM 服务（可选，用于 LLMRelayController）
        llm_params: 摘要调用的请求参数（可选，用于 LLMRelayController）
        transcript_config: 记忆生成视图的截断上限，LLM 接力按同一组上限渲染近期事件

    Returns:
        BaseRelayController 实例

    Examples:
        >>> from hivememory.config.patchouli import (
        ...     GenerationTranscriptConfig,
        ...     RelayControllerConfig,
        ...     SimpleRelayConfig,
        ... )
        >>> config = RelayControllerConfig(engine=SimpleRelayConfig())
        >>> controller = create_relay_controller(
        ...     config, transcript_config=GenerationTranscriptConfig()
        ... )
        >>> isinstance(controller, SimpleRelayController)
        True
    """
    from hivememory.config.patchouli import (
        LLMRelayConfig,
        SimpleRelayConfig,
    )

    if not config.enable:
        return NoOpRelayController()

    impl_config = config.engine

    if isinstance(impl_config, SimpleRelayConfig):
        return SimpleRelayController()

    elif isinstance(impl_config, LLMRelayConfig):
        return LLMRelayController(
            summary_llm=llm_service,
            llm_params=llm_params,
            summary_timeout_seconds=impl_config.summary_timeout_seconds,
            transcript_config=transcript_config,
        )

    raise ValueError(f"未知的 RelayController 配置类型: {type(impl_config)}")


__all__ = [
    "BaseRelayController",
    "NoOpRelayController",
    "SimpleRelayController",
    "LLMRelayController",
    "create_relay_controller",
]
