"""
Global Gateway - 全局智能网关

实现"一次计算，多处复用"的统一入口，采用两级处理机制：
- L1: 规则拦截器（零开销）
- L2: 语义分析核心（LLM + Function Calling）

作者: HiveMemory Team
版本: 2.0
"""

import logging
import time
from typing import List, Optional

from hivememory.core.config import GatewayConfig
from hivememory.core.llm import BaseLLMService
from hivememory.gateway.interfaces import Gateway, Interceptor, SemanticAnalyzer
from hivememory.generation.models import ConversationMessage
from hivememory.gateway.interceptors import RuleInterceptor
from hivememory.gateway.semantic_analyzer import LLMAnalyzer
from hivememory.gateway.models import (
    ContentPayload,
    GatewayIntent,
    GatewayResult,
    InterceptorResult,
    MemorySignal,
)
from hivememory.gateway.prompts import get_system_prompt

logger = logging.getLogger(__name__)


class GlobalGateway(Gateway):
    """
    全局智能网关

    采用两级处理机制：
    1. L1: 规则拦截器 - 零开销拦截系统指令和无效文本
    2. L2: 语义分析核心 - 使用 LLM + Function Calling 进行意图分类、指代消解、元数据提取

    输出遵循 InternalProtocol_v2.0.md 的 GatewayResult 结构。
    下游模块可以复用 Gateway 的输出：
    - Hot Path (检索): 使用 rewritten_query + search_keywords
    - Cold Path (感知): 使用 rewritten_query 作为语义锚点
    - Cold Path (生成): 使用 worth_saving 判断是否入库

    示例:
        >>> from hivememory.gateway import GlobalGateway
        >>> from hivememory.core.llm import get_worker_llm_service
        >>>
        >>> llm_service = get_worker_llm_service()
        >>> gateway = GlobalGateway(llm_service=llm_service)
        >>>
        >>> result = gateway.process("我之前设置的 API Key 是什么？")
        >>> print(f"Intent: {result.intent}")
        >>> print(f"Rewritten: {result.content_payload.rewritten_query}")
    """

    def __init__(
        self,
        llm_service: BaseLLMService,
        config: Optional[GatewayConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化全局网关

        Args:
            llm_service: LLM 服务实例
            config: Gateway 配置（可选，使用默认值）
            system_prompt: 自定义系统提示词（可选）
        """
        self.config = config or GatewayConfig()
        self.llm_service = llm_service
        self.system_prompt = system_prompt or get_system_prompt(
            variant=self.config.prompt_variant,
            language=self.config.prompt_language,
        )

        # L1 规则拦截器
        self.rule_interceptor: Optional[Interceptor] = None
        if self.config.enable_l1_interceptor:
            self.rule_interceptor = RuleInterceptor(
                enable_system=True,
                enable_chat=True,
                custom_system_patterns=self.config.custom_system_patterns,
                custom_chat_patterns=self.config.custom_chat_patterns,
            )

        # L2 语义分析器
        self.semantic_analyzer: Optional[SemanticAnalyzer] = None
        if self.config.enable_l2_semantic:
            self.semantic_analyzer = LLMAnalyzer(
                llm_service=llm_service,
                config=self.config,
                system_prompt=self.system_prompt,
            )

        logger.info(
            f"GlobalGateway initialized: "
            f"llm={llm_service.model}, "
            f"L1_interceptor={self.config.enable_l1_interceptor}, "
            f"L2_semantic={self.config.enable_l2_semantic}, "
            f"context_window={self.config.context_window}"
        )

    def process(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None,
    ) -> GatewayResult:
        """
        处理用户查询

        这是 Gateway 的主要入口方法，执行完整的两级处理流程。

        Args:
            query: 用户原始查询
            context: 对话上下文（可选），用于指代消解

        Returns:
            GatewayResult: 结构化分析结果
        """
        start_time = time.time()

        try:
            # L1: 规则拦截
            if self.rule_interceptor:
                interceptor_result = self.rule_interceptor.intercept(query)
                if interceptor_result is not None and interceptor_result.hit:
                    logger.debug(
                        f"L1 拦截命中: {interceptor_result.intent} - {interceptor_result.reason}"
                    )
                    return self._build_interceptor_result(
                        interceptor_result,
                        query,
                        time.time() - start_time,
                    )

            # L2: 语义分析核心
            if self.semantic_analyzer:
                result = self.semantic_analyzer.analyze(query, context or [])
            else:
                # 如果禁用 L2，返回保守结果
                result = GatewayResult.fallback(
                    query, reason="L2 semantic analysis disabled"
                )

            processing_time = time.time() - start_time
            result.processing_time_ms = processing_time * 1000
            result.model_used = self.llm_service.model

            logger.info(
                f"Gateway 处理完成: "
                f"intent={result.intent.value}, "
                f"worth_saving={result.memory_signal.worth_saving}, "
                f"latency={processing_time * 1000:.1f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Gateway 处理失败: {e}", exc_info=True)
            # 回退到保守策略
            return GatewayResult.fallback(query, reason=f"Processing error: {str(e)}")

    def _build_interceptor_result(
        self,
        interceptor_result: InterceptorResult,
        query: str,
        latency: float,
    ) -> GatewayResult:
        """
        构建拦截器结果

        Args:
            interceptor_result: 拦截器结果
            query: 原始查询
            latency: 处理延迟

        Returns:
            GatewayResult
        """
        # 确定是否值得保存
        # SYSTEM 和简单 CHAT 通常不值得保存
        worth_saving = interceptor_result.intent != GatewayIntent.SYSTEM

        return GatewayResult(
            intent=interceptor_result.intent,
            content_payload=ContentPayload(
                rewritten_query=query,
                search_keywords=[],
                target_filters={},
            ),
            memory_signal=MemorySignal(
                worth_saving=worth_saving,
                reason=f"L1 拦截: {interceptor_result.reason}",
            ),
            processing_time_ms=latency * 1000,
            model_used="L1_Rule_Interceptor",
        )


__all__ = [
    "GlobalGateway",
]
