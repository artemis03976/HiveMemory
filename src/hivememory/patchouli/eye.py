"""
帕秋莉·真理之眼 (The Eye of Patchouli)

定位：守门人与感知者
职责：
    - 流量入口和意图判断 (Hot)
    - 调用 GatewayService 获取原始结果
    - 处理 fallback、日志、数据转换
    - 产出 RetrievalRequest 和 Observation 供下游使用

作者: HiveMemory Team
版本: 2.2
"""

import logging
import time
from typing import List, Optional, Tuple

from hivememory.core.models import Identity, StreamMessage
from hivememory.patchouli.config import GatewayConfig
from hivememory.infrastructure.llm import BaseLLMService
from hivememory.engines.gateway.interfaces import BaseInterceptor, BaseSemanticAnalyzer
from hivememory.engines.gateway.interceptors import RuleInterceptor
from hivememory.engines.gateway.semantic_analyzer import LLMAnalyzer
from hivememory.engines.gateway.models import (
    GatewayIntent,
    GatewayResult,
)
from hivememory.engines.gateway.prompts import get_system_prompt
from hivememory.engines.gateway.service import GatewayService

# 导入协议消息
from hivememory.patchouli.protocol.models import Observation, RetrievalRequest

logger = logging.getLogger(__name__)


class TheEye:
    """
    帕秋莉·真理之眼 (The Eye of Patchouli)

    这是帕秋莉释放的一只"魔法之眼"悬浮在门口（交互层最前端），
    负责第一时间审视所有进来的访客（用户消息）。

    特性：
        - 同步阻塞
        - 极低延迟
        - 小模型驱动 (GPT-4o-mini / Local 7B)

    职责：
        1. 调用 GatewayService 进行数据操作
        2. 处理 fallback
        3. 记录日志
        4. 构建协议消息供下游使用

    示例:
        >>> from hivememory.patchouli.eye import TheEye
        >>> from hivememory.infrastructure.llm import get_worker_llm_service
        >>>
        >>> llm_service = get_worker_llm_service()
        >>> eye = TheEye(llm_service=llm_service)
        >>>
        >>> result = eye.gaze("我之前设置的 API Key 是什么？")
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
        初始化真理之眼

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

        # 创建组件
        interceptor: Optional[BaseInterceptor] = None
        if self.config.enable_l1_interceptor:
            interceptor = RuleInterceptor(
                enable_system=True,
                enable_chat=True,
                custom_system_patterns=self.config.custom_system_patterns,
                custom_chat_patterns=self.config.custom_chat_patterns,
            )

        semantic_analyzer: Optional[BaseSemanticAnalyzer] = None
        if self.config.enable_l2_semantic:
            semantic_analyzer = LLMAnalyzer(
                llm_service=llm_service,
                config=self.config,
                system_prompt=self.system_prompt,
            )

        # 创建 GatewayService（数据操作层）
        self._service = GatewayService(
            interceptor=interceptor,
            semantic_analyzer=semantic_analyzer,
        )

        logger.info(
            f"TheEye (真理之眼) initialized: "
            f"llm={llm_service.model}, "
            f"L1_interceptor={self.config.enable_l1_interceptor}, "
            f"L2_semantic={self.config.enable_l2_semantic}, "
        )

    def gaze(
        self,
        query: str,
        context: Optional[List[StreamMessage]] = None,
        identity: Optional[Identity] = None,
    ) -> Tuple[Optional[RetrievalRequest], Observation]:
        """
        审视用户查询（真理之眼的主要入口方法）

        这是 Eye 的核心方法，执行完整的两级处理流程。

        Args:
            query: 用户原始查询
            context: 对话上下文（可选），用于指代消解
            identity: 身份标识对象

        Returns:
            Tuple[Optional[RetrievalRequest], Observation]:
                - RetrievalRequest: 如果需要检索则返回请求对象，否则为 None
                - Observation: 总是返回感知信号对象
        """
        if identity is None:
            identity = Identity()

        start_time = time.time()
        result: GatewayResult

        try:
            # 调用数据操作层
            result = self._service.process(query, context)

            # 业务逻辑：添加元信息
            result.processing_time_ms = (time.time() - start_time) * 1000
            result.model_used = self.llm_service.model

            # 业务逻辑：日志记录
            logger.info(
                f"TheEye 处理完成: "
                f"intent={result.intent.value}, "
                f"worth_saving={result.memory_signal.worth_saving}, "
                f"latency={result.processing_time_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"TheEye 处理失败: {e}", exc_info=True)
            # Fallback 处理
            result = GatewayResult.fallback(query, reason=f"Processing error: {str(e)}")
            result.processing_time_ms = (time.time() - start_time) * 1000

        # 构建协议消息
        observation = self.build_observation(
            gateway_result=result,
            raw_query=query,
            identity=identity,
        )

        retrieval_request = self.build_retrieval_request(
            gateway_result=result,
            identity=identity,
        )

        return retrieval_request, observation

    def build_retrieval_request(
        self,
        gateway_result: GatewayResult,
        identity: Identity,
    ) -> Optional[RetrievalRequest]:
        """
        构建检索请求协议消息

        只有 RAG 意图才返回检索请求。

        Args:
            gateway_result: Gateway 处理结果
            identity: 身份标识对象

        Returns:
            RetrievalRequest 如果 intent 是 RAG，否则返回 None
        """
        if gateway_result.intent != GatewayIntent.RAG:
            return None

        return RetrievalRequest(
            semantic_query=gateway_result.rewritten_query,
            keywords=gateway_result.search_keywords,
            filters=gateway_result.target_filters,
            user_id=identity.user_id,
        )

    def build_observation(
        self,
        gateway_result: GatewayResult,
        raw_query: str,
        identity: Identity,
    ) -> Observation:
        """
        构建感知信号协议消息

        冷路径入口，发送给 LibrarianCore。

        Args:
            gateway_result: Gateway 处理结果
            raw_query: 原始查询
            identity: 身份标识对象

        Returns:
            Observation: 感知信号协议消息
        """
        return Observation(
            anchor=gateway_result.rewritten_query,
            raw_message=raw_query,
            role="user",
            identity=identity,
            gateway_context={
                "intent": gateway_result.intent.value,
                "worth_saving": gateway_result.worth_saving,
                "processing_time_ms": gateway_result.processing_time_ms,
            },
        )


__all__ = [
    "TheEye",
]
