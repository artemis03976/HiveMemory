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

from hivememory.engines.gateway.models import (
    GatewayIntent,
    GatewayResult,
)
from hivememory.engines.gateway.engine import GatewayEngine

# 导入协议消息
from hivememory.patchouli.protocol.models import Observation, RetrievalRequest

logger = logging.getLogger(__name__)

class TheEye:
    def __init__(
        self,
        engine: GatewayEngine,
    ):
        """
        初始化真理之眼

        Args:
            engine: Gateway 引擎实例
        """
        self._engine = engine

        logger.info(f"TheEye (真理之眼) 初始化完成")

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
            result = self._engine.process(query, context)

            # 业务逻辑：添加元信息
            result.processing_time_ms = (time.time() - start_time) * 1000
            result.model_used = self._engine.llm_service.model

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
