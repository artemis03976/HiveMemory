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
from typing import List, Optional

from hivememory.core.models import Identity, StreamMessage

from hivememory.engines.gateway.models import (
    GatewayIntent,
    GatewayResult,
)
from hivememory.engines.gateway.engine import GatewayEngine

# 导入协议消息
from hivememory.patchouli.protocol.models import EyeGazeResult

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

        logger.info(f"TheEye 真理之眼初始化完成")

    def gaze(
        self,
        query: str,
        context: Optional[List[StreamMessage]] = None,
        identity: Optional[Identity] = None,
    ) -> EyeGazeResult:
        """
        审视用户查询（真理之眼的主要入口方法）

        Eye 只负责感知和信息重整，不构建下游协议消息。
        返回 EyeGazeResult 供 Kernel 进行数据格式转换。

        Args:
            query: 用户原始查询
            context: 对话上下文（可选），用于指代消解
            identity: 身份标识对象

        Returns:
            EyeGazeResult: Eye 的统一输出模型
        """
        if identity is None:
            identity = Identity()

        start_time = time.time()

        try:
            # 调用数据操作层
            result = self._engine.process(query, context)

            # 业务逻辑：添加元信息
            result.processing_time_ms = (time.time() - start_time) * 1000

            # 业务逻辑：日志记录
            logger.info(
                f"TheEye 处理完成: "
                f"intent={result.intent.value}, "
                f"worth_saving={result.worth_saving}, "
                f"latency={result.processing_time_ms:.1f}ms"
            )

            return EyeGazeResult(
                intent=result.intent,
                rewritten_query=result.rewritten_query,
                search_keywords=result.search_keywords,
                worth_saving=result.worth_saving,
                raw_query=query,
                identity=identity,
                processing_time_ms=result.processing_time_ms,
                is_fallback=False,
            )

        except Exception as e:
            logger.error(f"TheEye 处理失败: {e}", exc_info=True)
            # Fallback 处理
            processing_time_ms = (time.time() - start_time) * 1000

            return EyeGazeResult(
                intent=GatewayIntent.RAG,
                rewritten_query=query,
                search_keywords=[],
                worth_saving=False,
                raw_query=query,
                identity=identity,
                processing_time_ms=processing_time_ms,
                is_fallback=True,
            )


__all__ = [
    "TheEye",
]
