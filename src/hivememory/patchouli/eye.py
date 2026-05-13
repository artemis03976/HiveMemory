"""
帕秋莉·真理之眼 (The Eye of Patchouli)

定位：Agentic Dispatcher（进程调度员）与感知者
职责：
    - 流量入口和意图判断 (Hot)
    - 读取活跃话题列表，执行 Agentic Routing（话题路由）
    - 调用 GatewayEngine 获取原始结果（含 target_topic）
    - 处理 fallback、日志、数据转换
    - 产出 EyeGazeResult 供 Kernel 进行数据格式转换

作者: HiveMemory Team
版本: 4.0 (Phase P1 — observer session 管理迁出至 PassiveObserverIngressor)
"""

import logging
import time
from typing import List, Optional

from hivememory.core.models import Identity

from hivememory.engines.gateway.models import GatewayIntent
from hivememory.engines.gateway.engine import GatewayEngine

from hivememory.patchouli.protocol.models import EyeGazeResult

logger = logging.getLogger(__name__)

class TheEye:
    def __init__(
        self,
        engine: GatewayEngine,
        bus=None,
    ):
        """
        初始化真理之眼 (Agentic Dispatcher)

        Args:
            engine: Gateway 引擎实例
            bus: SystemBus 实例，用于跨服务通信
        """
        self._engine = engine
        self._bus = bus

        logger.info(f"TheEye 真理之眼初始化完成 (Agentic Dispatcher)")

    async def gaze(
        self,
        query: str,
        topic_snapshots: Optional[List] = None,  # List[TopicSnapshot]
        identity: Optional[Identity] = None,
    ) -> EyeGazeResult:
        """
        TheEye 凝视：意图识别、查询重写、话题路由

        Args:
            query: 用户查询字符串
            topic_snapshots: 活跃话题快照列表（用于路由和指代消解）
            identity: 用户身份标识

        Returns:
            EyeGazeResult: Eye 分析结果
        """
        if identity is None:
            identity = Identity()

        start_time = time.time()

        try:
            from hivememory.engines.perception.context_converter import PerceptionContextConverter

            active_topics_menu = None
            if topic_snapshots:
                active_topics_menu = PerceptionContextConverter.snapshots_to_context_text(topic_snapshots)

            result = await self._engine.process(
                query,
                active_topics_menu=active_topics_menu,
            )
            result.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"TheEye 处理完成: "
                f"intent={result.intent.value}, "
                f"target_topic={result.target_topic}, "
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
                target_topic=result.target_topic,
                new_topic_title=result.new_topic_title,
                new_topic_summary=result.new_topic_summary,
            )

        except Exception as e:
            logger.error(f"TheEye 处理失败: {e}", exc_info=True)
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
                target_topic="NEW_TOPIC",
            )


__all__ = [
    "TheEye",
]
