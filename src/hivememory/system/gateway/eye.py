"""
System-level TheEye Gateway entrypoint.

TheEye owns GatewayEngine invocation, result conversion, latency accounting, and
fallback behavior. It does not own Patchouli topic state or retrieval.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

from hivememory.core.models import Identity, TopicSnapshot
from hivememory.core.protocol.models import EyeGazeResult
from hivememory.engines.gateway.engine import GatewayEngine
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.system.gateway.topic_context import render_topic_snapshots

logger = logging.getLogger(__name__)


class TheEye:
    def __init__(
        self,
        engine: GatewayEngine,
    ):
        """
        Initialize the system Gateway entrypoint.

        Args:
            engine: Gateway engine instance.
        """
        self._engine = engine
        logger.info("TheEye system gateway initialized")

    async def gaze(
        self,
        query: str,
        topic_snapshots: Optional[Sequence[TopicSnapshot]] = None,
        identity: Optional[Identity] = None,
    ) -> EyeGazeResult:
        """Run intent recognition, query rewriting, and topic routing."""
        if identity is None:
            identity = Identity()

        start_time = time.time()

        try:
            active_topics_menu = None
            if topic_snapshots:
                active_topics_menu = render_topic_snapshots(topic_snapshots)

            result = await self._engine.process(
                query,
                active_topics_menu=active_topics_menu,
            )
            result.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "TheEye processed query: "
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
                command=result.command,
            )

        except Exception as e:
            logger.error(f"TheEye processing failed: {e}", exc_info=True)
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


__all__ = ["TheEye"]
