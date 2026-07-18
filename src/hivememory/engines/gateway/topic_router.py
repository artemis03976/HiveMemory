"""只负责话题选择的 Gateway Engine 原语。"""

from __future__ import annotations

from collections.abc import Sequence

from hivememory.core.models import TopicSnapshot
from hivememory.engines.gateway.models import TopicRoutingResult
from hivememory.i18n import resolve_language
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.prompts.gateway import get_topic_router_system_prompt
from hivememory.system.config import TopicRouterConfig
from hivememory.utils.json_parser import parse_llm_json


class TopicRouterError(RuntimeError):
    """Topic Router 的预期能力或解析失败。"""


class TopicRouterEngine:
    """基于 LLM 的独立话题路由能力。"""

    def __init__(
        self,
        *,
        config: TopicRouterConfig,
        llm_service: BaseLLMService | None,
    ) -> None:
        self._config = config
        self._llm_service = llm_service
        self._language = resolve_language().value

    async def route(
        self,
        message: str,
        *,
        topic_snapshots: Sequence[TopicSnapshot],
    ) -> TopicRoutingResult:
        """选择已有话题或 NEW_TOPIC，不派生其他 Gateway 决策。"""

        if not self._config.enabled:
            return TopicRoutingResult(reason="Topic Router 已禁用")
        if self._llm_service is None:
            raise TopicRouterError("Topic Router 未配置 LLM 服务")

        prompt = get_topic_router_system_prompt(
            language=self._language,
            active_topics_menu=self._render_topics(topic_snapshots),
        )
        kwargs = {}
        if self._config.model_override:
            kwargs["model"] = self._config.model_override
        try:
            content = await self._llm_service.acomplete_json(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message},
                ],
                **kwargs,
            )
            payload = parse_llm_json(content)
            return self._validate_result(payload, topic_snapshots)
        except TopicRouterError:
            raise
        except Exception as exc:
            raise TopicRouterError(f"Topic Router 调用失败: {exc}") from exc

    @staticmethod
    def _render_topics(topic_snapshots: Sequence[TopicSnapshot]) -> str:
        if not topic_snapshots:
            return ""
        lines: list[str] = []
        for snapshot in topic_snapshots:
            lines.append(
                f"- {snapshot.topic_id}: {snapshot.topic_title} | "
                f"{snapshot.topic_summary or snapshot.state_summary}"
            )
        return "\n".join(lines)

    @staticmethod
    def _validate_result(
        payload: object,
        topic_snapshots: Sequence[TopicSnapshot],
    ) -> TopicRoutingResult:
        if not isinstance(payload, dict):
            raise TopicRouterError("Topic Router 未返回 JSON object")
        topic_id = payload.get("target_topic")
        if not isinstance(topic_id, str) or not topic_id:
            raise TopicRouterError("Topic Router 缺少 target_topic")
        allowed_ids = {snapshot.topic_id for snapshot in topic_snapshots}
        if topic_id != "NEW_TOPIC" and topic_id not in allowed_ids:
            raise TopicRouterError("Topic Router 返回了候选列表之外的话题")
        new_topic_title = payload.get("new_topic_title")
        new_topic_summary = payload.get("new_topic_summary")
        if topic_id == "NEW_TOPIC" and (
            not isinstance(new_topic_title, str)
            or not new_topic_title.strip()
            or not isinstance(new_topic_summary, str)
            or not new_topic_summary.strip()
        ):
            raise TopicRouterError("NEW_TOPIC 缺少标题或摘要")
        return TopicRoutingResult(
            topic_id=topic_id,
            new_topic_title=new_topic_title if topic_id == "NEW_TOPIC" else None,
            new_topic_summary=new_topic_summary if topic_id == "NEW_TOPIC" else None,
            reason=str(payload.get("reason") or ""),
        )


__all__ = ["TopicRouterEngine", "TopicRouterError"]
