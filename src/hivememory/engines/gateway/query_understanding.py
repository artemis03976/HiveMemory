"""第一代共享查询分析 Engine 原语。

一次 LLM 调用同时完成意图识别、指代消解与查询重写、检索关键词提取、
记忆价值初判，并在复合意图时给出可选的子意图列表。

技术债说明：该 Engine 当前覆盖了 QueryUnderstanding / IntentClassifier /
MemoryValueJudge / IntentDecomposer 四个候选能力边界，属于第一代
Resolver 的私有实现细节，后续按观测数据决定是否拆分。
"""

from __future__ import annotations

from hivememory.core.models import TopicData
from hivememory.engines.gateway.models import QueryUnderstandingResult
from hivememory.core.protocol.gateway import IntentType, MemoryWriteSignal
from hivememory.i18n import resolve_language
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.prompts.gateway import get_query_understanding_system_prompt
from hivememory.system.config import UserQueryAnalysisConfig
from hivememory.utils.json_parser import parse_llm_json


class QueryUnderstandingError(RuntimeError):
    """Query Understanding 的预期能力或解析失败。"""


class QueryUnderstandingEngine:
    """基于单次 LLM 调用的共享查询分析能力。"""

    def __init__(
        self,
        *,
        config: UserQueryAnalysisConfig,
        llm_service: BaseLLMService | None,
    ) -> None:
        self._config = config
        self._llm_service = llm_service
        self._language = resolve_language().value

    async def analyze(
        self,
        message: str,
        *,
        topic_data: TopicData | None = None,
    ) -> QueryUnderstandingResult:
        """完成一次共享查询分析，不执行检索、写入或话题选择。"""

        if self._llm_service is None:
            raise QueryUnderstandingError("Query Understanding 未配置 LLM 服务")

        prompt = get_query_understanding_system_prompt(
            language=self._language,
            topic_context=self._render_topic_context(topic_data),
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
            return self._validate_result(payload)
        except QueryUnderstandingError:
            raise
        except Exception as exc:
            raise QueryUnderstandingError(f"Query Understanding 调用失败: {exc}") from exc

    def _render_topic_context(self, topic_data: TopicData | None) -> str:
        if topic_data is None:
            return ""
        lines: list[str] = [f"{topic_data.topic_title}"]
        if topic_data.topic_summary:
            lines.append(topic_data.topic_summary)
        if topic_data.state_summary:
            lines.append(topic_data.state_summary)
        text_limit = self._config.context_text_limit
        for block in topic_data.recent_blocks(self._config.context_block_limit):
            user_text = block.user_query[:text_limit]
            assistant_text = block.assistant_final_text[:text_limit]
            if user_text:
                lines.append(f"user: {user_text}")
            if assistant_text:
                lines.append(f"assistant: {assistant_text}")
        return "\n".join(lines)

    @staticmethod
    def _validate_result(payload: object) -> QueryUnderstandingResult:
        if not isinstance(payload, dict):
            raise QueryUnderstandingError("Query Understanding 未返回 JSON object")
        rewritten_query = payload.get("rewritten_query")
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            raise QueryUnderstandingError("Query Understanding 缺少 rewritten_query")

        intent_raw = payload.get("intent_type")
        try:
            intent_type = IntentType(str(intent_raw))
        except ValueError:
            intent_type = IntentType.RAG

        signal_raw = payload.get("memory_write_signal")
        try:
            memory_write_signal = MemoryWriteSignal(str(signal_raw))
        except ValueError:
            memory_write_signal = MemoryWriteSignal.UNKNOWN

        keywords_raw = payload.get("search_keywords")
        search_keywords = (
            tuple(str(kw) for kw in keywords_raw if str(kw).strip())
            if isinstance(keywords_raw, list)
            else ()
        )

        sub_intents_raw = payload.get("sub_intents")
        sub_intents = (
            tuple(str(item) for item in sub_intents_raw if str(item).strip())
            if isinstance(sub_intents_raw, list)
            else ()
        )

        return QueryUnderstandingResult(
            intent_type=intent_type,
            rewritten_query=rewritten_query,
            search_keywords=search_keywords,
            memory_write_signal=memory_write_signal,
            sub_intents=sub_intents,
            reason=str(payload.get("reason") or ""),
        )


__all__ = ["QueryUnderstandingEngine", "QueryUnderstandingError"]
