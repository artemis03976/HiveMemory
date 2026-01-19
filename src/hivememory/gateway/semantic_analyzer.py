"""
L2 语义分析器实现

提供基于 LLM + Function Calling 的语义分析实现。

作者: HiveMemory Team
版本: 2.0
"""

import json
import logging
from typing import Any, Dict, List, Optional

from hivememory.core.config import GatewayConfig
from hivememory.core.llm import BaseLLMService
from hivememory.gateway.interfaces import SemanticAnalyzer
from hivememory.generation.models import ConversationMessage
from hivememory.gateway.models import (
    ContentPayload,
    GatewayIntent,
    GatewayResult,
    MemorySignal,
)
from hivememory.gateway.prompts import get_system_prompt

logger = logging.getLogger(__name__)


# Function Calling Schema 定义
GATEWAY_FUNCTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_user_query",
        "description": "分析用户查询的意图、重写查询并评估记忆价值",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["RAG", "CHAT", "TOOL", "SYSTEM"],
                    "description": "用户意图分类",
                },
                "rewritten_query": {
                    "type": "string",
                    "description": "指代消解后的完整、独立的查询",
                },
                "search_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "用于稀疏检索的关键词",
                },
                "memory_type": {
                    "type": "string",
                    "enum": [
                        "CODE_SNIPPET",
                        "FACT",
                        "URL_RESOURCE",
                        "REFLECTION",
                        "USER_PROFILE",
                        None,
                    ],
                    "description": "记忆类型过滤（可选）",
                },
                "worth_saving": {
                    "type": "boolean",
                    "description": "是否值得保存为长期记忆",
                },
                "reason": {
                    "type": "string",
                    "description": "判断理由",
                },
            },
            "required": [
                "intent",
                "rewritten_query",
                "search_keywords",
                "worth_saving",
                "reason",
            ],
        },
    },
}


class LLMAnalyzer(SemanticAnalyzer):
    """
    基于 LLM + Function Calling 的语义分析器
    """

    def __init__(
        self,
        llm_service: BaseLLMService,
        config: Optional[GatewayConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化 LLMAnalyzer

        Args:
            llm_service: LLM 服务实例
            config: Gateway 配置（可选，使用默认值）
            system_prompt: 自定义系统提示词（可选）
        """
        self.llm_service = llm_service
        self.config = config or GatewayConfig()
        self.system_prompt = system_prompt or get_system_prompt(
            variant=self.config.prompt_variant,
            language=self.config.prompt_language,
        )

    def analyze(
        self,
        query: str,
        context: List[ConversationMessage],
    ) -> GatewayResult:
        """
        执行语义分析

        使用 Function Calling 调用 LLM，获取结构化输出。

        Args:
            query: 用户原始查询
            context: 对话上下文（用于指代消解）

        Returns:
            GatewayResult: 包含意图、重写查询、关键词、记忆信号的结构化结果
        """
        # 构建消息
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加上下文（最近 N 条）
        if context and self.config.context_window > 0:
            context_str = self._format_context(context[-self.config.context_window :])
            messages.append(
                {
                    "role": "user",
                    "content": f"最近对话:\n{context_str}\n\n当前查询: {query}",
                }
            )
        else:
            messages.append({"role": "user", "content": query})

        # 调用 LLM (使用 Function Calling)
        try:
            response = self.llm_service.complete_with_tools(
                messages=messages,
                tools=[GATEWAY_FUNCTION_SCHEMA],
                tool_choice={
                    "type": "function",
                    "function": {"name": "analyze_user_query"},
                },
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )

            # 解析 Function Call 结果
            return self._parse_function_call_response(response, query)
            
        except Exception as e:
            logger.error(f"LLM 语义分析失败: {e}", exc_info=True)
            # 让上层处理回退逻辑，或者在这里直接返回 fallback
            # 为了保持接口一致性，如果发生未捕获异常，最好能抛出或返回 fallback
            # 原有的 Gateway 实现是在 process 这一层捕获所有异常并 fallback
            # 这里我们也应该尽量健壮，但如果 LLM 调用失败，返回 fallback 也是合理的
            raise e

    def _format_context(self, messages: List[ConversationMessage]) -> str:
        """
        格式化上下文为字符串

        Args:
            messages: 对话消息列表

        Returns:
            str: 格式化后的上下文字符串
        """
        lines = []
        for msg in messages:
            role_name = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role_name}: {msg.content}")
        return "\n".join(lines)

    def _parse_function_call_response(
        self,
        response: Any,
        original_query: str,
    ) -> GatewayResult:
        """
        解析 LLM Function Calling 响应

        Args:
            response: litellm 返回的响应对象
            original_query: 原始查询（用于回退）

        Returns:
            GatewayResult
        """
        try:
            # 检查响应结构
            if not hasattr(response, "choices") or not response.choices:
                logger.warning("Invalid response structure: no choices")
                return GatewayResult.fallback(
                    original_query, reason="Invalid response structure"
                )

            message = response.choices[0].message

            # 检查是否有 tool_calls
            if not hasattr(message, "tool_calls") or not message.tool_calls:
                logger.warning("No tool_calls in response")
                return GatewayResult.fallback(
                    original_query, reason="No tool_calls in response"
                )

            tool_call = message.tool_calls[0]

            # 解析 function arguments
            if not hasattr(tool_call, "function") or not hasattr(
                tool_call.function, "arguments"
            ):
                logger.warning("Invalid tool_call structure")
                return GatewayResult.fallback(
                    original_query, reason="Invalid tool_call structure"
                )

            arguments = tool_call.function.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            # 构建 GatewayResult
            return GatewayResult(
                intent=GatewayIntent(arguments["intent"]),
                content_payload=ContentPayload(
                    rewritten_query=arguments["rewritten_query"],
                    search_keywords=arguments.get("search_keywords", []),
                    target_filters=self._build_filters(arguments.get("memory_type")),
                ),
                memory_signal=MemorySignal(
                    worth_saving=arguments["worth_saving"],
                    reason=arguments["reason"],
                ),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error parsing function call: {e}")
            return GatewayResult.fallback(original_query, reason=f"Parse error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error parsing function call: {e}")
            return GatewayResult.fallback(
                original_query, reason=f"Unexpected error: {str(e)}"
            )

    def _build_filters(self, memory_type: Optional[str]) -> Dict[str, Any]:
        """
        构建过滤条件

        Args:
            memory_type: 记忆类型（可选）

        Returns:
            Dict[str, Any]: 过滤条件字典
        """
        filters = {}
        if memory_type and self.config.enable_memory_type_filter:
            filters["memory_type"] = memory_type
        return filters
