"""
HiveMemory - 记忆提取器 (Memory Extractor)

职责:
    调用 LLM 将自然对话转换为结构化的记忆草稿。

实现策略:
    - 使用 LiteLLM 统一接口
    - Pydantic 输出解析
    - JSON 容错与重试机制
    - 支持自定义 Prompt

作者: HiveMemory Team
版本: 0.1.0
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from hivememory.patchouli.config import ExtractorConfig
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.engines.generation.interfaces import BaseMemoryExtractor
from hivememory.engines.generation.models import ExtractedMemoryDraft
from hivememory.engines.generation.prompts.patchouli import PATCHOULI_SYSTEM_PROMPT, PATCHOULI_USER_PROMPT
from hivememory.utils.json_parser import parse_llm_json

logger = logging.getLogger(__name__)


ROLE_MAPPING = {"human": "user", "ai": "assistant", "system": "system"}

class LLMMemoryExtractor(BaseMemoryExtractor):
    """
    基于 LLM 的记忆提取器

    使用 LiteLLM 调用 DeepSeek/GPT 等模型，将对话转换为结构化记忆草稿。

    特性:
        - 支持自定义 Prompt
        - JSON 解析容错 (支持代码块、纯 JSON)
        - 自动重试机制
        - 详细日志记录

    Examples:
        >>> extractor = LLMMemoryExtractor()
        >>> draft = extractor.extract(
        ...     transcript="User: 如何解析日期?\nAssistant: 使用 datetime...",
        ...     metadata={"user_id": "user123", "session_id": "sess456"}
        ... )
        >>> print(draft.title)
        "Python 日期解析方法"
    """

    def __init__(
        self,
        llm_service: BaseLLMService = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ):
        """
        初始化 LLM 提取器

        Args:
            llm_service: LLM 服务实例（依赖注入）
            system_prompt: 自定义系统提示词
            user_prompt: 自定义用户提示词
        """

        self.llm_service = llm_service
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        # 初始化输出解析器
        self.output_parser = PydanticOutputParser(pydantic_object=ExtractedMemoryDraft)

        # 构建提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt or PATCHOULI_SYSTEM_PROMPT),
            ("user", self.user_prompt or PATCHOULI_USER_PROMPT),
        ])

        model_name = self.llm_service.config.model if self.llm_service and hasattr(self.llm_service, 'config') else "unknown"
        logger.info(f"LLMMemoryExtractor 初始化完成 (模型: {model_name})")

    def extract(
        self,
        transcript: str,
        metadata: Dict[str, Any]
    ) -> Optional[ExtractedMemoryDraft]:
        """
        提取记忆草稿

        工作流程:
            1. 构建 Prompt (注入格式说明和元信息)
            2. 调用 LLM API
            3. 解析 JSON 输出
            4. 验证 Schema

        Args:
            transcript: 格式化的对话文本
            metadata: 元信息字典，包含:
                - session_id: 会话ID
                - user_id: 用户ID
                - agent_id: Agent ID
                - timestamp: 时间戳 (可选)

        Returns:
            ExtractedMemoryDraft: 提取的草稿，失败时返回 None

        Raises:
            ExtractionError: LLM 调用失败或解析失败时抛出

        Examples:
            >>> extractor = LLMMemoryExtractor()
            >>> draft = extractor.extract(
            ...     transcript="User: 帮我写快排\nAssistant: 这是代码...",
            ...     metadata={"user_id": "u1", "session_id": "s1", "agent_id": "a1"}
            ... )
            >>> draft.memory_type
            'CODE_SNIPPET'
        """
        try:
            # Step 1: 构建 Prompt
            prompt_messages = self.prompt_template.format_messages(
                format_instructions=self.output_parser.get_format_instructions(),
                transcript=transcript,
                session_id=metadata.get("session_id", "unknown"),
                user_id=metadata.get("user_id", "unknown"),
                agent_id=metadata.get("agent_id", "unknown"),
                timestamp=metadata.get("timestamp", datetime.now().isoformat()),
            )

            # 转换为 LiteLLM 格式 (处理 LangChain 角色名)
            messages = self._convert_to_litellm_messages(prompt_messages)

            # Step 2: 调用 LLM (带重试)
            raw_output = self.llm_service.complete_with_retry(
                messages=messages,
            )

            if not raw_output:
                logger.error("LLM 返回空响应")
                return None

            # Step 3: 解析 JSON
            draft = parse_llm_json(
                raw_output,
                as_model=ExtractedMemoryDraft,
                default=None
            )

            if draft:
                logger.info(f"成功提取记忆草稿: '{draft.title}' (has_value={draft.has_value})")
            else:
                logger.warning("JSON 解析失败")

            return draft

        except Exception as e:
            logger.error(f"记忆提取失败: {e}", exc_info=True)
            return None

    def _convert_to_litellm_messages(self, langchain_messages) -> list[Dict[str, str]]:
        """
        转换 LangChain 消息格式为 LiteLLM 格式

        LangChain 使用 "human"/"ai"，OpenAI/DeepSeek 使用 "user"/"assistant"

        Args:
            langchain_messages: LangChain 消息列表

        Returns:
            list[Dict]: LiteLLM 格式的消息列表
        """
        return [
            {"role": ROLE_MAPPING.get(msg.type, msg.type), "content": msg.content}
            for msg in langchain_messages
        ]


class NoOpMemoryExtractor(BaseMemoryExtractor):
    """
    No-Op 记忆提取器

    不执行任何提取操作，总是返回 None。
    用于在配置未启用提取器时作为默认实现。
    """

    def extract(
        self,
        transcript: str,
        metadata: Dict[str, Any]
    ) -> Optional[ExtractedMemoryDraft]:
        """
        提取记忆草稿 (No-Op)

        Args:
            transcript: 格式化的对话文本
            metadata: 元信息

        Returns:
            None
        """
        return None


# 便捷函数
def create_extractor(
    config: ExtractorConfig,
    llm_service: BaseLLMService,
) -> BaseMemoryExtractor:
    """
    创建记忆提取器（支持配置）

    Args:
        config: 提取器配置（可选，使用默认配置）
        llm_service: LLM 服务实例（可选，支持依赖注入）

    Returns:
        BaseMemoryExtractor: LLM 提取器实例或 NoOp 实例

    Examples:
        >>> # 使用默认配置
        >>> extractor = create_extractor()
        >>>
        >>> # 使用自定义配置
        >>> from hivememory.patchouli.config import ExtractorConfig
        >>> config = ExtractorConfig(enabled=False)
        >>> extractor = create_extractor(config)
    """
    if not config.enabled:
        logger.info("MemoryExtractor 已禁用 (No-Op)")
        return NoOpMemoryExtractor()

    logger.info("MemoryExtractor 已启用")
    return LLMMemoryExtractor(
        llm_service=llm_service,
        config=config,
    )


__all__ = [
    "LLMMemoryExtractor",
    "NoOpMemoryExtractor",
    "create_extractor",
]
