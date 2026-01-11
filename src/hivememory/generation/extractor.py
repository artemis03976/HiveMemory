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

import json
import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime

import litellm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from hivememory.core.config import LLMConfig, get_librarian_llm_config
from hivememory.generation.interfaces import MemoryExtractor, ExtractionError
from hivememory.generation.models import ExtractedMemoryDraft
from hivememory.agents.prompts.patchouli import PATCHOULI_SYSTEM_PROMPT, PATCHOULI_USER_PROMPT

logger = logging.getLogger(__name__)

# ========== 实现类 ==========

class LLMMemoryExtractor(MemoryExtractor):
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
        llm_config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_retries: int = 2,
    ):
        """
        初始化 LLM 提取器

        Args:
            llm_config: LLM 配置（默认使用全局 Librarian 配置）
            system_prompt: 自定义系统提示词
            user_prompt: 自定义用户提示词
            max_retries: 最大重试次数
        """
        self.llm_config = llm_config or get_librarian_llm_config()
        self.max_retries = max_retries

        # 初始化输出解析器
        self.output_parser = PydanticOutputParser(pydantic_object=ExtractedMemoryDraft)

        # 构建提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt or PATCHOULI_SYSTEM_PROMPT),
            ("user", user_prompt or PATCHOULI_USER_PROMPT),
        ])

        logger.info(f"LLMMemoryExtractor 初始化完成 (模型: {self.llm_config.model})")

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
            raw_output = self._call_llm_with_retry(messages)

            if not raw_output:
                logger.error("LLM 返回空响应")
                return None

            # Step 3: 解析 JSON
            draft = self._parse_json_output(raw_output)

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
        messages = []
        for msg in langchain_messages:
            role = msg.type

            # 映射角色名
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"

            messages.append({"role": role, "content": msg.content})

        return messages

    def _call_llm_with_retry(self, messages: list[Dict[str, str]]) -> Optional[str]:
        """
        调用 LLM API (带重试机制)

        Args:
            messages: LiteLLM 格式的消息列表

        Returns:
            str: LLM 响应内容，失败时返回 None

        Raises:
            ExtractionError: 重试耗尽后仍失败
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"调用 LLM (尝试 {attempt + 1}/{self.max_retries})...")

                response = litellm.completion(
                    model=self.llm_config.model,
                    messages=messages,
                    api_key=self.llm_config.api_key,
                    api_base=self.llm_config.api_base,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                )

                # 提取响应
                raw_output = response.choices[0].message.content
                logger.debug(f"LLM 响应长度: {len(raw_output)} 字符")

                return raw_output

            except Exception as e:
                last_error = e
                logger.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")

                # 最后一次尝试失败时才抛出异常
                if attempt == self.max_retries - 1:
                    raise ExtractionError(f"LLM 调用失败，重试耗尽: {e}") from e

        return None

    def _parse_json_output(self, raw_output: str) -> Optional[ExtractedMemoryDraft]:
        """
        解析 LLM 输出的 JSON (容错版本)

        解析策略:
            1. 尝试直接解析 (纯 JSON)
            2. 提取 ```json 代码块
            3. 正则提取第一个 JSON 对象 {...}

        Args:
            raw_output: LLM 原始输出

        Returns:
            ExtractedMemoryDraft: 解析后的草稿，失败时返回 None

        Examples:
            >>> extractor = LLMMemoryExtractor()
            >>> draft = extractor._parse_json_output('{"title": "测试", ...}')
            >>> draft.title
            "测试"
        """
        try:
            # 策略 1: 直接解析
            data = json.loads(raw_output)
            return ExtractedMemoryDraft(**data)

        except json.JSONDecodeError:
            logger.debug("直接解析失败，尝试提取 JSON 代码块...")

            # 策略 2: 提取 ```json 代码块
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    data = json.loads(json_str)
                    return ExtractedMemoryDraft(**data)
                except Exception as e:
                    logger.warning(f"代码块解析失败: {e}")

            # 策略 3: 正则提取纯 JSON
            json_match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    data = json.loads(json_str)
                    return ExtractedMemoryDraft(**data)
                except Exception as e:
                    logger.warning(f"正则提取解析失败: {e}")

            # 所有策略失败
            logger.error(f"无法解析 JSON 输出: {raw_output[:200]}...")
            return None

        except Exception as e:
            logger.error(f"解析 JSON 失败: {e}")
            return None


# 便捷函数
def create_default_extractor(
    config: Optional["ExtractorConfig"] = None,
) -> MemoryExtractor:
    """
    创建默认提取器（支持配置）

    Args:
        config: 提取器配置（可选，使用默认配置）

    Returns:
        MemoryExtractor: LLM 提取器实例

    Examples:
        >>> # 使用默认配置
        >>> extractor = create_default_extractor()
        >>>
        >>> # 使用自定义配置
        >>> from hivememory.core.config import ExtractorConfig
        >>> config = ExtractorConfig(max_retries=3)
        >>> extractor = create_default_extractor(config)
    """
    if config is None:
        from hivememory.core.config import MemoryConfig
        # 从全局配置获取
        memory_config = MemoryConfig()
        config = memory_config.extraction.extractor

    # 准备 LLM 配置
    llm_config = config.llm_config
    if llm_config is None:
        llm_config = get_librarian_llm_config()

    # 覆盖 temperature 和 max_tokens
    if config.temperature is not None:
        llm_config = llm_config.model_copy(update={"temperature": config.temperature})
    if config.max_tokens is not None:
        llm_config = llm_config.model_copy(update={"max_tokens": config.max_tokens})

    return LLMMemoryExtractor(
        llm_config=llm_config,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        max_retries=config.max_retries,
    )


__all__ = [
    "ExtractedMemoryDraft",
    "LLMMemoryExtractor",
    "create_default_extractor",
]
