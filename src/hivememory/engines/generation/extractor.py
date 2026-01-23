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
import ast
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from hivememory.patchouli.config import LLMConfig, get_librarian_llm_config
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.infrastructure.llm.litellm_service import get_librarian_llm_service
from hivememory.engines.generation.interfaces import MemoryExtractor
from hivememory.engines.generation.models import ExtractedMemoryDraft
from hivememory.engines.generation.prompts.patchouli import PATCHOULI_SYSTEM_PROMPT, PATCHOULI_USER_PROMPT

logger = logging.getLogger(__name__)


ROLE_MAPPING = {"human": "user", "ai": "assistant", "system": "system"}

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
        llm_service: Optional[BaseLLMService] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_retries: int = 2,
    ):
        """
        初始化 LLM 提取器

        Args:
            llm_config: LLM 配置（默认使用全局 Librarian 配置）
            llm_service: LLM 服务实例（可选，支持依赖注入）
            system_prompt: 自定义系统提示词
            user_prompt: 自定义用户提示词
            max_retries: 最大重试次数
        """
        self.llm_config = llm_config or get_librarian_llm_config()

        # 支持直接传入 llm_service（便于测试）
        if llm_service is not None:
            self.llm_service = llm_service
        else:
            # 使用工厂函数创建服务实例
            self.llm_service = get_librarian_llm_service(config=self.llm_config)

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
            raw_output = self.llm_service.complete_with_retry(
                messages=messages,
                max_retries=self.max_retries
            )

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
        return [
            {"role": ROLE_MAPPING.get(msg.type, msg.type), "content": msg.content}
            for msg in langchain_messages
        ]

    def _parse_json_output(self, raw_output: str) -> Optional[ExtractedMemoryDraft]:
        """
        解析 LLM 输出的 JSON (增强容错版本)

        解析策略:
            1. 尝试 Markdown 代码块提取 (支持 json/JSON 或无标识)
            2. 尝试直接解析
            3. 尝试智能提取第一个 JSON 对象 (基于括号计数)
            4. 容错处理: 支持单引号、尾部逗号 (通过 ast.literal_eval)

        Args:
            raw_output: LLM 原始输出

        Returns:
            ExtractedMemoryDraft: 解析后的草稿，失败时返回 None
        """
        # 1. 预处理：移除可能的 BOM 头
        raw_output = raw_output.strip()
        if raw_output.startswith('\ufeff'):
            raw_output = raw_output[1:]

        # 定义尝试解析的候选字符串生成器
        def get_candidates(text):
            # 策略 A: Markdown 代码块 (增强版正则)
            # 匹配 ```json, ```JSON, ``` (无语言)
            code_block_pattern = r'```(?:[a-zA-Z]+)?\s*(\{.*?\})\s*```'
            matches = list(re.finditer(code_block_pattern, text, re.DOTALL))
            if matches:
                for match in matches:
                    yield match.group(1)
            
            # 策略 B: 原文 (如果原文本身就是 JSON)
            yield text
            
            # 策略 C: 智能提取第一个 JSON 对象 (基于括号计数)
            start_idx = text.find('{')
            if start_idx != -1:
                brace_count = 0
                in_string = False
                escape = False
                for i, char in enumerate(text[start_idx:], start=start_idx):
                    if char == '"' and not escape:
                        in_string = not in_string
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # 找到一个完整的对象
                                candidate = text[start_idx:i+1]
                                if candidate != text: 
                                    yield candidate
                                break
                    
                    if char == '\\':
                        escape = not escape
                    else:
                        escape = False
        
        # 定义解析函数
        def try_parse(json_str):
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # 尝试修复常见错误
                try:
                    # 尝试 ast.literal_eval 处理 Python 风格字典 (单引号, True/False)
                    return ast.literal_eval(json_str)
                except (ValueError, SyntaxError) as e2:
                    logger.debug(f"JSON 解析策略尝试失败: {str(e)}, {str(e2)}")
                return None

        # 遍历候选并解析
        for json_str in get_candidates(raw_output):
            data = try_parse(json_str)
            if data is not None and isinstance(data, dict):
                try:
                    return ExtractedMemoryDraft(**data)
                except Exception as e:
                    logger.warning(f"Schema 验证失败: {e}")
                    continue

        logger.error(f"无法解析 JSON 输出: {raw_output[:200]}...")
        return None


# 便捷函数
def create_default_extractor(
    config: Optional["ExtractorConfig"] = None,
    llm_service: Optional[BaseLLMService] = None,
) -> MemoryExtractor:
    """
    创建默认提取器（支持配置）

    Args:
        config: 提取器配置（可选，使用默认配置）
        llm_service: LLM 服务实例（可选，支持依赖注入）

    Returns:
        MemoryExtractor: LLM 提取器实例

    Examples:
        >>> # 使用默认配置
        >>> extractor = create_default_extractor()
        >>>
        >>> # 使用自定义配置
        >>> from hivememory.patchouli.config import ExtractorConfig
        >>> config = ExtractorConfig(max_retries=3)
        >>> extractor = create_default_extractor(config)
    """
    if config is None:
        from hivememory.patchouli.config import ExtractorConfig
        config = ExtractorConfig()

    # 准备 LLM 配置
    llm_config = config.llm_config
    if llm_config is None:
        llm_config = get_librarian_llm_config()

    return LLMMemoryExtractor(
        llm_config=llm_config,
        llm_service=llm_service,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        max_retries=config.max_retries,
    )


__all__ = [
    "LLMMemoryExtractor",
    "create_default_extractor",
]
