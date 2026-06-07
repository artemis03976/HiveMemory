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
import json
from typing import Dict, Any, Optional

from hivememory.system.config import ExtractorConfig
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.engines.generation.interfaces import BaseMemoryExtractor
from hivememory.engines.generation.models import ExtractedMemoryDraft, MergeResult
from hivememory.i18n import get_generation_prompt_text, resolve_language
from hivememory.utils.json_parser import parse_llm_json

logger = logging.getLogger(__name__)

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
        config: ExtractorConfig,
        llm_service: BaseLLMService = None,
    ):
        """
        初始化 LLM 提取器

        Args:
            config: 提取器配置
            llm_service: LLM 服务实例（依赖注入）
        """

        self.llm_service = llm_service
        self.language = resolve_language()
        self.normal_system_prompt = get_generation_prompt_text(
            "passive", "system_prompt", self.language,
        )
        self.normal_user_prompt = get_generation_prompt_text(
            "passive", "user_prompt", self.language,
        )
        self.write_system_prompt = get_generation_prompt_text(
            "write", "system_prompt", self.language,
        )
        self.write_user_prompt = get_generation_prompt_text(
            "write", "user_prompt", self.language,
        )
        self.write_reason_empty = get_generation_prompt_text(
            "write", "reason_empty", self.language,
        )
        self.update_system_prompt = get_generation_prompt_text(
            "update", "system_prompt", self.language,
        )
        self.update_user_prompt = get_generation_prompt_text(
            "update", "user_prompt", self.language,
        )
        self.update_new_content_empty = get_generation_prompt_text(
            "update", "new_content_empty", self.language,
        )
        self.update_transcript_empty = get_generation_prompt_text(
            "update", "transcript_empty", self.language,
        )

        self.format_instructions = self._build_format_instructions()

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
            # 检测模式: Mode B (WRITE) vs Mode A (默认)
            is_write_mode = metadata.get("mode") == "write"

            # Step 1: 构建 Prompt
            if is_write_mode:
                prompt_messages = [
                    {
                        "role": "system",
                        "content": self.write_system_prompt.format(
                            format_instructions=self.format_instructions
                        ),
                    },
                    {
                        "role": "user",
                        "content": self.write_user_prompt.format(
                            write_content=metadata.get("write_content", ""),
                            write_reason=metadata.get("write_reason") or self.write_reason_empty,
                            transcript=transcript,
                        ),
                    },
                ]
                logger.info("Mode B (主动响应): 使用 WRITE 专用提示词")
            else:
                prompt_messages = [
                    {
                        "role": "system",
                        "content": self.normal_system_prompt.format(
                            format_instructions=self.format_instructions
                        ),
                    },
                    {
                        "role": "user",
                        "content": self.normal_user_prompt.format(transcript=transcript),
                    },
                ]

            messages = prompt_messages

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
                # Mode B 强制入库: has_value=True, confidence_score=1.0
                if is_write_mode:
                    draft.has_value = True
                    draft.confidence_score = 1.0
                logger.info(f"成功提取记忆草稿: '{draft.title}' (has_value={draft.has_value})")
            else:
                logger.warning("JSON 解析失败")

            return draft

        except Exception as e:
            logger.error(f"记忆提取失败: {e}", exc_info=True)
            return None

    def _build_format_instructions(self) -> str:
        schema = ExtractedMemoryDraft.model_json_schema()
        return json.dumps(schema, ensure_ascii=False, indent=2)

    def merge(
        self,
        old_content: str,
        metadata: Dict[str, Any],
    ) -> Optional[MergeResult]:
        """
        Mode C: 执行 LLM 驱动的记忆合并 (UPDATE 指令)

        Args:
            old_content: 目标记忆的当前内容
            metadata: 元信息，包含:
                - instruction: 修改指令
                - new_content: 新素材 (可能为空)
                - memory_title: 目标记忆标题
                - memory_alias: 目标记忆别名
                - transcript: 近期对话上下文

        Returns:
            MergeResult: 合并结果，失败时返回 None
        """
        try:
            # Step 1: 构建 Merge Prompt
            new_content = metadata.get("new_content", "")
            prompt_messages = [
                {
                    "role": "system",
                    "content": self.update_system_prompt,
                },
                {
                    "role": "user",
                    "content": self.update_user_prompt.format(
                        old_payload=old_content,
                        instruction=metadata.get("instruction", ""),
                        new_content=new_content or self.update_new_content_empty,
                        transcript=metadata.get("transcript") or self.update_transcript_empty,
                        memory_title=metadata.get("memory_title", ""),
                        memory_alias=metadata.get("memory_alias", ""),
                    ),
                },
            ]
            logger.info("Mode C (合并更新): 使用 UPDATE 专用提示词")

            messages = prompt_messages

            # Step 2: 调用 LLM
            raw_output = self.llm_service.complete_with_retry(
                messages=messages,
            )

            if not raw_output:
                logger.error("LLM 返回空响应 (Mode C)")
                return None

            # Step 3: 解析 JSON → MergeResult
            result = parse_llm_json(
                raw_output,
                as_model=MergeResult,
                default=None,
            )

            if result:
                logger.info(f"成功合并记忆: changelog='{result.changelog}'")
            else:
                logger.warning("Mode C JSON 解析失败")

            return result

        except Exception as e:
            logger.error(f"记忆合并失败: {e}", exc_info=True)
            return None


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
        >>> from hivememory.system.config import ExtractorConfig
        >>> config = ExtractorConfig(enabled=False)
        >>> extractor = create_extractor(config)
    """
    if not config.enabled:
        logger.warning("MemoryExtractor 已禁用 (No-Op)")
        return NoOpMemoryExtractor()

    return LLMMemoryExtractor(
        llm_service=llm_service,
        config=config,
    )


__all__ = [
    "LLMMemoryExtractor",
    "NoOpMemoryExtractor",
    "create_extractor",
]
