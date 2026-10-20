"""
HiveMemory - 记忆提取器 (Memory Extractor)

职责:
    调用 LLM 将自然对话转换为结构化的记忆草稿。

实现策略:
    - 经 LLM 服务接口异步调用，等待模型期间不阻塞事件循环
    - Pydantic 输出解析
    - JSON 容错；只对提供商失败重试
    - 支持自定义 Prompt
"""

import json
import logging
from typing import Any

from hivememory.config.patchouli import ExtractorConfig
from hivememory.core.errors import ModelContextWindowExceededError, ModelProviderError
from hivememory.engines.generation.interfaces import BaseMemoryExtractor
from hivememory.engines.generation.models import ExtractedMemoryDraft, MergeResult
from hivememory.i18n import get_generation_prompt_text, resolve_language
from hivememory.infrastructure.llm.base import BaseLLMService, LLMRequestParams
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

logger = logging.getLogger(__name__)

# 一次提取或合并对提供商的最多请求次数（含首次）。
_MAX_LLM_ATTEMPTS = 2


class LLMMemoryExtractor(BaseMemoryExtractor):
    """
    基于 LLM 的记忆提取器

    经 LLM 服务接口调用 DeepSeek/GPT 等模型，将对话转换为结构化记忆草稿。

    特性:
        - 支持自定义 Prompt
        - JSON 解析容错 (支持代码块、纯 JSON)
        - 提供商失败重试，耗尽后以提供商失败结束
        - 详细日志记录

    失败语义:
        模型输出为空或无法解析时返回 ``None``，表示没有产出；提供商失败在
        重试耗尽后抛出 ``ModelProviderError``，上下文超窗不重试；程序错误与
        取消原样传播。

    Examples:
        >>> extractor = LLMMemoryExtractor(config, llm_service, llm_params)
        >>> draft = await extractor.extract(
        ...     transcript="User: 如何解析日期?\nAssistant: 使用 datetime...",
        ...     metadata={"mode": "write", "write_content": "如何解析日期"}
        ... )
        >>> print(draft.title)
        "Python 日期解析方法"
    """

    def __init__(
        self,
        config: ExtractorConfig,
        llm_service: BaseLLMService,
        llm_params: LLMRequestParams,
    ):
        """
        初始化 LLM 提取器

        Args:
            config: 提取器配置
            llm_service: LLM 服务实例（依赖注入）
            llm_params: 由 Librarian 组件 LLM 配置转换的请求参数
        """

        self._llm_service = llm_service
        self._llm_params = llm_params
        self.language = resolve_language()
        self.normal_system_prompt = get_generation_prompt_text(
            "passive",
            "system_prompt",
            self.language,
        )
        self.normal_user_prompt = get_generation_prompt_text(
            "passive",
            "user_prompt",
            self.language,
        )
        self.write_system_prompt = get_generation_prompt_text(
            "write",
            "system_prompt",
            self.language,
        )
        self.write_user_prompt = get_generation_prompt_text(
            "write",
            "user_prompt",
            self.language,
        )
        self.write_reason_empty = get_generation_prompt_text(
            "write",
            "reason_empty",
            self.language,
        )
        self.update_system_prompt = get_generation_prompt_text(
            "update",
            "system_prompt",
            self.language,
        )
        self.update_user_prompt = get_generation_prompt_text(
            "update",
            "user_prompt",
            self.language,
        )
        self.update_new_content_empty = get_generation_prompt_text(
            "update",
            "new_content_empty",
            self.language,
        )
        self.update_transcript_empty = get_generation_prompt_text(
            "update",
            "transcript_empty",
            self.language,
        )

        self.format_instructions = self._build_format_instructions()

        logger.info(f"LLMMemoryExtractor 初始化完成 (模型: {llm_params.model})")

    async def _complete(self, messages: list[dict[str, str]]) -> str:
        """调用 LLM 并返回文本；只对提供商失败重试，最后一次的失败原样抛出。

        超窗时同一请求必然再次失败，不重试；程序错误与取消不重试、原样传播。
        """
        for attempt in range(1, _MAX_LLM_ATTEMPTS):
            try:
                completion = await self._llm_service.complete(messages, self._llm_params)
                return completion.text
            except ModelContextWindowExceededError:
                raise
            except ModelProviderError as error:
                logger.warning(f"LLM 调用失败 (尝试 {attempt}/{_MAX_LLM_ATTEMPTS}): {error}")
        completion = await self._llm_service.complete(messages, self._llm_params)
        return completion.text

    async def extract(
        self, transcript: str, metadata: dict[str, Any]
    ) -> ExtractedMemoryDraft | None:
        """
        提取记忆草稿

        工作流程:
            1. 构建 Prompt (注入格式说明和元信息)
            2. 调用 LLM API
            3. 解析 JSON 输出
            4. 验证 Schema

        Args:
            transcript: 格式化的对话文本
            metadata: 模式相关的生成元信息（身份归属一律由 WorkspaceIdentity
                承载，不在此传递），实际调用包含:
                - mode: 生成模式（"write" / "update"）
                - write_content / write_reason: Mode B WRITE 指令内容与理由
                - instruction / new_content / memory_title 等: Mode C merge 上下文

        Returns:
            ExtractedMemoryDraft: 提取的草稿；模型输出为空或无法解析时返回 None

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或上下文超窗

        Examples:
            >>> draft = await extractor.extract(
            ...     transcript="User: 帮我写快排\nAssistant: 这是代码...",
            ...     metadata={"mode": "write", "write_content": "帮我写快排",
            ...               "write_reason": "用户明确要求"}
            ... )
            >>> draft.memory_type
            'CODE_SNIPPET'
        """
        # 检测模式: Mode B (WRITE) vs Mode A (默认)
        is_write_mode = metadata.get("mode") == "write"

        # Step 1: 构建 Prompt
        if is_write_mode:
            messages = [
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
            messages = [
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

        # Step 2: 调用 LLM（只对提供商失败重试）
        raw_output = await self._complete(messages)

        if not raw_output:
            logger.error("LLM 返回空响应")
            return None

        # Step 3: 解析 JSON；只把模型文本无法解析或不符合草稿结构视为没有产出
        try:
            draft = parse_llm_json(raw_output, as_model=ExtractedMemoryDraft)
        except JSONParseError:
            logger.warning("JSON 解析失败")
            return None

        # Mode B 强制入库: has_value=True, confidence_score=1.0
        if is_write_mode:
            draft.has_value = True
            draft.confidence_score = 1.0
        logger.info(f"成功提取记忆草稿: '{draft.title}' (has_value={draft.has_value})")
        return draft

    def _build_format_instructions(self) -> str:
        schema = ExtractedMemoryDraft.model_json_schema()
        return json.dumps(schema, ensure_ascii=False, indent=2)

    async def merge(
        self,
        old_content: str,
        metadata: dict[str, Any],
    ) -> MergeResult | None:
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
            MergeResult: 合并结果；模型输出为空或无法解析时返回 None

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或上下文超窗
        """
        # Step 1: 构建 Merge Prompt
        new_content = metadata.get("new_content", "")
        messages = [
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

        # Step 2: 调用 LLM（只对提供商失败重试）
        raw_output = await self._complete(messages)

        if not raw_output:
            logger.error("LLM 返回空响应 (Mode C)")
            return None

        # Step 3: 解析 JSON → MergeResult；只把模型文本无法解析或不符合结构视为没有产出
        try:
            result = parse_llm_json(raw_output, as_model=MergeResult)
        except JSONParseError:
            logger.warning("Mode C JSON 解析失败")
            return None

        logger.info(f"成功合并记忆: changelog='{result.changelog}'")
        return result


class NoOpMemoryExtractor(BaseMemoryExtractor):
    """
    No-Op 记忆提取器

    不执行任何提取操作，总是返回 None。
    用于提取器未启用或没有可用 LLM 时的默认实现。
    """

    async def extract(
        self, transcript: str, metadata: dict[str, Any]
    ) -> ExtractedMemoryDraft | None:
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
    llm_service: BaseLLMService | None,
    llm_params: LLMRequestParams | None,
) -> BaseMemoryExtractor:
    """
    创建记忆提取器（支持配置）

    Args:
        config: 提取器配置
        llm_service: LLM 服务实例（依赖注入）
        llm_params: 由 Librarian 组件 LLM 配置转换的请求参数；未配置模型时为 None

    Returns:
        BaseMemoryExtractor: 启用且 LLM 齐备时为 LLM 提取器，否则为 NoOp 实例

    Examples:
        >>> from hivememory.config.patchouli import ExtractorConfig
        >>> extractor = create_extractor(ExtractorConfig(enabled=False), None, None)
    """
    if not config.enabled:
        logger.warning("MemoryExtractor 已禁用 (No-Op)")
        return NoOpMemoryExtractor()
    if llm_service is None or llm_params is None:
        logger.warning("MemoryExtractor 未配置 LLM (No-Op)")
        return NoOpMemoryExtractor()

    return LLMMemoryExtractor(
        config=config,
        llm_service=llm_service,
        llm_params=llm_params,
    )


__all__ = [
    "LLMMemoryExtractor",
    "NoOpMemoryExtractor",
    "create_extractor",
]
