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

from pydantic import BaseModel

from hivememory.config.patchouli import ExtractorConfig
from hivememory.core.errors import ModelContextWindowExceededError, ModelProviderError
from hivememory.core.models import MemoryAtom
from hivememory.engines.generation.interfaces import BaseMemoryExtractor
from hivememory.engines.generation.models import (
    ExtractedMemoryDraft,
    ExtractionResult,
    MergeResult,
    RevisionResult,
)
from hivememory.i18n import get_generation_prompt_text, resolve_language
from hivememory.infrastructure.llm.base import BaseLLMService, LLMCompletion, LLMRequestParams
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

logger = logging.getLogger(__name__)

# 一次提取、修订或合并对提供商的最多请求次数（含首次）。
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
        模型输出为空或无法解析时没有产出（话题提取为空列表，其余为 ``None``）；
        提供商失败在重试耗尽后抛出 ``ModelProviderError``，上下文超窗不重试；
        话题提取的输出被截断时同样抛出 ``ModelProviderError``；程序错误与
        取消原样传播。

    Examples:
        >>> extractor = LLMMemoryExtractor(config, llm_service, llm_params)
        >>> drafts = await extractor.extract("User: 如何解析日期?\nAssistant: 使用 datetime...")
        >>> print(drafts[0].title)
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
        self.write_title_empty = get_generation_prompt_text(
            "write",
            "title_empty",
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

        # 话题提取输出草稿列表，WRITE 只输出一条草稿，两者的 JSON Schema 不同。
        self.passive_format_instructions = _schema_text(ExtractionResult)
        self.write_format_instructions = _schema_text(ExtractedMemoryDraft)
        self.merge_system_prompt = get_generation_prompt_text(
            "merge", "system_prompt", self.language
        )
        self.merge_user_prompt = get_generation_prompt_text("merge", "user_prompt", self.language)

        logger.info(f"LLMMemoryExtractor 初始化完成 (模型: {llm_params.model})")

    async def _complete(self, messages: list[dict[str, str]]) -> LLMCompletion:
        """调用 LLM 并返回完整结果；只对提供商失败重试，最后一次的失败原样抛出。

        超窗时同一请求必然再次失败，不重试；程序错误与取消不重试、原样传播。
        """
        for attempt in range(1, _MAX_LLM_ATTEMPTS):
            try:
                return await self._llm_service.complete(messages, self._llm_params)
            except ModelContextWindowExceededError:
                raise
            except ModelProviderError as error:
                logger.warning(f"LLM 调用失败 (尝试 {attempt}/{_MAX_LLM_ATTEMPTS}): {error}")
        return await self._llm_service.complete(messages, self._llm_params)

    async def extract(self, transcript: str) -> list[ExtractedMemoryDraft]:
        """
        Mode A：从一段话题材料中提取彼此独立的记忆草稿。

        整段输出按一个对象解析，任一条草稿不符合结构时整段视为没有产出。
        输出达到上限被截断时抛出提供商失败而不是视为没有产出：截断的列表
        无法解析，当作没有产出会让整段话题不留记忆、任务却成功结束。同一
        请求会再次截断，因此不重试。

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或输出被截断
        """
        messages = [
            {
                "role": "system",
                "content": self.normal_system_prompt.format(
                    format_instructions=self.passive_format_instructions
                ),
            },
            {"role": "user", "content": self.normal_user_prompt.format(transcript=transcript)},
        ]
        completion = await self._complete(messages)
        if completion.finish_reason == "length":
            raise ModelProviderError("话题提取的输出达到 max_tokens 上限被截断")
        if not completion.text:
            logger.error("LLM 返回空响应")
            return []
        try:
            result = parse_llm_json(completion.text, as_model=ExtractionResult)
        except JSONParseError:
            logger.warning("JSON 解析失败")
            return []
        logger.info(f"成功提取 {len(result.memories)} 条记忆草稿")
        return result.memories

    async def extract_write(
        self, transcript: str, *, content: str, reason: str | None, title: str | None
    ) -> ExtractedMemoryDraft | None:
        """
        Mode B：以 WRITE 提交的内容为核心，结合背景对话结构化为一条草稿。

        明确的保存意图强制入库：草稿的 has_value 为真、置信度为 1.0。agent 拟定的标题
        随提示词交给模型，由模型按提示词沿用，只在它与内容不符或缺少检索名词时修改。

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或上下文超窗
        """
        messages = [
            {
                "role": "system",
                "content": self.write_system_prompt.format(
                    format_instructions=self.write_format_instructions
                ),
            },
            {
                "role": "user",
                "content": self.write_user_prompt.format(
                    write_title=title or self.write_title_empty,
                    write_content=content,
                    write_reason=reason or self.write_reason_empty,
                    transcript=transcript,
                ),
            },
        ]
        raw_output = (await self._complete(messages)).text
        if not raw_output:
            logger.error("LLM 返回空响应 (Mode B)")
            return None
        try:
            draft = parse_llm_json(raw_output, as_model=ExtractedMemoryDraft)
        except JSONParseError:
            logger.warning("Mode B JSON 解析失败")
            return None
        draft.has_value = True
        draft.confidence_score = 1.0
        logger.info(f"成功提取记忆草稿: '{draft.title}'")
        return draft

    async def revise(
        self,
        old_content: str,
        metadata: dict[str, Any],
    ) -> RevisionResult | None:
        """
        Mode C: 执行 LLM 驱动的记忆修订 (UPDATE 指令)

        Args:
            old_content: 目标记忆的当前内容
            metadata: 元信息，包含:
                - instruction: 修改指令
                - new_content: 新素材 (可能为空)
                - memory_title: 目标记忆标题
                - memory_alias: 目标记忆别名
                - transcript: 近期对话上下文

        Returns:
            RevisionResult: 修订结果；模型输出为空或无法解析时返回 None

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或上下文超窗
        """
        # Step 1: 构建修订提示词
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
        logger.info("Mode C (修订): 使用 UPDATE 专用提示词")

        # Step 2: 调用 LLM（只对提供商失败重试）
        raw_output = (await self._complete(messages)).text

        if not raw_output:
            logger.error("LLM 返回空响应 (Mode C)")
            return None

        # Step 3: 解析 JSON → RevisionResult；只把模型文本无法解析或不符合结构视为没有产出
        try:
            result = parse_llm_json(raw_output, as_model=RevisionResult)
        except JSONParseError:
            logger.warning("Mode C JSON 解析失败")
            return None

        logger.info(f"成功修订记忆: changelog='{result.changelog}'")
        return result

    async def merge(self, retained: MemoryAtom, absorbed: MemoryAtom) -> MergeResult | None:
        """合并两条记忆；只重试提供商失败，无效输出不使用追加或覆盖回退。"""
        # 以内容更新时间标明新旧，不能用访问时间或版本号比较两条独立记忆。
        newer = "M" if retained.meta.updated_at >= absorbed.meta.updated_at else "N"
        messages = [
            {"role": "system", "content": self.merge_system_prompt},
            {
                "role": "user",
                "content": self.merge_user_prompt.format(
                    retained=self._merge_material(retained),
                    absorbed=self._merge_material(absorbed),
                    newer=newer,
                ),
            },
        ]
        raw_output = (await self._complete(messages)).text
        if not raw_output:
            logger.warning("合并模型返回空响应")
            return None
        try:
            return parse_llm_json(raw_output, as_model=MergeResult)
        except JSONParseError:
            logger.warning("合并模型输出无法解析")
            return None

    @staticmethod
    def _merge_material(memory: MemoryAtom) -> str:
        """只向模型提供融合所需的内容字段，正式身份不由模型决定。"""
        return json.dumps(
            {
                "title": memory.index.title,
                "summary": memory.index.summary,
                "tags": memory.index.tags,
                "memory_type": memory.index.memory_type.value,
                "content": memory.payload.content,
            },
            ensure_ascii=False,
        )


class NoOpMemoryExtractor(BaseMemoryExtractor):
    """
    No-Op 记忆提取器

    不执行任何提取操作：话题提取总是返回空列表，其余均没有产出。
    用于提取器未启用或没有可用 LLM 时的默认实现。
    """

    async def extract(self, transcript: str) -> list[ExtractedMemoryDraft]:
        return []


def _schema_text(model: type[BaseModel]) -> str:
    """把输出模型的 JSON Schema 渲染为提示词中的格式说明。"""
    return json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2)


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
