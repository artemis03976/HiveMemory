"""
HiveMemory - 记忆提取器 (Memory Extractor)

职责:
    调用 LLM 将自然对话转换为结构化的记忆草稿，并为 WRITE、修订与合并取得模型的输出。

实现策略:
    - 经 LLM 服务接口异步调用，等待模型期间不阻塞事件循环
    - Pydantic 输出解析
    - JSON 容错；只对提供商失败重试（LLM 接口模块的 ``complete_with_provider_retry``）
    - WRITE、修订与合并只返回模型原文，草稿与编辑集的解析、校验与带错误的重试由生成引擎负责
"""

import json
import logging

from pydantic import BaseModel

from hivememory.config.patchouli import ExtractorConfig
from hivememory.core.errors import ModelProviderError
from hivememory.core.models import MemoryAtom
from hivememory.engines.generation.edits import OutputRetry
from hivememory.engines.generation.interfaces import BaseMemoryExtractor
from hivememory.engines.generation.models import ExtractedMemoryDraft, ExtractionResult
from hivememory.i18n import (
    Language,
    get_generation_prompt_text,
    get_generation_schema_description,
    resolve_language,
)
from hivememory.infrastructure.llm.base import (
    BaseLLMService,
    LLMCompletion,
    LLMRequestParams,
    complete_with_provider_retry,
)
from hivememory.utils.json_parser import JSONParseError, parse_llm_json

logger = logging.getLogger(__name__)

# 提取器用到的提示词：生成模式 → 键。构造时一次读完，提示词工作台在构造期间替换读取函数。
_PROMPT_KEYS = {
    "passive": ("system_prompt", "user_prompt"),
    "write": ("system_prompt", "user_prompt", "reason_empty", "title_empty", "retry_invalid"),
    "update": ("system_prompt", "user_prompt", "new_content_empty", "transcript_empty"),
    "merge": ("system_prompt", "user_prompt"),
    "edit_set": ("retry_invalid",),
}


class LLMMemoryExtractor(BaseMemoryExtractor):
    """
    基于 LLM 的记忆提取器

    经 LLM 服务接口调用 DeepSeek/GPT 等模型，将对话转换为结构化记忆草稿。

    失败语义:
        话题提取的模型输出为空或无法解析时没有产出（空列表）；WRITE、修订与合并只返回模型
        原文，是否合格由调用方判断。提供商失败在重试耗尽后抛出 ``ModelProviderError``，
        上下文超窗不重试；话题提取的输出被截断时同样抛出 ``ModelProviderError``；程序错误
        与取消原样传播。
    """

    def __init__(
        self,
        config: ExtractorConfig,
        llm_service: BaseLLMService,
        llm_params: LLMRequestParams,
    ):
        """
        Args:
            config: 提取器配置
            llm_service: LLM 服务实例（依赖注入）
            llm_params: 由 Librarian 组件 LLM 配置转换的请求参数
        """
        self._llm_service = llm_service
        self._llm_params = llm_params
        self.language = resolve_language()
        self._prompts = {
            (mode, key): get_generation_prompt_text(mode, key, self.language)
            for mode, keys in _PROMPT_KEYS.items()
            for key in keys
        }
        # 话题提取输出草稿列表，WRITE 只输出一条草稿，两者的 JSON Schema 不同。
        self.passive_format_instructions = _schema_text(ExtractionResult, self.language)
        self.write_format_instructions = _schema_text(ExtractedMemoryDraft, self.language)
        logger.info("LLMMemoryExtractor 初始化完成 (模型: %s)", llm_params.model)

    def prompt(self, mode: str, key: str) -> str:
        """构造时读到的一段提示词。"""
        return self._prompts[mode, key]

    async def _complete(self, messages: list[dict[str, str]]) -> LLMCompletion:
        return await complete_with_provider_retry(self._llm_service, messages, self._llm_params)

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
                "content": self.prompt("passive", "system_prompt").format(
                    format_instructions=self.passive_format_instructions
                ),
            },
            {
                "role": "user",
                "content": self.prompt("passive", "user_prompt").format(transcript=transcript),
            },
        ]
        completion = await self._complete(messages)
        if completion.finish_reason == "length":
            raise ModelProviderError("话题提取的输出达到 max_tokens 上限被截断")
        if not completion.text:
            logger.warning("LLM 返回空响应")
            return []
        try:
            result = parse_llm_json(completion.text, as_model=ExtractionResult)
        except JSONParseError:
            logger.warning("JSON 解析失败")
            return []
        logger.info("成功提取 %s 条记忆草稿", len(result.memories))
        return result.memories

    async def extract_write(
        self,
        transcript: str,
        *,
        content: str,
        reason: str | None,
        title: str | None,
        retry: OutputRetry | None = None,
    ) -> str:
        """
        Mode B：请模型以 WRITE 提交的内容为核心、结合背景对话写出一条草稿，返回模型原文。

        agent 拟定的标题随提示词交给模型，由模型按提示词沿用，只在它与内容不符或缺少检索
        名词时修改。

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或上下文超窗
        """
        messages = [
            {
                "role": "system",
                "content": self.prompt("write", "system_prompt").format(
                    format_instructions=self.write_format_instructions
                ),
            },
            {
                "role": "user",
                "content": self.prompt("write", "user_prompt").format(
                    write_title=title or self.prompt("write", "title_empty"),
                    write_content=content,
                    write_reason=reason or self.prompt("write", "reason_empty"),
                    transcript=transcript,
                ),
            },
        ]
        return (await self._complete(self._with_retry(messages, retry, "write"))).text

    async def revise(
        self,
        memory: MemoryAtom,
        *,
        instruction: str,
        new_content: str | None,
        transcript: str,
        retry: OutputRetry | None = None,
    ) -> str:
        """Mode C：请模型按 UPDATE 的指令与新材料写出对目标记忆的编辑集，返回模型原文。

        模型看到目标记忆的标题、别名、摘要、标签与正文。

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或上下文超窗
        """
        messages = [
            {"role": "system", "content": self.prompt("update", "system_prompt")},
            {
                "role": "user",
                "content": self.prompt("update", "user_prompt").format(
                    memory_title=memory.index.title,
                    memory_alias=memory.index.alias or "",
                    memory_summary=memory.index.summary,
                    memory_tags=json.dumps(memory.index.tags, ensure_ascii=False),
                    old_payload=memory.payload.content,
                    instruction=instruction,
                    new_content=new_content or self.prompt("update", "new_content_empty"),
                    transcript=transcript or self.prompt("update", "transcript_empty"),
                ),
            },
        ]
        return (await self._complete(self._with_retry(messages, retry, "edit_set"))).text

    async def merge(
        self, retained: MemoryAtom, absorbed: MemoryAtom, *, retry: OutputRetry | None = None
    ) -> str:
        """合并：请模型写出把被吸收记忆独有的内容补进保留记忆的编辑集，返回模型原文。

        Raises:
            ModelProviderError: 提供商失败且重试耗尽，或上下文超窗
        """
        # 以内容更新时间标明新旧，不能用访问时间或版本号比较两条独立记忆。
        newer = "M" if retained.meta.updated_at >= absorbed.meta.updated_at else "N"
        messages = [
            {"role": "system", "content": self.prompt("merge", "system_prompt")},
            {
                "role": "user",
                "content": self.prompt("merge", "user_prompt").format(
                    retained=self._merge_material(retained),
                    absorbed=self._merge_material(absorbed),
                    newer=newer,
                ),
            },
        ]
        return (await self._complete(self._with_retry(messages, retry, "edit_set"))).text

    def _with_retry(
        self, messages: list[dict[str, str]], retry: OutputRetry | None, mode: str
    ) -> list[dict[str, str]]:
        """重试时把上一次的输出与它被拒的说明接在对话之后，模型据此改正。

        说明的写法按输出的种类取 ``mode`` 下的 ``retry_invalid``：草稿与编辑集各有一段。
        """
        if retry is None:
            return messages
        output, rejection = retry
        reason = self.prompt(mode, "retry_invalid").format(details=rejection)
        return [
            *messages,
            {"role": "assistant", "content": output},
            {"role": "user", "content": reason},
        ]

    @staticmethod
    def _merge_material(memory: MemoryAtom) -> str:
        """只向模型提供合并所需的内容字段，正式身份不由模型决定。"""
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


def _schema_text(model: type[BaseModel], language: Language) -> str:
    """把输出模型的 JSON Schema 渲染为提示词中的格式说明，说明文字与提示词同一语言。

    Schema 的说明文字来自模型的类 docstring 与字段 description，原文是中文；原样嵌进英文
    提示词时，模型会跟着中文说明用中文写记忆。结构仍取自模型，只逐项换掉说明文字。
    """
    schema = model.model_json_schema()
    for definition in (schema, *schema.get("$defs", {}).values()):
        name = definition["title"]
        nodes = {name: definition}
        nodes.update({f"{name}.{field}": node for field, node in definition["properties"].items()})
        for key, node in nodes.items():
            if "description" in node:
                node["description"] = get_generation_schema_description(
                    key, node["description"], language
                )
    return json.dumps(schema, ensure_ascii=False, indent=2)


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
