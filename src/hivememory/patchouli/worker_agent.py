"""
Worker Agent Service - 无状态 LLM 文本生成服务

定位：PatchouliSystem 的"引擎"，纯粹的文本生成器。
职责：
    - 封装 LLM API 调用
    - 使用 MTP Stop Sequence (⟫) 实现生成中断
    - 检测 MTP 指令并返回结构化结果
    - 不持有任何状态（无 Session、无 Buffer、无业务逻辑）

架构定位：
    PatchouliSystem 持有 WorkerAgentService，在递归生成循环中调用。
    WorkerAgentService 不知道 MTP 协议的语义，只负责检测 ⟪ 定界符。

    PatchouliSystem
    ├── TheEye (Gateway)
    ├── PatchouliKernel (Orchestrator)
    └── WorkerAgentService (LLM Engine)  ← 本模块

对应设计文档: MemoryToolProtocol.md Section 3.1 & 6.4

作者: HiveMemory Team
版本: 1.0
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import litellm

from hivememory.patchouli.config import LLMConfig
from hivememory.patchouli.protocol.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_STOP_SEQUENCE,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    单次 LLM 生成的结构化结果

    Attributes:
        text: LLM 生成的完整文本
        finish_reason: API 返回的停止原因 ("stop" / "length" / "end_turn" 等)
        was_mtp_interrupted: 是否因 MTP Stop Sequence 中断且文本含 ⟪
        prefix_text: ⟪ 之前的自然语言文本 (无 MTP 时等于 text)
        mtp_fragment: 从 ⟪ 开始的 MTP 指令片段 (无 MTP 时为空)
    """
    text: str = ""
    finish_reason: str = "stop"
    was_mtp_interrupted: bool = False
    prefix_text: str = ""
    mtp_fragment: str = ""


class WorkerAgentService:
    """
    无状态 LLM 文本生成服务

    封装 litellm.completion() 调用，自动注入 MTP Stop Sequence，
    并对返回结果进行 MTP 中断检测。

    使用示例:
        >>> from hivememory.patchouli.worker_agent import WorkerAgentService
        >>> from hivememory.patchouli.config import LLMConfig
        >>>
        >>> service = WorkerAgentService(config=LLMConfig(model="gpt-4o"))
        >>> result = service.generate([{"role": "user", "content": "Hello"}])
        >>> if result.was_mtp_interrupted:
        ...     print(f"MTP detected: {result.mtp_fragment}")
    """

    def __init__(self, config: LLMConfig):
        """
        Args:
            config: LLM 配置 (model, api_key, api_base, temperature, max_tokens)
        """
        self._config = config
        logger.info(
            f"WorkerAgentService 初始化完成 (model={config.model})"
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> GenerationResult:
        """
        单次 LLM 生成，使用 stop=[MTP_STOP_SEQUENCE]

        当 LLM 生成 ⟫ 时 API 会提前返回，此方法检测文本中是否包含 ⟪
        来判断是否为 MTP 指令中断。

        Args:
            messages: OpenAI 格式的消息列表
            **kwargs: 传递给 litellm.completion() 的额外参数

        Returns:
            GenerationResult: 结构化生成结果
        """
        try:
            response = litellm.completion(
                model=self._config.model,
                messages=messages,
                api_key=self._config.api_key,
                api_base=self._config.api_base,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                stop=[MTP_STOP_SEQUENCE],
                **kwargs,
            )
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            raise

        text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"

        # 记录 token 使用
        if hasattr(response, "usage") and response.usage:
            logger.info(
                f"LLM 生成完成 (model={self._config.model}, "
                f"tokens={response.usage.total_tokens}, "
                f"finish_reason={finish_reason})"
            )

        # MTP 中断检测
        last_open = text.rfind(MTP_LEFT_DELIMITER)
        was_mtp = finish_reason == "stop" and last_open != -1

        if was_mtp:
            logger.info(
                f"MTP 中断检测: ⟪ 位于 offset={last_open}, "
                f"prefix_len={last_open}, fragment_len={len(text) - last_open}"
            )

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            was_mtp_interrupted=was_mtp,
            prefix_text=text[:last_open] if was_mtp else text,
            mtp_fragment=text[last_open:] if was_mtp else "",
        )


__all__ = [
    "GenerationResult",
    "WorkerAgentService",
]
