"""
HiveMemory 灰度仲裁器 (Grey Area Arbiter)

负责处理语义相似度处于灰度区间（0.40-0.75）的模糊情况，
使用更精细的模型来判断两个意图是否属于同一任务流。

核心功能：
    1. 定义灰度仲裁的抽象接口
    2. 提供 Reranker-based 实现（首选）
    3. 提供 SLM-based 实现（降级方案）

参考: docs/mod/DiscourseContinuity.md

作者: HiveMemory Team
版本: 2.0.0
"""

import logging
from typing import Optional

from hivememory.infrastructure.llm import BaseLLMService
from hivememory.infrastructure.rerank.base import BaseRerankService
from hivememory.engines.perception.interfaces import BaseArbiter
from hivememory.patchouli.config import (
    ArbiterConfig,
    RerankerArbiterConfig,
    SLMArbiterConfig,
)

logger = logging.getLogger(__name__)


# 默认的 SLM 仲裁提示词（模块级别，便于外部访问和自定义）
DEFAULT_ARBITER_PROMPT = """判断以下两个意图是否属于同一个任务流？

上文任务摘要: {previous_context}
当前请求: {current_query}

判定规则:
- 存在因果、递进、指代关系 -> 同一任务 (YES)
- 完全无关的新话题 -> 不同任务 (NO)

只输出 YES 或 NO，不要输出其他内容。"""


class RerankerArbiter(BaseArbiter):
    """
    基于 Reranker 的灰度仲裁器

    使用 Cross-Encoder Reranker 模型进行仲裁。

    优势：
        - 专为句子对相似度设计，精度高
        - 推理速度快（分类任务，无生成）
        - 计算成本低

    Args:
        config: RerankerArbiterConfig 配置
        reranker_service: Reranker 服务实例

    Examples:
        >>> from hivememory.infrastructure.rerank import get_reranker_service
        >>> from hivememory.patchouli.config import RerankerArbiterConfig
        >>> reranker = get_reranker_service()
        >>> config = RerankerArbiterConfig(threshold=0.5)
        >>> arbiter = RerankerArbiter(config, reranker)
    """

    def __init__(
        self,
        config: RerankerArbiterConfig,
        reranker_service: BaseRerankService,
    ):
        self.config = config
        self.reranker = reranker_service
        self.threshold = config.threshold

        logger.info(
            f"RerankerArbiter 初始化: threshold={self.threshold}, "
            f"model={getattr(reranker_service, 'model_name', 'unknown')}"
        )

    def should_continue_topic(
        self,
        previous_context: str,
        current_query: str,
        similarity_score: float,
    ) -> bool:
        """
        使用 Reranker 判断是否继续话题

        Args:
            previous_context: 上一轮上下文
            current_query: 当前查询
            similarity_score: 原始相似度（用于日志）

        Returns:
            bool: 是否继续话题
        """
        if not previous_context or not current_query:
            logger.debug("上下文或查询为空，默认继续")
            return True

        try:
            # 使用 Reranker 计算分数
            scores = self.reranker.compute_score(
                pairs=[[previous_context, current_query]],
                batch_size=1,
            )

            if not scores:
                logger.warning("Reranker 返回空分数，默认继续")
                return True

            reranker_score = float(scores[0])
            should_continue = reranker_score >= self.threshold

            logger.debug(
                f"Reranker 仲裁: "
                f"similarity={similarity_score:.3f} -> "
                f"reranker={reranker_score:.3f} -> "
                f"{'ADSORB' if should_continue else 'SPLIT'}"
            )

            return should_continue

        except Exception as e:
            logger.error(f"Reranker 仲裁失败: {e}，默认继续")
            return True


class SLMArbiter(BaseArbiter):
    """
    基于小型语言模型（SLM）的灰度仲裁器

    使用 LLM 进行结构化输出判断。

    优势：
        - 理解能力更强，可处理复杂关系
        - 可通过 Prompt 精确控制判断逻辑

    劣势：
        - 推理速度较慢（需要生成文本）
        - 成本较高

    Args:
        config: SLMArbiterConfig 配置
        llm_service: LLM 服务实例

    Examples:
        >>> from hivememory.infrastructure.llm import get_llm_service
        >>> from hivememory.patchouli.config import SLMArbiterConfig
        >>> llm = get_llm_service()
        >>> config = SLMArbiterConfig()
        >>> arbiter = SLMArbiter(config, llm)
    """

    def __init__(
        self,
        config: SLMArbiterConfig,
        llm_service: BaseLLMService,
    ):
        self.config = config
        self.llm = llm_service
        self.prompt_template = config.prompt_template or DEFAULT_ARBITER_PROMPT
        self.threshold = config.threshold

        logger.info("SLMArbiter 初始化")

    def should_continue_topic(
        self,
        previous_context: str,
        current_query: str,
        similarity_score: float,
    ) -> bool:
        """
        使用 LLM 判断是否继续话题

        Args:
            previous_context: 上一轮上下文
            current_query: 当前查询
            similarity_score: 原始相似度（用于日志）

        Returns:
            bool: 是否继续话题
        """
        if not previous_context or not current_query:
            logger.debug("上下文或查询为空，默认继续")
            return True

        try:
            # 构建提示词
            prompt = self.prompt_template.format(
                previous_context=previous_context[:500],  # 限制长度
                current_query=current_query[:500],
            )

            # 调用 LLM
            response = self._call_llm(prompt)

            # 解析响应
            should_continue = self._parse_response(response)

            logger.debug(
                f"SLM 仲裁: "
                f"similarity={similarity_score:.3f} -> "
                f"response={response.strip()} -> "
                f"{'ADSORB' if should_continue else 'SPLIT'}"
            )

            return should_continue

        except Exception as e:
            logger.error(f"SLM 仲裁失败: {e}，默认继续")
            return True

    def _call_llm(self, prompt: str) -> str:
        """
        调用 LLM 服务

        Args:
            prompt: 提示词

        Returns:
            str: LLM 响应
        """
        messages = [{"role": "user", "content": prompt}]
        return self.llm.complete(messages)

    def _parse_response(self, response: str) -> bool:
        """
        解析 LLM 响应为布尔值

        Args:
            response: LLM 响应文本

        Returns:
            bool: 是否继续话题
        """
        # 清理响应
        cleaned = response.strip().upper()

        # 判断 YES/NO
        if cleaned.startswith("YES"):
            return True
        elif cleaned.startswith("NO"):
            return False
        else:
            # 无法解析，默认继续（保守策略）
            logger.warning(f"无法解析 LLM 响应: {response}，默认继续")
            return True


class NoOpArbiter(BaseArbiter):
    """
    无操作仲裁器（用于测试或降级）

    始终返回默认值，不调用任何模型。

    Args:
        default_action: 默认动作（True=继续，False=切分）
    """

    def __init__(self, default_action: bool = True):
        self.default_action = default_action
        logger.info(f"NoOpArbiter 初始化: default_action={default_action}")

    def should_continue_topic(
        self,
        previous_context: str,
        current_query: str,
        similarity_score: float,
    ) -> bool:
        """返回默认动作"""
        logger.debug(
            f"NoOpArbiter: similarity={similarity_score:.3f} -> "
            f"{'ADSORB' if self.default_action else 'SPLIT'}"
        )
        return self.default_action


def create_arbiter(
    config: ArbiterConfig,
    reranker_service: Optional[BaseRerankService] = None,
    llm_service: Optional[BaseLLMService] = None,
) -> BaseArbiter:
    """
    灰度仲裁器工厂函数

    Args:
        config: 仲裁器配置
        reranker_service: Reranker 服务 (用于 RerankerArbiter)
        llm_service: LLM 服务 (用于 SLMArbiter)

    Returns:
        BaseArbiter: 仲裁器实例
    """
    if not config.enabled:
        return NoOpArbiter(default_action=True)

    impl_config = config.engine

    if isinstance(impl_config, RerankerArbiterConfig):
        if not reranker_service:
            logger.warning("启用 RerankerArbiter 但未提供服务，回退到 NoOpArbiter")
            return NoOpArbiter()
        return RerankerArbiter(config=impl_config, reranker_service=reranker_service)

    elif isinstance(impl_config, SLMArbiterConfig):
        if not llm_service:
            logger.warning("启用 SLMArbiter 但未提供服务，回退到 NoOpArbiter")
            return NoOpArbiter()
        return SLMArbiter(config=impl_config, llm_service=llm_service)

    return NoOpArbiter()


__all__ = [
    "RerankerArbiter",
    "SLMArbiter",
    "NoOpArbiter",
    "create_arbiter",
]
