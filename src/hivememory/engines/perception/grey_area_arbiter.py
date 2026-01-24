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

from hivememory.infrastructure.rerank.base import BaseRerankService
from hivememory.engines.perception.interfaces import GreyAreaArbiter

logger = logging.getLogger(__name__)


# 默认的 SLM 仲裁提示词（模块级别，便于外部访问和自定义）
DEFAULT_ARBITER_PROMPT = """判断以下两个意图是否属于同一个任务流？

上文任务摘要: {previous_context}
当前请求: {current_query}

判定规则:
- 存在因果、递进、指代关系 -> 同一任务 (YES)
- 完全无关的新话题 -> 不同任务 (NO)

只输出 YES 或 NO，不要输出其他内容。"""


class RerankerArbiter(GreyAreaArbiter):
    """
    基于 Reranker 的灰度仲裁器

    使用 Cross-Encoder Reranker 模型进行仲裁。

    优势：
        - 专为句子对相似度设计，精度高
        - 推理速度快（分类任务，无生成）
        - 计算成本低

    Args:
        reranker_service: Reranker 服务实例
        arbiter_threshold: 仲裁阈值（默认 0.5）
        verbose: 是否输出详细日志

    Examples:
        >>> from hivememory.infrastructure.rerank import get_reranker_service
        >>> reranker = get_reranker_service()
        >>> arbiter = RerankerArbiter(reranker)
        >>> result = arbiter.should_continue_topic(
        ...     "写代码",
        ...     "部署服务器",
        ...     0.55
        ... )
    """

    def __init__(
        self,
        reranker_service: BaseRerankService,
        arbiter_threshold: float = 0.5,
        verbose: bool = False,
    ):
        self.reranker = reranker_service
        self.threshold = arbiter_threshold
        self.verbose = verbose

        logger.info(
            f"RerankerArbiter 初始化: threshold={arbiter_threshold}, "
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

        if not self.is_available():
            logger.warning("Reranker 不可用，默认继续")
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

            if self.verbose:
                logger.info(
                    f"Reranker 仲裁: "
                    f"similarity={similarity_score:.3f} -> "
                    f"reranker={reranker_score:.3f} -> "
                    f"{'ADSORB' if should_continue else 'SPLIT'}"
                )

            return should_continue

        except Exception as e:
            logger.error(f"Reranker 仲裁失败: {e}，默认继续")
            return True

    def is_available(self) -> bool:
        """检查 Reranker 是否可用"""
        try:
            return self.reranker.is_loaded()
        except Exception:
            return False


class SLMArbiter(GreyAreaArbiter):
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
        llm_service: LLM 服务实例
        arbiter_prompt: 自定义仲裁提示词（可选）
        arbiter_threshold: 当 LLM 输出不确定时的阈值（暂未使用）

    Examples:
        >>> from hivememory.infrastructure.llm import get_llm_service
        >>> llm = get_llm_service()
        >>> arbiter = SLMArbiter(llm)
        >>> result = arbiter.should_continue_topic(
        ...     "写代码",
        ...     "部署服务器",
        ...     0.55
        ... )
    """

    def __init__(
        self,
        llm_service: any,  # LLM 服务接口（避免循环导入）
        arbiter_prompt: Optional[str] = None,
        arbiter_threshold: float = 0.5,
        verbose: bool = False,
    ):
        self.llm = llm_service
        self.prompt_template = arbiter_prompt or DEFAULT_ARBITER_PROMPT
        self.threshold = arbiter_threshold
        self.verbose = verbose

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

        if not self.is_available():
            logger.warning("LLM 不可用，默认继续")
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

            if self.verbose:
                logger.info(
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
        # 尝试不同的 LLM 接口
        if hasattr(self.llm, "complete"):
            # LangChain 风格
            return self.llm.complete(prompt).text
        elif hasattr(self.llm, "generate"):
            # OpenAI 风格
            response = self.llm.generate([prompt])
            return response.generations[0][0].text
        elif hasattr(self.llm, "chat"):
            # Chat 接口
            from hivememory.core.models import Message
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages)
            return response.content
        elif callable(self.llm):
            # 直接可调用
            return self.llm(prompt)
        else:
            raise ValueError(f"不支持的 LLM 接口: {type(self.llm)}")

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

    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        try:
            return self.llm is not None
        except Exception:
            return False


class NoOpArbiter(GreyAreaArbiter):
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


__all__ = [
    "RerankerArbiter",
    "SLMArbiter",
    "NoOpArbiter",
    "DEFAULT_ARBITER_PROMPT",
]
