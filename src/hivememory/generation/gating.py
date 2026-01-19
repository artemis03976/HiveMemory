"""
HiveMemory - 价值评估器 (Value Gater)

.. deprecated::
    v2.0: 此模块的功能已被 Global Gateway 的 memory_signal 替代。

    使用 `hivememory.gateway.GlobalGateway` 获取 worth_saving 信号。
    此模块保留作为备用或二次验证，将在未来版本中移除。

职责:
    判断对话片段是否有长期记忆价值，过滤无价值的闲聊和噪音。

实现策略:
    - RuleBasedGater: 基于关键词黑名单的规则引擎
    - LLMAssistedGater: LLM 辅助判断（可选）
    - HybridGater: 规则 + LLM 结合

作者: HiveMemory Team
版本: 0.1.0
"""

import warnings

# 发出废弃警告
warnings.warn(
    "generation/gating.py is deprecated since v2.0. "
    "Use hivememory.gateway.GlobalGateway.memory_signal.worth_saving instead. "
    "This module is kept as a backup and will be removed in v2.1.",
    DeprecationWarning,
    stacklevel=2
)

import logging
from typing import List, Set, Optional
import re

from hivememory.generation.models import ConversationMessage
from hivememory.generation.interfaces import ValueGater

logger = logging.getLogger(__name__)


class RuleBasedGater(ValueGater):
    """
    基于规则的价值评估器

    策略:
        1. 黑名单过滤 - 简单寒暄和确认
        2. 白名单强制保留 - 代码、配置等
        3. 长度过滤 - 过短对话通常无价值
        4. 实质内容检测 - 包含专业词汇

    优点:
        - 快速、无成本
        - 可解释性强

    缺点:
        - 规则难以覆盖所有情况
        - 需要持续维护规则库
    """

    # 黑名单关键词 - 简单寒暄和确认
    TRIVIAL_PATTERNS: Set[str] = {
        # 寒暄
        "你好", "您好", "hi", "hello", "嗨",

        # 确认
        "好的", "明白", "了解", "收到", "ok", "好", "是的", "嗯",

        # 感谢
        "谢谢", "感谢", "多谢", "thanks", "thank you",

        # 道歉
        "抱歉", "对不起", "sorry",

        # 闲聊
        "随便聊聊", "没事", "没什么", "算了",

        # 简单提问
        "在吗", "怎么样", "好不好",

        # 辅助词
        "帮助", "聊天", "高兴",
    }

    # 白名单关键词 - 强制保留
    VALUABLE_PATTERNS: Set[str] = {
        # 代码相关
        "代码", "函数", "class", "def", "import", "function", "```",

        # 配置相关
        "配置", "设置", "参数", "环境变量", "config",

        # 技术术语
        "API", "数据库", "算法", "架构", "设计模式",

        # 问题解决
        "错误", "bug", "修复", "解决", "问题",

        # 文档资料
        "文档", "教程", "链接", "资料", "URL", "http",
    }

    def __init__(
        self,
        min_total_length: int = 20,
        min_substantive_length: int = 10,
    ):
        """
        初始化规则评估器

        Args:
            min_total_length: 对话总长度最小值（字符数）
            min_substantive_length: 实质内容最小长度
        """
        self.min_total_length = min_total_length
        self.min_substantive_length = min_substantive_length

    def evaluate(self, messages: List[ConversationMessage]) -> bool:
        """
        评估对话是否有价值

        评估流程:
            1. 白名单检查 → 快速通过
            2. 长度过滤 → 过短内容丢弃
            3. 黑名单检查 → 简单寒暄丢弃
            4. 实质内容检测 → 最终判断

        Args:
            messages: 对话消息列表

        Returns:
            bool: True 表示有价值，False 表示无价值

        Examples:
            >>> gater = RuleBasedGater()
            >>> messages = [
            ...     ConversationMessage(role="user", content="你好"),
            ...     ConversationMessage(role="assistant", content="你好！有什么可以帮助你的吗？")
            ... ]
            >>> gater.evaluate(messages)
            False

            >>> messages = [
            ...     ConversationMessage(role="user", content="帮我写个快排算法"),
            ...     ConversationMessage(role="assistant", content="好的，这是代码...")
            ... ]
            >>> gater.evaluate(messages)
            True
        """
        if not messages:
            logger.debug("空消息列表，判定为无价值")
            return False

        # 合并所有消息内容
        full_text = " ".join(msg.content for msg in messages)

        # Step 1: 白名单检查 - 包含技术关键词直接通过
        if self._contains_valuable_keywords(full_text):
            logger.debug("命中白名单关键词，判定为有价值")
            return True

        # Step 2: 长度过滤 - 过短对话通常无价值
        if len(full_text) < self.min_total_length:
            logger.debug(f"对话长度过短 ({len(full_text)} < {self.min_total_length})，判定为无价值")
            return False

        # Step 3: 黑名单检查 - 纯寒暄直接丢弃
        if self._is_pure_trivial(full_text):
            logger.debug("对话为纯寒暄/确认，判定为无价值")
            return False

        # Step 4: 实质内容检测
        substantive_text = self._extract_substantive_content(full_text)
        if len(substantive_text) < self.min_substantive_length:
            logger.debug(f"实质内容过少 ({len(substantive_text)} < {self.min_substantive_length})，判定为无价值")
            return False

        # 默认通过
        logger.debug("通过规则检查，判定为有价值")
        return True

    def _contains_valuable_keywords(self, text: str) -> bool:
        """
        检查是否包含白名单关键词

        Args:
            text: 对话文本

        Returns:
            bool: 是否包含有价值关键词
        """
        text_lower = text.lower()
        return any(pattern.lower() in text_lower for pattern in self.VALUABLE_PATTERNS)

    def _is_pure_trivial(self, text: str) -> bool:
        """
        检查是否为纯寒暄/确认

        策略:
            移除寒暄词后，剩余内容很少

        Args:
            text: 对话文本

        Returns:
            bool: 是否为纯寒暄
        """
        # 移除所有黑名单词汇
        cleaned_text = text
        for pattern in self.TRIVIAL_PATTERNS:
            cleaned_text = cleaned_text.replace(pattern, "")

        # 移除标点和空白
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        cleaned_text = cleaned_text.strip()

        # 如果剩余内容很少，则判定为纯寒暄
        return len(cleaned_text) < 5

    def _extract_substantive_content(self, text: str) -> str:
        """
        提取实质内容

        策略:
            移除寒暄词、标点、过多空白

        Args:
            text: 对话文本

        Returns:
            str: 实质内容
        """
        # 移除黑名单词汇
        cleaned_text = text
        for pattern in self.TRIVIAL_PATTERNS:
            cleaned_text = cleaned_text.replace(pattern, "")

        # 移除多余空白
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text


class LLMAssistedGater(ValueGater):
    """
    LLM 辅助的价值评估器

    策略:
        使用轻量级模型 (GPT-4o-mini / Haiku) 判断对话价值。

    优点:
        - 理解上下文和语义
        - 泛化能力强

    缺点:
        - 增加延迟和成本
        - 需要 LLM API

    注意:
        该实现为骨架，实际使用时需要配置 LLM API。
    """

    EVALUATION_PROMPT = """请判断以下对话是否有长期记忆价值。

**有价值的对话**:
- 包含代码实现
- 技术问题解答
- 配置和设置
- 用户偏好声明
- 重要决策记录

**无价值的对话**:
- 简单寒暄 ("你好"、"谢谢")
- 纯确认回复 ("好的"、"明白")
- 闲聊内容

对话内容:
{transcript}

请回答: 有价值 或 无价值
"""

    def __init__(self, llm_client=None, llm_config: Optional[object] = None):
        """
        初始化 LLM 评估器

        Args:
            llm_client: LLM 客户端（可选）
            llm_config: LLM 配置（可选，预留）
        """
        self.llm_client = llm_client
        self.llm_config = llm_config

    def evaluate(self, messages: List[ConversationMessage]) -> bool:
        """
        使用 LLM 评估对话价值

        Args:
            messages: 对话消息列表

        Returns:
            bool: True 表示有价值

        Raises:
            NotImplementedError: 当前为骨架实现
        """
        # TODO: 实现 LLM 调用逻辑
        logger.warning("LLMAssistedGater 未实现，降级为 RuleBasedGater")

        # 降级到规则引擎
        fallback_gater = RuleBasedGater()
        return fallback_gater.evaluate(messages)


class HybridGater(ValueGater):
    """
    混合评估器

    策略:
        1. 规则引擎快速过滤明显无价值内容
        2. LLM 精细判断边界情况

    优点:
        - 兼顾速度和准确性
        - 降低 LLM 调用成本
    """

    def __init__(self, rule_gater: RuleBasedGater, llm_gater: LLMAssistedGater):
        """
        初始化混合评估器

        Args:
            rule_gater: 规则评估器
            llm_gater: LLM 评估器
        """
        self.rule_gater = rule_gater
        self.llm_gater = llm_gater

    def evaluate(self, messages: List[ConversationMessage]) -> bool:
        """
        混合评估

        流程:
            1. 规则引擎快速判断
            2. 如果规则不确定，调用 LLM

        Args:
            messages: 对话消息列表

        Returns:
            bool: True 表示有价值
        """
        # 先用规则引擎
        rule_result = self.rule_gater.evaluate(messages)

        # TODO: 添加"不确定"状态，触发 LLM 二次判断
        # 当前简化为直接返回规则结果
        return rule_result


# 便捷函数
def create_default_gater(
    config: Optional["GaterConfig"] = None,
) -> ValueGater:
    """
    创建默认评估器（支持配置）

    Args:
        config: 评估器配置（可选，使用默认配置）

    Returns:
        ValueGater: 规则引擎实例

    Examples:
        >>> # 使用默认配置
        >>> gater = create_default_gater()
        >>>
        >>> # 使用自定义配置
        >>> from hivememory.core.config import GaterConfig
        >>> config = GaterConfig(min_total_length=30)
        >>> gater = create_default_gater(config)
    """
    if config is None:
        from hivememory.core.config import GaterConfig
        config = GaterConfig()

    gater_type = config.gater_type

    if gater_type == "llm":
        # LLM 辅助评估器
        return LLMAssistedGater(llm_config=config.llm_config)

    elif gater_type == "hybrid":
        # 混合评估器
        rule_gater = RuleBasedGater(
            min_total_length=config.min_total_length,
            min_substantive_length=config.min_substantive_length,
        )
        # 支持自定义黑白名单
        if config.trivial_patterns:
            rule_gater.TRIVIAL_PATTERNS = set(config.trivial_patterns)
        if config.valuable_patterns:
            rule_gater.VALUABLE_PATTERNS = set(config.valuable_patterns)

        llm_gater = LLMAssistedGater(llm_config=config.llm_config)
        return HybridGater(rule_gater=rule_gater, llm_gater=llm_gater)

    else:  # rule (默认)
        # 规则引擎评估器
        gater = RuleBasedGater(
            min_total_length=config.min_total_length,
            min_substantive_length=config.min_substantive_length,
        )
        # 支持自定义黑白名单
        if config.trivial_patterns:
            gater.TRIVIAL_PATTERNS = set(config.trivial_patterns)
        if config.valuable_patterns:
            gater.VALUABLE_PATTERNS = set(config.valuable_patterns)
        return gater


__all__ = [
    "RuleBasedGater",
    "LLMAssistedGater",
    "HybridGater",
    "create_default_gater",
]
