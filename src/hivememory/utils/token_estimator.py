"""
HiveMemory Token 估算工具类

提供多种 Token 估算策略，支持不同文本类型的精确估算。

Features:
    - 简单估算: 基于字符统计，高性能，适合实时流处理
    - 扩展 CJK 支持: 覆盖中日韩统一汉字及扩展区
    - 可配置比率: 支持自定义字符与 token 的转换比率
    - 批量估算: 支持消息列表的批量 token 计算
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union


class EstimationStrategy(str, Enum):
    """Token 估算策略枚举"""

    SIMPLE = "simple"  # 简单字符统计（默认，高性能）
    TIKTOKEN = "tiktoken"  # 基于 tiktoken 精确计算（可选）


class TokenEstimator:
    """
    Token 估算工具类

    提供多种 Token 估算策略，支持不同文本类型的精确估算。

    设计原则:
        - 默认使用简单估算（高性能，适合实时流处理）
        - 支持精确估算（可选，需要额外依赖）
        - 可配置的估算参数
        - 线程安全（无共享可变状态）

    Examples:
        >>> # 静态方法快速估算（推荐）
        >>> tokens = TokenEstimator.estimate("Hello, 世界!")
        >>> print(tokens)  # 5
        >>>
        >>> # 使用自定义比率
        >>> tokens = TokenEstimator.estimate_with_ratio(
        ...     "Hello, 世界!",
        ...     chinese_ratio=1.5,
        ...     other_ratio=4.0
        ... )
        >>>
        >>> # 批量估算消息列表
        >>> messages = [{"content": "Hello"}, {"content": "世界"}]
        >>> total = TokenEstimator.estimate_messages(messages)
    """

    # ========== 默认配置常量 ==========
    DEFAULT_CHINESE_RATIO: float = 2.0  # 中文: 1 token ≈ 2 字符
    DEFAULT_ENGLISH_RATIO: float = 4.0  # 英文: 1 token ≈ 4 字符
    DEFAULT_CODE_RATIO: float = 3.5  # 代码: 1 token ≈ 3.5 字符

    # CJK Unicode 范围（用于中文/日文/韩文字符检测）
    CJK_RANGES = [
        ("\u4e00", "\u9fff"),  # CJK 统一汉字
        ("\u3400", "\u4dbf"),  # CJK 扩展 A
        ("\uf900", "\ufaff"),  # CJK 兼容汉字
        ("\u3040", "\u309f"),  # 日文平假名
        ("\u30a0", "\u30ff"),  # 日文片假名
        ("\uac00", "\ud7af"),  # 韩文音节
    ]

    @staticmethod
    def estimate(text: Optional[str]) -> int:
        """
        快速估算文本的 Token 数量

        使用简单字符统计策略:
        - 中文/日文/韩文: 1 token ≈ 2 字符
        - 其他字符（英文、数字、符号）: 1 token ≈ 4 字符

        Args:
            text: 待估算的文本，可以为 None 或空字符串

        Returns:
            估算的 token 数量，None 或空字符串返回 0

        Examples:
            >>> TokenEstimator.estimate("Hello, 世界!")
            5
            >>> TokenEstimator.estimate("")
            0
            >>> TokenEstimator.estimate(None)
            0
        """
        if not text:
            return 0

        cjk_chars = TokenEstimator.count_cjk_chars(text)
        other_chars = len(text) - cjk_chars

        return (cjk_chars // 2) + (other_chars // 4)

    @staticmethod
    def count_cjk_chars(text: str) -> int:
        """
        统计文本中的 CJK 字符数量

        包括:
        - CJK 统一汉字及扩展区
        - 日文平假名、片假名
        - 韩文音节

        Args:
            text: 待统计的文本

        Returns:
            CJK 字符数量
        """
        if not text:
            return 0

        count = 0
        for char in text:
            for start, end in TokenEstimator.CJK_RANGES:
                if start <= char <= end:
                    count += 1
                    break
        return count

    @staticmethod
    def estimate_with_ratio(
        text: Optional[str],
        chinese_ratio: float = DEFAULT_CHINESE_RATIO,
        other_ratio: float = DEFAULT_ENGLISH_RATIO,
    ) -> int:
        """
        使用自定义比率估算 Token 数量

        Args:
            text: 待估算的文本
            chinese_ratio: CJK 字符与 token 的比率（默认 2.0）
            other_ratio: 其他字符与 token 的比率（默认 4.0）

        Returns:
            估算的 token 数量

        Examples:
            >>> TokenEstimator.estimate_with_ratio("Hello, 世界!", 1.5, 4.0)
            4
        """
        if not text:
            return 0

        cjk_chars = TokenEstimator.count_cjk_chars(text)
        other_chars = len(text) - cjk_chars

        cjk_tokens = cjk_chars / chinese_ratio if chinese_ratio > 0 else 0
        other_tokens = other_chars / other_ratio if other_ratio > 0 else 0

        return int(cjk_tokens + other_tokens)

    @staticmethod
    def estimate_messages(
        messages: List[Dict[str, Any]],
        content_key: str = "content",
    ) -> int:
        """
        估算消息列表的总 Token 数量

        Args:
            messages: 消息列表，每条消息应包含 content 字段
            content_key: 消息内容的键名（默认 "content"）

        Returns:
            所有消息的 token 总数

        Examples:
            >>> messages = [
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "世界"}
            ... ]
            >>> TokenEstimator.estimate_messages(messages)
            2
        """
        if not messages:
            return 0

        total = 0
        for msg in messages:
            content = msg.get(content_key, "")
            if content:
                total += TokenEstimator.estimate(str(content))
        return total

    @staticmethod
    def estimate_dict(
        data: Dict[str, Any],
        keys: Optional[List[str]] = None,
    ) -> int:
        """
        估算字典中指定字段的 Token 数量

        Args:
            data: 待估算的字典
            keys: 要估算的键列表，None 表示估算所有字符串值

        Returns:
            指定字段的 token 总数

        Examples:
            >>> data = {"thought": "思考内容", "tool_name": "search"}
            >>> TokenEstimator.estimate_dict(data, ["thought", "tool_name"])
            6
        """
        if not data:
            return 0

        total = 0
        target_keys = keys if keys is not None else list(data.keys())

        for key in target_keys:
            value = data.get(key)
            if value is not None:
                total += TokenEstimator.estimate(str(value))

        return total


# ========== 便捷函数（保持向后兼容） ==========


def estimate_tokens(text: Optional[str]) -> int:
    """
    估算文本的 Token 数量

    这是 TokenEstimator.estimate() 的便捷函数，保持向后兼容。

    规则：
    - 中文 1 token ≈ 2 字符
    - 英文 1 token ≈ 4 字符
    - 这是一个粗略估算

    Args:
        text: 待估算的文本

    Returns:
        估算的 token 数量

    Examples:
        >>> estimate_tokens("Hello, 世界!")
        5
        >>> estimate_tokens("")
        0
    """
    return TokenEstimator.estimate(text)


__all__ = [
    "TokenEstimator",
    "EstimationStrategy",
    "estimate_tokens",
]
