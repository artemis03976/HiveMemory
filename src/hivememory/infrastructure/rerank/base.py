"""
Rerank 服务基础模块

包含 Rerank 服务抽象接口。
"""

from abc import ABC, abstractmethod


class BaseRerankService(ABC):
    """
    Rerank 服务抽象接口

    定义所有 Rerank 服务必须实现的方法。
    """

    @abstractmethod
    async def compute_score(self, pairs: list[list[str]]) -> list[float]:
        """
        计算文本对的相似度分数

        Args:
            pairs: 文本对列表 [[query, passage], ...]

        Returns:
            分数列表
        """
        pass
