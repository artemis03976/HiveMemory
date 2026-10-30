"""
HiveMemory Embedding 基础模块

包含 Embedding 服务抽象接口与编码结果类型。
"""

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class DenseSparseEncoding(TypedDict):
    """单条文本同时请求 dense 与 sparse 时的编码结果。"""

    dense: list[float]
    sparse_text: str


# encode 的全部返回形状：dense 单条/批量、sparse 原文、dense+sparse 组合。
type EmbeddingOutput = (
    list[float] | list[list[float]] | str | list[str] | DenseSparseEncoding | dict[str, Any]
)


class BaseEmbeddingService(ABC):
    """
    Embedding 服务抽象接口

    定义所有 embedding 服务必须实现的方法。
    """

    @abstractmethod
    async def encode(
        self,
        dense_texts: str | list[str] | None = None,
        sparse_texts: str | list[str] | None = None,
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """
        编码 dense 向量，或返回 sparse 原文供向量库的 BM25 使用

        Args:
            dense_texts: 需要 dense 向量的单个文本或文本列表
            sparse_texts: 需要 sparse 检索的单个文本或文本列表

        Returns:
            dense only 为向量或向量列表；sparse only 为原文；两者都给时为
            ``{"dense": ..., "sparse_text": ...}``
        """
        pass
