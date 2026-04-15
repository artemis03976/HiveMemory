"""
FastEmbed Cross-Encoder Reranker 服务实现
"""

import logging
from typing import List

from hivememory.infrastructure.rerank.base import SingletonModelService

logger = logging.getLogger(__name__)


class FastEmbedRerankerService(SingletonModelService):
    """
    基于 FastEmbed TextCrossEncoder 的 Reranker 服务
    """

    def _load_model(self) -> None:
        """加载 FastEmbed Cross-Encoder 模型"""
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            raise ImportError(
                "fastembed 未安装。请运行: pip install fastembed"
            )

        logger.info(f"正在加载 Reranker 模型: {self.model_name}")
        try:
            self._model = TextCrossEncoder(
                model_name=self.model_name,
            )
            logger.info("Reranker 模型加载完成")
        except Exception as e:
            logger.error(f"Reranker 模型加载失败: {e}")
            raise

    def compute_score(
        self,
        pairs: List[List[str]],
        batch_size: int = 256,
        max_length: int = 512
    ) -> List[float]:
        """
        计算文本对的相似度分数

        Args:
            pairs: [[query, doc], ...] 格式的文本对列表
            batch_size: 批处理大小（FastEmbed 内部处理）
            max_length: 最大长度（FastEmbed 内部处理）

        Returns:
            分数列表，与输入 pairs 顺序一致
        """
        if not pairs:
            return []

        try:
            query = pairs[0][0]
            documents = [p[1] for p in pairs]
            scores = list(self.model.rerank(query, documents, batch_size=batch_size))
            return [float(score) for score in scores]
        except Exception as e:
            logger.error(f"Reranker 计算失败: {e}")
            raise


def get_fast_embed_reranker_service(
    config: "RerankerConfig"
) -> FastEmbedRerankerService:
    """
    获取全局 FastEmbed Reranker 服务实例（单例）
    """
    return FastEmbedRerankerService(config=config)
