"""
FlagEmbedding Reranker 服务实现
"""

import logging
from typing import List, Optional
from functools import lru_cache

from hivememory.infrastructure.rerank.base import SingletonModelService

logger = logging.getLogger(__name__)


class FlagRerankerService(SingletonModelService):
    """
    基于 FlagEmbedding 的 Reranker 服务
    """

    def _load_model(self) -> None:
        """加载 FlagReranker 模型"""
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "FlagEmbedding 未安装。请运行: pip install FlagEmbedding"
            )

        logger.info(f"正在加载 Reranker 模型: {self.model_name}")
        try:
            self._model = FlagReranker(
                model_name_or_path=self.model_name,
                device=self.device,
                use_fp16=self.use_fp16,
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
        """
        if not pairs:
            return []

        try:
            return self.model.compute_score(
                pairs,
                batch_size=batch_size,
                max_length=max_length
            )
        except Exception as e:
            logger.error(f"Reranker 计算失败: {e}")
            raise


@lru_cache(maxsize=1)
def get_flag_reranker_service(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cpu",
    use_fp16: bool = True
) -> FlagRerankerService:
    """
    获取全局 FlagReranker 服务实例（单例）
    """
    return FlagRerankerService(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16
    )
