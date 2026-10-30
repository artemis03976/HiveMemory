"""
FastEmbed Cross-Encoder Reranker 服务实现
"""

import logging
from typing import TYPE_CHECKING

from hivememory.infrastructure.model_service import SingletonModelService
from hivememory.infrastructure.rerank.base import BaseRerankService

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    from hivememory.config.patchouli import RerankerConfig

logger = logging.getLogger(__name__)


class FastEmbedRerankerService(SingletonModelService["TextCrossEncoder"], BaseRerankService):
    """
    基于 FastEmbed TextCrossEncoder 的 Reranker 服务
    """

    config: "RerankerConfig"

    def _load_model(self) -> "TextCrossEncoder":
        """加载 FastEmbed Cross-Encoder 模型"""
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            raise ImportError("fastembed 未安装。请运行: pip install fastembed")

        logger.info(f"正在加载 Reranker 模型: {self.model_name}")
        try:
            # 关闭 ONNX Runtime 的 CPU 内存池：开启时推理的峰值分配在推理结束后仍留在进程内。
            return TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                enable_cpu_mem_arena=False,
            )
        except Exception as e:
            logger.error(f"Reranker 模型加载失败: {e}")
            raise

    async def compute_score(self, pairs: list[list[str]]) -> list[float]:
        """
        计算文本对的相似度分数

        计算（含模型未加载时的加载）在推理线程中执行，每批送入配置的
        ``batch_size`` 对；输入按模型的 512 token 上限截断。

        Args:
            pairs: [[query, doc], ...] 格式的文本对列表

        Returns:
            分数列表，与输入 pairs 顺序一致
        """
        if not pairs:
            return []
        return await self._run_with_model(self._score_pairs, pairs)

    def _score_pairs(self, model: "TextCrossEncoder", pairs: list[list[str]]) -> list[float]:
        """在推理线程中计算重排分数。"""
        try:
            query = pairs[0][0]
            documents = [p[1] for p in pairs]
            scores = list(model.rerank(query, documents, batch_size=self.config.batch_size))
            return [float(score) for score in scores]
        except Exception as e:
            logger.error(f"Reranker 计算失败: {e}")
            raise


def get_fast_embed_reranker_service(config: "RerankerConfig") -> FastEmbedRerankerService:
    """
    获取全局 FastEmbed Reranker 服务实例（单例）
    """
    return FastEmbedRerankerService(config=config)
