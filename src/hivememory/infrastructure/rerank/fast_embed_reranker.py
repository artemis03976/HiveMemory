"""
FastEmbed Cross-Encoder Reranker 服务实现
"""

import logging
from typing import TYPE_CHECKING

from hivememory.infrastructure.rerank.base import SingletonModelService

if TYPE_CHECKING:
    from hivememory.config.patchouli import RerankerConfig

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
            raise ImportError("fastembed 未安装。请运行: pip install fastembed")

        logger.info(f"正在加载 Reranker 模型: {self.model_name}")
        try:
            # 关闭 ONNX Runtime 的 CPU 内存池：开启时推理的峰值分配在推理结束后仍留在进程内。
            self._model = TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                enable_cpu_mem_arena=False,
            )
            logger.info("Reranker 模型加载完成")
        except Exception as e:
            logger.error(f"Reranker 模型加载失败: {e}")
            raise

    async def compute_score(self, pairs: list[list[str]]) -> list[float]:
        """
        计算文本对的相似度分数

        计算（含首次使用时的模型加载）在推理线程中执行，每批送入配置的
        ``batch_size`` 对；输入按模型的 512 token 上限截断。

        Args:
            pairs: [[query, doc], ...] 格式的文本对列表

        Returns:
            分数列表，与输入 pairs 顺序一致
        """
        if not pairs:
            return []
        return await self._inference.run(self._score_pairs, pairs)

    def _score_pairs(self, pairs: list[list[str]]) -> list[float]:
        """在推理线程中计算重排分数。"""
        try:
            query = pairs[0][0]
            documents = [p[1] for p in pairs]
            scores = list(self.model.rerank(query, documents, batch_size=self.config.batch_size))
            return [float(score) for score in scores]
        except Exception as e:
            logger.error(f"Reranker 计算失败: {e}")
            raise


def get_fast_embed_reranker_service(config: "RerankerConfig") -> FastEmbedRerankerService:
    """
    获取全局 FastEmbed Reranker 服务实例（单例）
    """
    return FastEmbedRerankerService(config=config)
