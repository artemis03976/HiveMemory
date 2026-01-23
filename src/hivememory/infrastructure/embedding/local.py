"""
HiveMemory 本地 Embedding 服务 (SentenceTransformers 实现)
"""

import logging
from typing import List, Union, Optional, TYPE_CHECKING
from functools import lru_cache

import numpy as np

from hivememory.patchouli.config import load_app_config
if TYPE_CHECKING:
    from hivememory.patchouli.config import EmbeddingConfig
from hivememory.infrastructure.embedding.base import SingletonModelService

logger = logging.getLogger(__name__)


class LocalEmbeddingService(SingletonModelService):
    """
    本地 Embedding 服务 (SentenceTransformers)
    
    使用 sentence-transformers 库加载轻量级模型。
    默认模型：all-MiniLM-L6-v2
    """
    
    def _load_model(self) -> None:
        """加载 sentence-transformers 模型"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装。"
                "请运行: pip install sentence-transformers"
            )

        logger.info(f"正在加载 Embedding 模型: {self.model_name}")
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"模型加载完成，向量维度: {dim}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> Union[List[float], List[List[float]]]:
        """编码文本为向量"""
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress
            )
        except Exception as e:
            logger.warning(f"Encoding 失败，使用零向量: {e}")
            dim = self.get_dimension()
            embeddings = np.zeros((len(texts), dim))

        if single_input:
            return embeddings[0].tolist()
        return embeddings.tolist()

    def compute_similarity(
        self,
        text1: str,
        text2: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        """计算余弦相似度"""
        vec1 = self.encode(text1, normalize=True)

        single_target = isinstance(text2, str)
        if single_target:
            text2 = [text2]

        vecs2 = self.encode(text2, normalize=True)

        vec1_array = np.array(vec1)
        vecs2_array = np.array(vecs2)

        similarities = np.dot(vecs2_array, vec1_array)

        if single_target:
            return float(similarities)
        return similarities.tolist()

    def get_dimension(self) -> int:
        """获取向量维度"""
        if self._model is None:
            # 预设默认维度
            if "all-MiniLM-L6-v2" in self.model_name:
                return 384
            elif "bge-small" in self.model_name:
                return 384
            elif "bge-base" in self.model_name:
                return 768
            else:
                return 384
        return self.model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedding_service(
    config: Optional["EmbeddingConfig"] = None
) -> LocalEmbeddingService:
    """
    获取全局 Embedding 服务实例（单例）
    
    Args:
        config: Embedding 配置对象。如果为 None，则加载全局配置。
    """
    if config is None:
        config = load_app_config().embedding
    
    return LocalEmbeddingService(config=config)
