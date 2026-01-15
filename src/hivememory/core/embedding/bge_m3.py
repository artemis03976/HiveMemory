"""
HiveMemory BGE-M3 混合 Embedding 服务 (FlagEmbedding 实现)
"""

import logging
import threading
from typing import List, Union, Dict, Any, Optional
from functools import lru_cache

import numpy as np

from hivememory.core.embedding.base import SingletonModelService

logger = logging.getLogger(__name__)


class BGEM3EmbeddingService(SingletonModelService):
    """
    BGE-M3 混合 Embedding 服务
    
    支持:
    - Dense Vector: 语义向量 (维度 1024)
    - Sparse Vector: 稀疏向量 (词汇权重)
    """

    def _load_model(self) -> None:
        """加载 BGE-M3 模型"""
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding 未安装。请运行: pip install FlagEmbedding"
            )

        logger.info(f"正在加载 BGE-M3 模型: {self.model_name}")
        try:
            self._model = BGEM3FlagModel(
                model_name_or_path=self.model_name,
                device=self.device,
                use_fp16=self.device != "cpu",
                show_progress_bar=False,
            )
            logger.info("BGE-M3 模型加载完成")
        except Exception as e:
            logger.error(f"BGE-M3 模型加载失败: {e}")
            raise

    def encode(
        self,
        dense_texts: Union[str, List[str], None] = None,
        sparse_texts: Union[str, List[str], None] = None,
        **kwargs  # 兼容基类签名
    ) -> Union[List[float], Dict[int, float], Dict[str, Any], List[Dict[str, Any]]]:
        """
        编码文本为稠密向量、稀疏向量或混合向量
        """
        # 至少需要一种输入
        if dense_texts is None and sparse_texts is None:
            # 如果是为了兼容 BaseEmbeddingService.encode(texts=...) 调用
            if 'texts' in kwargs:
                dense_texts = kwargs['texts']
            else:
                raise ValueError("至少需要提供 dense_texts 或 sparse_texts 参数")

        # 处理稠密向量
        dense_result = None
        if dense_texts is not None:
            single_dense_input = isinstance(dense_texts, str)
            dense_input_list = [dense_texts] if single_dense_input else dense_texts

            try:
                output = self.model.encode(
                    dense_input_list,
                    batch_size=1,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                dense_vecs = output['dense_vecs']
                if single_dense_input:
                    dense_result = dense_vecs[0].tolist()
                else:
                    dense_result = dense_vecs.tolist()
            except Exception as e:
                logger.warning(f"稠密向量编码失败: {e}")
                size = 1 if single_dense_input else len(dense_input_list)
                dense_result = np.zeros((size, 1024)).tolist() if not single_dense_input else []

        # 处理稀疏向量
        sparse_result = None
        if sparse_texts is not None:
            single_sparse_input = isinstance(sparse_texts, str)
            sparse_input_list = [sparse_texts] if single_sparse_input else sparse_texts

            try:
                output = self.model.encode(
                    sparse_input_list,
                    batch_size=1,
                    return_dense=False,
                    return_sparse=True,
                    return_colbert_vecs=False,
                    verbose=False
                )
                lexical_weights = output['lexical_weights']

                if single_sparse_input:
                    sparse_weights = lexical_weights[0]
                    sparse_result = {
                        int(k): float(v) for k, v in sparse_weights.items()
                    }
                else:
                    sparse_result = []
                    for weights in lexical_weights:
                        sparse_result.append({
                            int(k): float(v) for k, v in weights.items()
                        })
            except Exception as e:
                logger.warning(f"稀疏向量编码失败: {e}")
                sparse_result = {} if single_sparse_input else []

        # 返回结果
        if dense_result is not None and sparse_result is not None:
            return {"dense": dense_result, "sparse": sparse_result}
        elif dense_result is not None:
            return dense_result
        else:
            return sparse_result

    def get_dimension(self) -> int:
        """获取稠密向量维度"""
        return 1024


@lru_cache(maxsize=1)
def get_bge_m3_service(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    cache_dir: Optional[str] = None
) -> BGEM3EmbeddingService:
    """
    获取全局 BGE-M3 服务实例（单例）
    """
    return BGEM3EmbeddingService(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir
    )
