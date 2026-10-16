"""Patchouli 配置公共表面的回归测试。"""

import pytest
from pydantic import ValidationError

from hivememory.config.patchouli import (
    HybridRetrieverConfig,
    MemoryGenerationConfig,
    MemoryRetrievalConfig,
)


def test_active_generation_material_defaults_to_five_recent_blocks() -> None:
    assert MemoryGenerationConfig().active_recent_blocks == 5
    assert MemoryGenerationConfig(active_recent_blocks=0).active_recent_blocks == 0


def test_active_generation_material_rejects_negative_block_count() -> None:
    with pytest.raises(ValidationError, match="active_recent_blocks"):
        MemoryGenerationConfig(active_recent_blocks=-1)


def test_retriever_level_reranker_is_rejected_with_migration_hint() -> None:
    """旧位置 retriever.reranker 必须显式失败，而不是被 extra=ignore 静默丢弃。

    拒绝发生在按 type 分派检索器之前，任一检索器类型都走同一检查。
    """
    with pytest.raises(ValidationError, match="已移至 retrieval.reranker"):
        MemoryRetrievalConfig.model_validate(
            {"retriever": {"type": "hybrid", "reranker": {"enabled": False}}}
        )


def test_hybrid_retriever_requires_at_least_one_recall_path() -> None:
    with pytest.raises(ValidationError, match="至少需要启用 dense 或 sparse"):
        HybridRetrieverConfig(dense={"enabled": False}, sparse={"enabled": False})


@pytest.mark.parametrize(
    ("retriever", "message"),
    [
        ({"type": "sparse", "score_threshold": 0.3}, "Sparse 召回配置不再支持 score_threshold"),
        ({"type": "hybrid", "score_threshold": 0.3}, "Hybrid 检索配置不再支持 score_threshold"),
        (
            {"type": "hybrid", "fusion": {"type": "rrf", "final_top_k": 5}},
            "融合配置不再支持 final_top_k",
        ),
        (
            {"type": "hybrid", "fusion": {"type": "adaptive", "final_top_k": 5}},
            "融合配置不再支持 final_top_k",
        ),
    ],
    ids=["sparse", "hybrid", "rrf", "adaptive"],
)
def test_removed_threshold_and_truncation_keys_fail_explicitly(retriever, message) -> None:
    """已删除的阈值与截取键必须显式失败，而不是被 extra=ignore 静默丢弃。"""
    with pytest.raises(ValidationError, match=message):
        MemoryRetrievalConfig.model_validate({"retriever": retriever})


def test_dense_recall_keeps_its_own_score_threshold() -> None:
    """阈值只属于 Dense 召回：单独与 Hybrid 内的 dense 路径都接受配置值。"""
    standalone = MemoryRetrievalConfig.model_validate(
        {"retriever": {"type": "dense", "score_threshold": 0.3}}
    )
    hybrid = MemoryRetrievalConfig.model_validate(
        {"retriever": {"type": "hybrid", "dense": {"score_threshold": 0.3}}}
    )

    assert standalone.retriever.score_threshold == 0.3
    assert hybrid.retriever.dense.score_threshold == 0.3


def test_retrieval_defaults_dense_threshold_and_result_count() -> None:
    config = MemoryRetrievalConfig()

    assert config.retriever.dense.score_threshold == 0.5
    assert config.default_top_k == 5


def test_default_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="default_top_k"):
        MemoryRetrievalConfig(default_top_k=0)
