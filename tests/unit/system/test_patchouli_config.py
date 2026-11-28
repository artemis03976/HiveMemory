"""Patchouli 配置公共表面的回归测试。"""

import pytest
from pydantic import ValidationError

from hivememory.config.patchouli import (
    HybridRetrieverConfig,
    LLMSummaryWriterConfig,
    MemoryGenerationConfig,
    MemoryPerceptionConfig,
    MemoryRetrievalConfig,
)


def test_active_generation_material_defaults_to_five_recent_blocks() -> None:
    assert MemoryGenerationConfig().active_recent_blocks == 5
    assert MemoryGenerationConfig(active_recent_blocks=0).active_recent_blocks == 0


def test_active_generation_material_rejects_negative_block_count() -> None:
    with pytest.raises(ValidationError, match="active_recent_blocks"):
        MemoryGenerationConfig(active_recent_blocks=-1)


def test_settle_after_defaults_to_one_hour() -> None:
    """代码缺省与随附配置相同，写进展记录的阈值（缺省 1800 秒）因此比沉淀的短。"""
    config = MemoryPerceptionConfig()
    assert (config.settle_after_seconds, config.progress_record_after_seconds) == (3600, 1800)


def test_thresholds_are_read_by_their_action_names() -> None:
    """两个阈值按到点后做的事命名；旧键 idle_timeout_seconds 是未知键，启动时显式失败。"""
    config = MemoryPerceptionConfig(settle_after_seconds=60, progress_record_after_seconds=30)
    assert (config.settle_after_seconds, config.progress_record_after_seconds) == (60, 30)
    with pytest.raises(ValidationError, match="idle_timeout_seconds"):
        MemoryPerceptionConfig.model_validate({"idle_timeout_seconds": 60})


def test_volume_thresholds_may_reach_the_fold_threshold() -> None:
    """沉淀的阈值可以等于折叠阈值，进展记录的只要比它小。"""
    config = MemoryPerceptionConfig(
        progress_record_after_tokens=1,
        settle_after_tokens=4096,
        engine={"fold_token_threshold": 4096},
    )
    assert config.settle_after_tokens == config.engine.fold_token_threshold


@pytest.mark.parametrize(
    ("progress_tokens", "settle_tokens", "fold_tokens"),
    [(4096, 4096, 8192), (1024, 8192, 4096)],
    ids=["progress-not-below-settle", "settle-above-fold"],
)
def test_volume_thresholds_must_stay_ordered_below_the_fold_threshold(
    progress_tokens, settle_tokens, fold_tokens
) -> None:
    """进展记录的阈值不小于沉淀的、或沉淀的超过折叠阈值时，配置加载失败。"""
    with pytest.raises(ValidationError, match="progress_record_after_tokens < settle_after_tokens"):
        MemoryPerceptionConfig(
            progress_record_after_tokens=progress_tokens,
            settle_after_tokens=settle_tokens,
            engine={"fold_token_threshold": fold_tokens},
        )


def test_max_resident_topics_is_read_from_the_perception_section() -> None:
    """驻留容量在 perception 段；engine 段的旧位置是未知键，启动时显式失败。"""
    assert MemoryPerceptionConfig(max_resident_topics=3).max_resident_topics == 3
    with pytest.raises(ValidationError, match="max_resident_topics"):
        MemoryPerceptionConfig.model_validate({"engine": {"max_resident_topics": 3}})


def test_summary_writer_replaces_the_relay_section() -> None:
    """折叠摘要与进展记录的写法在 engine.summary_writer；旧的 relay 段是未知键，显式失败。"""
    config = MemoryPerceptionConfig.model_validate(
        {"engine": {"summary_writer": {"type": "llm", "summary_timeout_seconds": 5}}}
    )
    assert config.engine.summary_writer == LLMSummaryWriterConfig(summary_timeout_seconds=5)
    with pytest.raises(ValidationError, match="relay"):
        MemoryPerceptionConfig.model_validate({"engine": {"relay": {"engine": {"type": "llm"}}}})


def test_settle_mode_defaults_to_one_shot_and_accepts_only_two_modes() -> None:
    """旧配置没有这一项时取一次性生成，升级后沉淀行为不变。"""
    assert MemoryGenerationConfig().settle_mode == "one_shot"
    assert MemoryGenerationConfig(settle_mode="librarian").settle_mode == "librarian"
    with pytest.raises(ValidationError, match="settle_mode"):
        MemoryGenerationConfig(settle_mode="agent")


def test_librarian_budget_defaults_to_twelve_steps_and_five_search_results() -> None:
    librarian = MemoryGenerationConfig().librarian
    assert (librarian.max_steps, librarian.search_top_k) == (12, 5)


@pytest.mark.parametrize("field", ["max_steps", "search_top_k"])
def test_librarian_budget_must_be_positive(field) -> None:
    with pytest.raises(ValidationError, match=field):
        MemoryGenerationConfig(librarian={field: 0})


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
    assert config.retriever.top_k == 5


def test_retrieval_result_count_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        HybridRetrieverConfig(top_k=0)
