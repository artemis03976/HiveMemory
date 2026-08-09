"""Patchouli 配置公共表面的回归测试。"""

from hivememory.system.config import (
    AdaptiveWeightedFusionConfig,
    MemoryGenerationConfig,
    MemoryLifecycleConfig,
    PatchouliConfig,
    RetrievalModeConfig,
)


def test_memory_generation_queue_policy_has_local_defaults() -> None:
    config = MemoryGenerationConfig()

    assert config.queue_capacity == 128
    assert config.queue_max_concurrency == 2
    assert config.queue_timeout_seconds == 300.0
    assert config.queue_max_attempts == 3


def test_unwired_fields_are_absent_from_patchouli_config_surface() -> None:
    assert "time_weight" not in RetrievalModeConfig.model_fields
    assert "time_weight" not in RetrievalModeConfig.model_json_schema()["properties"]
    assert "high_watermark" not in MemoryLifecycleConfig.model_fields
    assert "high_watermark" not in MemoryLifecycleConfig.model_json_schema()["properties"]

    config = PatchouliConfig()
    payload = config.model_dump(mode="json")
    adaptive = AdaptiveWeightedFusionConfig()

    assert "high_watermark" not in payload["lifecycle"]
    for mode_name in ("debug_mode", "concept_mode", "timeline_mode", "brainstorm_mode"):
        assert "time_weight" not in getattr(adaptive, mode_name).model_dump()


def test_legacy_unwired_fields_are_pruned_during_validation() -> None:
    mode = RetrievalModeConfig.model_validate({"time_weight": 0.8})
    lifecycle = MemoryLifecycleConfig.model_validate({"high_watermark": 80.0})

    assert "time_weight" not in mode.model_dump()
    assert "high_watermark" not in lifecycle.model_dump()


def test_adaptive_fusion_keeps_wired_mode_weights() -> None:
    config = AdaptiveWeightedFusionConfig()

    assert config.debug_mode.dense_weight == 0.3
    assert config.debug_mode.sparse_weight == 0.9
    assert config.concept_mode.dense_weight == 0.8
    assert config.concept_mode.sparse_weight == 0.2
    assert config.timeline_mode.dense_weight == 0.4
    assert config.timeline_mode.sparse_weight == 0.3
    assert config.brainstorm_mode.dense_weight == 0.6
    assert config.brainstorm_mode.sparse_weight == 0.1
