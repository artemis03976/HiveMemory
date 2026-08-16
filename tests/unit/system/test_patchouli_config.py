"""Patchouli 配置公共表面的回归测试。"""

from hivememory.system.config import (
    AdaptiveWeightedFusionConfig,
    MemoryGenerationConfig,
    MemoryLifecycleConfig,
    PatchouliConfig,
    RetrievalModeConfig,
)


def test_queue_max_attempts_not_in_memory_generation_config() -> None:
    assert "queue_max_attempts" not in MemoryGenerationConfig.model_fields


def test_legacy_memory_generation_retry_config_is_pruned() -> None:
    config = MemoryGenerationConfig.model_validate({"queue_max_attempts": 3})

    assert "queue_max_attempts" not in config.model_dump()


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
