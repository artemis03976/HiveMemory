from hivememory.config.app import HiveMemoryConfig
from hivememory.config.memory_compiler import CascadeContextStrategyConfig


def test_memory_compiler_config_accepts_cascade_strategy():
    config = HiveMemoryConfig(
        memory_compiler={
            "retrieval_context": {
                "strategy": {
                    "type": "cascade",
                    "max_memory_tokens": 123,
                    "full_payload_count": 2,
                }
            }
        }
    )

    # discriminated union 能解析 cascade 分支且字段落位
    strategy = config.memory_compiler.retrieval_context.strategy
    assert isinstance(strategy, CascadeContextStrategyConfig)
    assert strategy.type == "cascade"


def test_new_llm_env_vars_override_legacy_aliases(monkeypatch):
    monkeypatch.setenv("HIVEMEMORY__LLM__GATEWAY__API_KEY", "legacy-gateway-key")
    monkeypatch.setenv("HIVEMEMORY__SHARED__LLM__GATEWAY__API_KEY", "new-gateway-key")

    config = HiveMemoryConfig()

    assert config.shared.llm.gateway.api_key == "new-gateway-key"


def test_runtime_budget_fields_are_loaded_from_configuration(tmp_path, monkeypatch):
    """发布配置中的 runtime 段同时承载循环上限与输入预算参数。"""
    path = tmp_path / "runtime.yaml"
    path.write_text(
        "alice:\n  runtime:\n    max_loop_iterations: 4\n"
        "    default_context_window: 20000\n    safety_margin: 500\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(path))
    config = HiveMemoryConfig().alice.runtime

    assert config.max_loop_iterations == 4
    assert config.default_context_window == 20000
    assert config.safety_margin == 500
