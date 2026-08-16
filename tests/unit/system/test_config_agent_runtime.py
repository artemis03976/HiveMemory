from hivememory.system.config import HiveMemoryConfig
from hivememory.system.config.memory_compiler import CascadeContextStrategyConfig


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
