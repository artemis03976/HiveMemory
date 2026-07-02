from hivememory.system.config import HiveMemoryConfig
from hivememory.system.config.memory_compiler import CompactContextStrategyConfig


def test_agent_runtime_config_has_loop_iteration_default():
    config = HiveMemoryConfig()

    assert config.alice.runtime.max_loop_iterations == 10


def test_runtime_events_config_has_defaults():
    config = HiveMemoryConfig()

    assert config.runtime_events.enabled is True
    assert config.runtime_events.buffer_size == 1000
    assert config.runtime_events.subscriber_queue_size == 100


def test_memory_compiler_config_has_retrieval_context_default():
    config = HiveMemoryConfig()

    assert isinstance(
        config.memory_compiler.retrieval_context.strategy,
        CompactContextStrategyConfig,
    )
    assert config.memory_compiler.retrieval_context.strategy.type == "compact"


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

    strategy = config.memory_compiler.retrieval_context.strategy
    assert strategy.type == "cascade"
    assert strategy.max_memory_tokens == 123
    assert strategy.full_payload_count == 2


def test_new_llm_env_vars_override_legacy_aliases(monkeypatch):
    monkeypatch.setenv("HIVEMEMORY__LLM__GATEWAY__API_KEY", "legacy-gateway-key")
    monkeypatch.setenv("HIVEMEMORY__SHARED__LLM__GATEWAY__API_KEY", "new-gateway-key")

    config = HiveMemoryConfig()

    assert config.shared.llm.gateway.api_key == "new-gateway-key"
