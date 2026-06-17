from hivememory.system.config import HiveMemoryConfig


def test_agent_runtime_config_has_loop_iteration_default():
    config = HiveMemoryConfig()

    assert config.alice.runtime.max_loop_iterations == 10


def test_runtime_events_config_has_defaults():
    config = HiveMemoryConfig()

    assert config.runtime_events.enabled is True
    assert config.runtime_events.buffer_size == 1000
    assert config.runtime_events.subscriber_queue_size == 100
