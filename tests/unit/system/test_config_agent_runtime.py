from hivememory.system.config import HiveMemoryConfig


def test_agent_runtime_config_has_loop_iteration_default():
    config = HiveMemoryConfig()

    assert config.agent_runtime.max_loop_iterations == 10
