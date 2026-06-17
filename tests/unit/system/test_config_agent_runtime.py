from hivememory.system.config import HiveMemoryConfig


def test_agent_runtime_config_has_loop_iteration_default():
    config = HiveMemoryConfig()

    assert config.alice.runtime.max_loop_iterations == 10


def test_runtime_events_config_has_defaults():
    config = HiveMemoryConfig()

    assert config.runtime_events.enabled is True
    assert config.runtime_events.buffer_size == 1000
    assert config.runtime_events.subscriber_queue_size == 100


def test_legacy_llm_env_vars_map_to_shared_config(monkeypatch):
    monkeypatch.setenv("HIVEMEMORY__LLM__WORKER__MODEL", "legacy-worker-model")
    monkeypatch.setenv("HIVEMEMORY__LLM__WORKER__API_KEY", "legacy-worker-key")
    monkeypatch.setenv("HIVEMEMORY__LLM__WORKER__API_BASE", "https://legacy.example")
    monkeypatch.setenv("HIVEMEMORY__LLM__LIBRARIAN__API_KEY", "legacy-librarian-key")

    config = HiveMemoryConfig()

    assert config.shared.llm.worker.model == "legacy-worker-model"
    assert config.shared.llm.worker.api_key == "legacy-worker-key"
    assert config.shared.llm.worker.api_base == "https://legacy.example"
    assert config.shared.llm.librarian.api_key == "legacy-librarian-key"


def test_new_llm_env_vars_override_legacy_aliases(monkeypatch):
    monkeypatch.setenv("HIVEMEMORY__LLM__GATEWAY__API_KEY", "legacy-gateway-key")
    monkeypatch.setenv("HIVEMEMORY__SHARED__LLM__GATEWAY__API_KEY", "new-gateway-key")

    config = HiveMemoryConfig()

    assert config.shared.llm.gateway.api_key == "new-gateway-key"


def test_legacy_llm_env_vars_load_from_dotenv_file(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        """
shared:
  llm:
    worker:
      model: yaml-worker-model
      api_key: null
      api_base: null
""",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "HIVEMEMORY__LLM__WORKER__API_KEY=dotenv-worker-key",
                "HIVEMEMORY__LLM__WORKER__API_BASE=https://dotenv.example",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(config_path))

    config = HiveMemoryConfig()

    assert config.shared.llm.worker.model == "yaml-worker-model"
    assert config.shared.llm.worker.api_key == "dotenv-worker-key"
    assert config.shared.llm.worker.api_base == "https://dotenv.example"
