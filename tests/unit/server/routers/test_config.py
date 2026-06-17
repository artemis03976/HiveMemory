import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import hivememory.server.routers.config as config_router_module
from hivememory.system.config import HiveMemoryConfig
from hivememory.server import deps
from hivememory.server.routers.config import router


def _create_test_app(mock_system):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_system] = lambda: mock_system
    return app


def test_update_config_persists_to_local_file(tmp_path, monkeypatch):
    old_config = HiveMemoryConfig()
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(config_path))
    mock_system = MagicMock()
    mock_system.config = old_config

    app = _create_test_app(mock_system)
    client = TestClient(app)

    payload = old_config.model_dump(mode="json")
    payload["system"]["debug"] = not payload["system"]["debug"]

    response = client.post("/api/v1/config", json=payload)
    assert response.status_code == 200

    assert mock_system.config.system.debug == payload["system"]["debug"]
    assert config_path.exists()
    with open(config_path, "r", encoding="utf-8") as f:
        persisted = yaml.safe_load(f)
    assert persisted["system"]["debug"] == payload["system"]["debug"]


def test_update_config_validation_error_does_not_persist(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(config_path))

    existing_data = {"system": {"debug": True}}
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing_data, f, allow_unicode=True, sort_keys=False)

    old_config = HiveMemoryConfig()
    mock_system = MagicMock()
    mock_system.config = old_config

    app = _create_test_app(mock_system)
    client = TestClient(app)

    response = client.post("/api/v1/config", json={"patchouli": {"storage": {"port": "invalid-port"}}})
    assert response.status_code == 400
    assert mock_system.config is old_config
    with open(config_path, "r", encoding="utf-8") as f:
        persisted = yaml.safe_load(f)
    assert persisted == existing_data


def test_update_config_write_failure_keeps_runtime_config(tmp_path, monkeypatch):
    old_config = HiveMemoryConfig()
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(config_path))
    mock_system = MagicMock()
    mock_system.config = old_config

    def _raise_replace(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(config_router_module.os, "replace", _raise_replace)

    app = _create_test_app(mock_system)
    client = TestClient(app)

    payload = old_config.model_dump(mode="json")
    payload["system"]["debug"] = not payload["system"]["debug"]

    response = client.post("/api/v1/config", json=payload)
    assert response.status_code == 500
    assert mock_system.config is old_config
