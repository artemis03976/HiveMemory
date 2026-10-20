from unittest.mock import MagicMock

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import hivememory.server.routers.config as config_router_module
from hivememory.config.app import HiveMemoryConfig
from hivememory.i18n import get_default_language, set_default_language
from hivememory.server import deps
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.routers.config import router


def _create_test_app(mock_system):
    app = FastAPI()
    install_error_handlers(app)
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
    client = TestClient(app, raise_server_exceptions=False)

    payload = old_config.model_dump(mode="json")
    payload["system"]["debug"] = not payload["system"]["debug"]

    response = client.post("/api/v1/config", json=payload)
    assert response.status_code == 200

    assert mock_system.config.system.debug == payload["system"]["debug"]
    assert config_path.exists()
    with open(config_path, encoding="utf-8") as f:
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
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/v1/config", json={"patchouli": {"storage": {"port": "invalid-port"}}}
    )
    assert response.status_code == 400
    assert response.json() == {
        "error": "request.validation_failed",
        "detail": "配置输入不符合要求",
        "issues": [
            {
                "loc": ["patchouli", "storage", "port"],
                "type": "int_parsing",
                "message": "应提供整数",
            }
        ],
    }
    assert mock_system.config is old_config
    with open(config_path, encoding="utf-8") as f:
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
    client = TestClient(app, raise_server_exceptions=False)

    payload = old_config.model_dump(mode="json")
    payload["system"]["debug"] = not payload["system"]["debug"]

    response = client.post("/api/v1/config", json=payload)
    assert response.status_code == 500
    assert response.json() == {
        "error": "server.internal_error",
        "detail": "服务器内部错误，请稍后再试",
    }
    assert mock_system.config is old_config


def test_default_config_query_keeps_process_default_language():
    set_default_language("en")
    client = TestClient(_create_test_app(MagicMock()))

    response = client.get("/api/v1/config/defaults")

    assert response.status_code == 200
    assert get_default_language() == "en"


def test_update_config_saves_language_without_changing_running_default(tmp_path, monkeypatch):
    """保存的默认语言在下次装配时生效，运行中的进程默认语言保持不变。"""
    old_config = HiveMemoryConfig()
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(tmp_path / "config.yaml"))
    mock_system = MagicMock()
    mock_system.config = old_config
    payload = old_config.model_dump(mode="json")
    payload["i18n"]["default_language"] = "en"

    response = TestClient(_create_test_app(mock_system)).post("/api/v1/config", json=payload)

    assert response.status_code == 200
    assert mock_system.config.i18n.default_language == "en"
    assert get_default_language() == "zh"
