"""主配置接口：读取期望配置，保存只把差异写进个人覆盖文件，立即生效的字段写入运行配置。"""

from unittest.mock import MagicMock

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.config import files as config_files
from hivememory.config.app import HiveMemoryConfig
from hivememory.i18n import get_default_language, set_default_language
from hivememory.server import deps
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.routers.config import router

QDRANT_KEY = "qdrant-secret-key-0003"
_CONFIG_YAML = "alice:\n  runtime:\n    max_loop_iterations: 4\n"


def _create_test_app(mock_system):
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_system] = lambda: mock_system
    return app


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """以 tmp_path 中的 config.yaml 作为主配置，个人覆盖文件在它旁边。"""
    (tmp_path / "config.yaml").write_text(_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(tmp_path / "config.yaml"))
    return tmp_path


@pytest.fixture
def system(config_dir):
    """运行配置按启动时的方式从文件装载。"""
    mock_system = MagicMock()
    mock_system.config = HiveMemoryConfig()
    return mock_system


@pytest.fixture
def client(system):
    return TestClient(_create_test_app(system), raise_server_exceptions=False)


def _overrides(config_dir):
    path = config_dir / "config.local.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None


def test_config_responses_return_only_masked_api_keys(config_dir, system, client, monkeypatch):
    """接口对本机任何进程开放：读取、保存与默认值的响应中都不出现明文密钥。"""
    monkeypatch.setenv("HIVEMEMORY__QDRANT__API_KEY", QDRANT_KEY)
    system.config = HiveMemoryConfig()

    read = client.get("/api/v1/config")
    saved = client.post("/api/v1/config", json=read.json()["config"])
    defaults = client.get("/api/v1/config/defaults")

    for response in (read, saved, defaults):
        assert response.status_code == 200
        assert QDRANT_KEY not in response.text
    assert read.json()["config"]["patchouli"]["storage"]["api_key"] == "qdr...0003"
    assert defaults.json()["patchouli"]["storage"]["api_key"] == "qdr...0003"


def test_saving_config_neither_changes_nor_persists_api_keys(config_dir, system, client):
    """密钥不经主配置接口修改：提交的值一律沿用运行时值，覆盖文件不含 api_key。"""
    system.config.patchouli.storage.api_key = QDRANT_KEY
    payload = client.get("/api/v1/config").json()["config"]
    payload["patchouli"]["storage"]["api_key"] = "new-qdrant-secret"
    payload["logging"]["level"] = "DEBUG"

    response = client.post("/api/v1/config", json=payload)

    assert response.status_code == 200
    assert system.config.patchouli.storage.api_key == QDRANT_KEY
    assert _overrides(config_dir) == {"logging": {"level": "DEBUG"}}


def test_overrides_hold_exactly_the_differences_from_defaults(config_dir, client):
    """覆盖文件只含与默认值不同的叶子；改回默认值的字段从中消失，提交默认值后文件被删除。

    默认值含仓库中的 config.yaml，它从不被写。
    """
    payload = client.get("/api/v1/config").json()["config"]
    payload["logging"]["level"] = "DEBUG"
    payload["alice"]["runtime"]["max_loop_iterations"] = 6

    assert client.post("/api/v1/config", json=payload).status_code == 200
    assert _overrides(config_dir) == {
        "logging": {"level": "DEBUG"},
        "alice": {"runtime": {"max_loop_iterations": 6}},
    }

    payload["logging"]["level"] = "INFO"
    assert client.post("/api/v1/config", json=payload).status_code == 200
    assert _overrides(config_dir) == {"alice": {"runtime": {"max_loop_iterations": 6}}}

    defaults = client.get("/api/v1/config/defaults").json()
    assert defaults["alice"]["runtime"]["max_loop_iterations"] == 4
    assert client.post("/api/v1/config", json=defaults).status_code == 200
    assert _overrides(config_dir) is None
    assert (config_dir / "config.yaml").read_text(encoding="utf-8") == _CONFIG_YAML


def test_saving_applies_immediate_fields_and_reports_the_rest_as_pending(system, client):
    """立即生效的字段写入运行配置；其余字段只进入覆盖文件，列为等待重启，运行中的进程默认语言不变。"""
    payload = client.get("/api/v1/config").json()["config"]
    payload["patchouli"]["retrieval"]["retriever"]["top_k"] = 8
    payload["i18n"]["default_language"] = "en"

    response = client.post("/api/v1/config", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["i18n"]["default_language"] == "en"
    assert "patchouli.retrieval.retriever.top_k" in body["immediate_fields"]
    assert body["pending_restart"] == ["i18n.default_language"]
    assert system.config.patchouli.retrieval.retriever.top_k == 8
    assert system.config.i18n.default_language == "zh"
    assert get_default_language() == "zh"
    assert client.get("/api/v1/config").json()["pending_restart"] == ["i18n.default_language"]


def test_update_config_validation_error_does_not_persist(config_dir, system, client):
    running = system.config

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
    assert system.config is running
    assert _overrides(config_dir) is None


def test_update_config_write_failure_keeps_runtime_config(config_dir, system, client, monkeypatch):
    """覆盖文件写入失败时不原地应用任何字段。"""

    def _raise_replace(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(config_files.os, "replace", _raise_replace)
    payload = client.get("/api/v1/config").json()["config"]
    payload["patchouli"]["retrieval"]["retriever"]["top_k"] = 8

    response = client.post("/api/v1/config", json=payload)

    assert response.status_code == 500
    assert response.json() == {
        "error": "server.internal_error",
        "detail": "服务器内部错误，请稍后再试",
    }
    assert system.config.patchouli.retrieval.retriever.top_k == 5
    assert _overrides(config_dir) is None


def test_default_config_query_keeps_process_default_language():
    set_default_language("en")
    client = TestClient(_create_test_app(MagicMock()))

    response = client.get("/api/v1/config/defaults")

    assert response.status_code == 200
    assert get_default_language() == "en"
