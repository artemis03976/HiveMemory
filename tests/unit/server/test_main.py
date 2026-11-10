"""Server 启动入口测试：线程异常钩子、默认启动方式与桌面模式。"""

import io
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from hivememory.server import __main__ as server_main
from hivememory.server import app as app_module
from hivememory.server.management_credential import is_desktop_mode


def _hook_args(thread):
    return SimpleNamespace(
        exc_type=RuntimeError,
        exc_value=RuntimeError("boom"),
        exc_traceback=None,
        thread=thread,
    )


def test_thread_exception_hook_logs_error(caplog):
    server_main._install_thread_exception_hook()

    thread = threading.Thread(name="worker-1")
    args = _hook_args(thread)

    with caplog.at_level(logging.ERROR, logger="hivememory.server"):
        threading.excepthook(args)

    assert any(
        "未捕获线程异常" in record.message and "worker-1" in record.message
        for record in caplog.records
    )


def test_thread_exception_hook_tolerates_missing_thread(caplog):
    server_main._install_thread_exception_hook()

    args = _hook_args(thread=None)

    with caplog.at_level(logging.ERROR, logger="hivememory.server"):
        threading.excepthook(args)

    assert any("未捕获线程异常" in record.message for record in caplog.records)


def test_main_runs_uvicorn_with_expected_args(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("hivememory.server.__main__.uvicorn.run", fake_run)
    monkeypatch.setattr("hivememory.server.__main__._install_thread_exception_hook", lambda: None)
    monkeypatch.delenv("HIVEMEMORY_BACKEND_HOST", raising=False)

    server_main.main([])

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "hivememory.server.app:app"
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8769
    assert kwargs["log_level"] == "info"


def test_main_installs_thread_hook_before_running_uvicorn(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append("uvicorn")

    monkeypatch.setattr("hivememory.server.__main__.uvicorn.run", fake_run)
    monkeypatch.setattr(
        "hivememory.server.__main__._install_thread_exception_hook",
        lambda: calls.append("hook"),
    )

    server_main.main([])

    assert calls == ["hook", "uvicorn"]


class _ServerUntilShutdownRequested:
    """代替 uvicorn.Server：运行到有人请求关闭为止，超时即失败。"""

    instances: list["_ServerUntilShutdownRequested"] = []

    def __init__(self, config):
        self.config = config
        self.should_exit = False
        self.instances.append(self)

    def run(self):
        deadline = time.monotonic() + 5
        while not self.should_exit:
            if time.monotonic() > deadline:
                raise AssertionError("stdin 结束之后没有请求关闭")
            time.sleep(0.01)


@pytest.fixture
def desktop_server(monkeypatch):
    """替换 uvicorn.Server，并在测试结束时撤销桌面模式写进真实应用的状态。"""
    _ServerUntilShutdownRequested.instances = []
    monkeypatch.setattr(server_main.uvicorn, "Server", _ServerUntilShutdownRequested)
    monkeypatch.setattr(server_main, "_install_thread_exception_hook", lambda: None)
    monkeypatch.setattr(app_module.app.state, "management_credential", None, raising=False)
    monkeypatch.setattr(app_module.app.state, "desktop_mode", False, raising=False)
    return _ServerUntilShutdownRequested.instances


def test_desktop_mode_serves_with_stdin_credential_until_stdin_closes(monkeypatch, desktop_server):
    monkeypatch.setattr("sys.stdin", io.StringIO("desktop-credential\n"))

    server_main.main(["--desktop"])

    [server] = desktop_server
    assert server.config.app is app_module.app
    assert server.config.port == 8769
    assert app_module.app.state.management_credential == "desktop-credential"
    assert is_desktop_mode(app_module.app)
    assert server.should_exit is True


def test_desktop_mode_without_credential_exits_before_serving(monkeypatch, desktop_server):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    with pytest.raises(SystemExit, match="管理凭据"):
        server_main.main(["--desktop"])

    assert desktop_server == []
    assert not is_desktop_mode(app_module.app)
