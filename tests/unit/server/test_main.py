"""Server 启动入口测试。"""

import logging
import threading
from types import SimpleNamespace

from hivememory.server import __main__ as server_main


def _hook_args(thread):
    return SimpleNamespace(
        exc_type=RuntimeError,
        exc_value=RuntimeError("boom"),
        exc_traceback=None,
        thread=thread,
    )


def test_configure_logging_sets_levels():
    server_main._configure_logging()
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("hivememory").level == logging.INFO


def test_install_thread_exception_hook():
    server_main._install_thread_exception_hook()
    assert threading.excepthook is not None


def test_thread_exception_hook_logs_error(caplog):
    server_main._install_thread_exception_hook()

    thread = threading.Thread(name="worker-1")
    args = _hook_args(thread)

    with caplog.at_level(logging.ERROR, logger="hivememory.server"):
        threading.excepthook(args)

    assert any(
        "未捕获线程异常" in record.message
        and "worker-1" in record.message
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
    monkeypatch.setattr(
        "hivememory.server.__main__._configure_logging", lambda: None
    )
    monkeypatch.setattr(
        "hivememory.server.__main__._install_thread_exception_hook", lambda: None
    )

    server_main.main()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "hivememory.server.app:app"
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8769
    assert kwargs["log_level"] == "info"


def test_main_runs_setup_helpers_in_order(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append("uvicorn")

    monkeypatch.setattr("hivememory.server.__main__.uvicorn.run", fake_run)
    monkeypatch.setattr(
        "hivememory.server.__main__._configure_logging",
        lambda: calls.append("logging"),
    )
    monkeypatch.setattr(
        "hivememory.server.__main__._install_thread_exception_hook",
        lambda: calls.append("hook"),
    )

    server_main.main()

    assert calls == ["logging", "hook", "uvicorn"]
