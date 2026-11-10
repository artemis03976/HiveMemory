"""HiveMemory Server 启动入口"""

import argparse
import logging
import os
import sys
import threading

import uvicorn

logger = logging.getLogger("hivememory.server")

# 自定义端口，避开 Windows 保留端口段；外部 harness 的插件都连这个端口。
BACKEND_PORT = 8769


def _install_thread_exception_hook() -> None:
    def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        exc_info = (
            (args.exc_type, args.exc_value, args.exc_traceback)
            if args.exc_value is not None
            else None
        )
        logger.error(
            "未捕获线程异常: thread=%s",
            getattr(args.thread, "name", "unknown"),
            exc_info=exc_info,
        )

    threading.excepthook = _handle_thread_exception


def _backend_host() -> str:
    # 默认只监听本机地址；需要对外监听时显式设置，变量与开发脚本同名。
    return os.getenv("HIVEMEMORY_BACKEND_HOST", "127.0.0.1")


def _run_desktop() -> None:
    """桌面模式：由桌面客户端的主进程启动，经 stdin 交来管理凭据，stdin 结束时正常关闭。

    stdin 的第一行是管理凭据，只留在内存中。Windows 上不能向另一个进程发送 SIGTERM，
    结束进程会跳过话题结算，所以主进程以关闭 stdin 请求关闭；主进程意外退出时管道同样
    关闭，后端不会成为没有人持有凭据的孤儿进程。
    """
    credential = sys.stdin.readline().strip()
    if not credential:
        raise SystemExit("桌面模式需要在 stdin 的第一行收到管理凭据")

    # 应用模块在导入时装配 MCP 入口；非桌面模式由 uvicorn 按模块路径导入它。
    from hivememory.server.app import app
    from hivememory.server.management_credential import use_desktop_credential

    use_desktop_credential(app, credential)
    server = uvicorn.Server(
        uvicorn.Config(app, host=_backend_host(), port=BACKEND_PORT, log_level="info")
    )
    threading.Thread(
        target=_shutdown_when_stdin_closes, args=(server,), name="desktop-stdin", daemon=True
    ).start()
    server.run()


def _shutdown_when_stdin_closes(server: uvicorn.Server) -> None:
    """读到 stdin 结束时请求正常关闭，与 Ctrl+C 走同一条路径，退出前结算内存中的话题。"""
    sys.stdin.read()
    logger.info("桌面主进程关闭了 stdin，开始正常关闭")
    server.should_exit = True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hivememory-server")
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="由桌面客户端启动：stdin 第一行是管理凭据，stdin 结束时正常关闭",
    )
    args = parser.parse_args(argv)

    _install_thread_exception_hook()
    if args.desktop:
        _run_desktop()
        return
    uvicorn.run(
        "hivememory.server.app:app",
        host=_backend_host(),
        port=BACKEND_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
