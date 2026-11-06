"""HiveMemory Server 启动入口"""

import logging
import os
import threading

import uvicorn


def _install_thread_exception_hook() -> None:
    def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        exc_info = (
            (args.exc_type, args.exc_value, args.exc_traceback)
            if args.exc_value is not None
            else None
        )
        logging.getLogger("hivememory.server").error(
            f"未捕获线程异常: thread={getattr(args.thread, 'name', 'unknown')}",
            exc_info=exc_info,
        )

    threading.excepthook = _handle_thread_exception


def main():
    _install_thread_exception_hook()
    uvicorn.run(
        "hivememory.server.app:app",
        # 默认只监听本机地址；需要对外监听时显式设置，变量与开发脚本同名。
        host=os.getenv("HIVEMEMORY_BACKEND_HOST", "127.0.0.1"),
        port=8769,  # 自定义端口，避开 Windows 保留端口段
        log_level="info",
    )


if __name__ == "__main__":
    main()
