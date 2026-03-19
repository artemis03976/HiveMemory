"""HiveMemory Server 启动入口"""

import logging
import threading

import uvicorn


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("apscheduler").setLevel(logging.INFO)
    logging.getLogger("hivememory").setLevel(logging.INFO)


def _install_thread_exception_hook() -> None:
    def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("hivememory.server").error(
            f"未捕获线程异常: thread={getattr(args.thread, 'name', 'unknown')}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
    threading.excepthook = _handle_thread_exception


def main():
    _configure_logging()
    _install_thread_exception_hook()
    uvicorn.run(
        "hivememory.server.app:app",
        host="0.0.0.0",
        port=8769,  # Custom port to avoid Windows reserved ranges
        log_level="info",
    )


if __name__ == "__main__":
    main()
