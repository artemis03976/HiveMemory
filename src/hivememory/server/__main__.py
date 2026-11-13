"""HiveMemory Server 启动入口"""

import argparse
import logging
import os
import shutil
import sys
import threading
from pathlib import Path

import uvicorn

from hivememory.config.app import HiveMemoryConfig, load_app_config

logger = logging.getLogger("hivememory.server")

# 自定义端口，避开 Windows 保留端口段；外部 harness 的插件都连这个端口。
DEFAULT_BACKEND_PORT = 8769


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


def _backend_port() -> int:
    # 开发实例与安装版同时运行时换用另一个端口；变量与开发脚本同名。
    return int(os.getenv("HIVEMEMORY_BACKEND_PORT", str(DEFAULT_BACKEND_PORT)))


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
        uvicorn.Config(app, host=_backend_host(), port=_backend_port(), log_level="info")
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


def _delete_memory_data(config: HiveMemoryConfig) -> list[Path]:
    """删除后端拥有的记忆数据，返回实际删除的路径。

    包括 Artifact、长期归档、关闭时写盘的三份文件（保存的生成工作、写入句柄登记、生成
    任务历史，各含读不回的 rejected 文件）与预设 Agent 的写入标记：句柄与任务都指向
    记忆库中的内容，记忆删除后不能留下；删除标记后，下次启动重新写入预设 Agent。由
    桌面端在关闭后端与 Qdrant 之后调用，记忆库的 Qdrant 存储由桌面端删除。配置、密钥、
    接入凭据、模型缓存与日志保留。相对路径按工作目录（数据根）解析；只删这些路径，不删
    数据根本身，数据根指向仓库 checkout 时仓库文件不受影响。
    """
    patchouli = config.patchouli
    shutdown_files = [
        Path(patchouli.shutdown.pending_work_file),
        Path(config.workspace.write_intents.record_file),
        Path(patchouli.shutdown.task_history_file),
    ]
    targets = [
        Path(patchouli.artifacts.root_dir),
        Path(patchouli.lifecycle.archiver.archive_dir),
        Path(config.system.preset_agent_profiles_marker),
    ]
    for path in shutdown_files:
        targets.append(path)
        targets.extend(path.parent.glob(f"{path.stem}.rejected-*{path.suffix}"))
    deleted: list[Path] = []
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        else:
            continue
        deleted.append(target)
    return deleted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hivememory-server")
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="由桌面客户端启动：stdin 第一行是管理凭据，stdin 结束时正常关闭",
    )
    parser.add_argument(
        "--delete-memory-data",
        action="store_true",
        help=(
            "删除记忆数据后退出：Artifact、长期归档、保存的生成工作、写入句柄、生成任务历史"
            "与预设 Agent 标记"
        ),
    )
    args = parser.parse_args(argv)

    if args.delete_memory_data:
        for path in _delete_memory_data(load_app_config()):
            print(f"已删除: {path}")
        return
    _install_thread_exception_hook()
    if args.desktop:
        _run_desktop()
        return
    uvicorn.run(
        "hivememory.server.app:app",
        host=_backend_host(),
        port=_backend_port(),
        log_level="info",
    )


if __name__ == "__main__":
    main()
