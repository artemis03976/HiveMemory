"""配置文件的位置、写入与从模板生成：访问登记、模型注册表与主配置覆盖文件共用。

主配置的默认值 ``config.yaml`` 与各个模板随程序发布；本机的个人配置不受版本控制，
放在数据根下的 ``configs`` 目录，生效文件不存在时从模板复制一份。写入都先写同目录的
临时文件再原子替换，写入中途失败不会留下半截文件。

关闭时写盘、启动时读回的数据文件（保存的生成工作、写入句柄登记、生成任务历史）共用
一种格式：JSON 对象带 ``format_version`` 与一个条目列表，经
``write_json_items`` 写出、``take_json_items`` 读回即删除，读不回时整个文件按
``rejected_path`` 改名保留。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, overload

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_ROOT_ENV",
    "SHIPPED_CONFIGS_DIR",
    "UnreadableFileError",
    "personal_configs_dir",
    "rejected_path",
    "seed_from_template",
    "take_json_items",
    "write_atomically",
    "write_json_items",
]

# 随程序发布的配置目录：src/hivememory/config/files.py → 程序根目录 / configs。
# 开发时程序根目录就是仓库根目录。
SHIPPED_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"

# 数据根：本机的个人配置、记忆库、模型缓存与日志所在的目录，桌面客户端把它设为用户选择的
# 目录。数据根之下的布局与仓库根目录相同。
DATA_ROOT_ENV = "HIVEMEMORY_DATA_ROOT"


def personal_configs_dir() -> Path:
    """本机个人配置文件所在的目录：数据根下的 ``configs``。

    没有指定数据根时，数据根就是程序根目录，个人配置与随程序发布的配置同目录。
    ``.env`` 与相对路径的数据目录（模型缓存、归档、Artifact）按工作目录解析，
    指定数据根的进程以数据根为工作目录启动。
    """
    data_root = os.getenv(DATA_ROOT_ENV)
    if data_root:
        return Path(data_root) / "configs"
    return SHIPPED_CONFIGS_DIR


def write_atomically(path: Path, content: bytes) -> None:
    """把 ``content`` 原子写入 ``path``。

    POSIX 上临时文件只有当前用户可读写，替换后沿用这一权限；Windows 沿用目录的
    ACL。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}_", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def rejected_path(path: Path) -> Path:
    """读不回的数据文件改名保留的位置：同目录，文件名带 ``rejected-`` 与 UTC 时间戳。"""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return path.with_name(f"{path.stem}.rejected-{stamp}{path.suffix}")


class UnreadableFileError(Exception):
    """数据文件无法解析、格式版本不认识、没有条目列表或条目读不回。

    原文件已改名为 ``rejected_path``，留在原目录供排查。
    """

    def __init__(self, rejected_path: Path) -> None:
        super().__init__(f"文件无法读回，已改存为 {rejected_path.name}")
        self.rejected_path = rejected_path


def write_json_items(path: Path, *, format_version: int, key: str, items: Sequence[object]) -> None:
    """把条目列表连同格式版本写成 JSON 对象，原子写入 ``path``。

    磁盘错误抛出 ``OSError``；文本中有不能编码为 UTF-8 的字符（例如半个代理对）时抛出
    ``UnicodeEncodeError``。两种情况都不留下文件。
    """
    document = {"format_version": format_version, key: list(items)}
    write_atomically(path, json.dumps(document, ensure_ascii=False).encode("utf-8"))


@overload
def take_json_items(path: Path, *, format_version: int, key: str) -> list[object] | None: ...


@overload
def take_json_items[T](
    path: Path, *, format_version: int, key: str, decode: Callable[[Any], T]
) -> list[T] | None: ...


def take_json_items(
    path: Path,
    *,
    format_version: int,
    key: str,
    decode: Callable[[Any], Any] | None = None,
) -> list[Any] | None:
    """读出并删除 ``path``，返回 ``key`` 下的条目；文件不存在时返回 ``None``。

    先删除再交给调用方：调用方处理条目的中途崩溃时这些条目丢失，但下一次启动不会再读回
    同一份文件、把处理过的条目再处理一次。给出 ``decode`` 时，条目在删除文件之前逐个
    解码，以 ``ValueError``、``KeyError`` 或 ``TypeError`` 表示读不回：任一条目读不回，
    整份文件都不读回。文件无法解析、格式版本不认识、没有条目列表或条目读不回时，整个文件
    改名为同目录的 rejected 文件，抛出 :class:`UnreadableFileError`。
    """
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("document must be an object")
        if document.get("format_version") != format_version:
            raise ValueError(f"unknown format version: {document.get('format_version')!r}")
        items = document.get(key)
        if not isinstance(items, list):
            raise ValueError(f"document has no {key!r} list")
        decoded = items if decode is None else [decode(item) for item in items]
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as error:
        rejected = rejected_path(path)
        path.replace(rejected)
        raise UnreadableFileError(rejected) from error
    path.unlink()
    return decoded


def seed_from_template(path: Path, template: Path) -> None:
    """把模板原样复制为生效文件（保留注释）；模板缺失时显式失败。"""
    write_atomically(path, template.read_bytes())
    logger.info("已从模板生成配置文件: path=%s, template=%s", path, template.name)
