"""配置文件的写入与从模板生成：访问登记、模型注册表与主配置覆盖文件共用。

本机的个人配置不受版本控制，仓库只发布同目录的模板；生效文件不存在时从模板
复制一份。写入都先写同目录的临时文件再原子替换，写入中途失败不会留下半截文件。
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["seed_from_template", "write_atomically"]


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


def seed_from_template(path: Path, template: Path) -> None:
    """把模板原样复制为生效文件（保留注释）；模板缺失时显式失败。"""
    write_atomically(path, template.read_bytes())
    logger.info("已从模板生成配置文件: path=%s, template=%s", path, template.name)
