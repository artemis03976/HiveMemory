"""配置文件的位置、写入与从模板生成：访问登记、模型注册表与主配置覆盖文件共用。

主配置的默认值 ``config.yaml`` 与各个模板随程序发布；本机的个人配置不受版本控制，
放在数据根下的 ``configs`` 目录，生效文件不存在时从模板复制一份。写入都先写同目录的
临时文件再原子替换，写入中途失败不会留下半截文件。
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_ROOT_ENV",
    "SHIPPED_CONFIGS_DIR",
    "personal_configs_dir",
    "seed_from_template",
    "write_atomically",
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


def seed_from_template(path: Path, template: Path) -> None:
    """把模板原样复制为生效文件（保留注释）；模板缺失时显式失败。"""
    write_atomically(path, template.read_bytes())
    logger.info("已从模板生成配置文件: path=%s, template=%s", path, template.name)
