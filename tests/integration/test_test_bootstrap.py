"""测试启动配置与真实原生依赖的文件副作用隔离。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_bootstrap_prevents_onnx_telemetry_session_file(tmp_path: Path) -> None:
    """新进程加载测试配置后导入 Qdrant，退出也不能留下遥测会话文件。"""
    environment = os.environ.copy()
    # 不继承父 pytest 的禁用开关，确保真正验证子进程的启动配置。
    environment.pop("ORT_DISABLE_TELEMETRY", None)
    conftest = Path(__file__).resolve().parents[1] / "conftest.py"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1]); import qdrant_client",
            str(conftest),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # 原生 SDK 不走 Python open；必须等进程退出，再检查真实文件系统。
    assert not (tmp_path / ":memory:.ses").exists()
