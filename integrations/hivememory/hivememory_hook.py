"""HiveMemory 连接器的启动脚本，由插件的 hook 运行，第一个参数是 harness。

Claude Code 的 hook 在 bash 中找 ``python3``，没有时用 ``python``（Windows 的 Python
安装包只提供 ``python``，Claude Code 在 Windows 上以 Git Bash 运行 hook）；Codex
的 hook 直接运行 ``python``，命令文本中的插件根目录由 Codex 替换，不依赖 shell。

把插件根目录放到导入路径最前，再交给 ``connector`` 的命令行入口；连接器只用
标准库，不需要安装依赖。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from connector.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
