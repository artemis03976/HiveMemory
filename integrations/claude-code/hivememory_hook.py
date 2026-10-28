"""HiveMemory 连接器的启动脚本：由插件 hooks 以 ``python3`` 直接运行。

把插件根目录放到导入路径最前，再交给 ``connector`` 的命令行入口；连接器只用
标准库，不需要安装依赖。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from connector.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
