"""连接器测试的导入路径：连接器不在 hivememory 包内，按插件运行时的方式导入。"""

import sys
from pathlib import Path

CONNECTOR_ROOT = Path(__file__).resolve().parents[3] / "integrations" / "claude-code"

if str(CONNECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(CONNECTOR_ROOT))
