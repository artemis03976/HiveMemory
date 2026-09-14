"""
AliceSystem 单元测试（unit 保留集）

系统 facade 对全局总线路由挂载/卸载的真实协作测试已迁移至
tests/integration/alice/test_system.py。本文件保留 health 报告测试。
"""

import pytest

from hivememory.alice.system import AliceSystem
from hivememory.system.config import HiveMemoryConfig


@pytest.mark.asyncio
async def test_health_reports_runtime_health():
    system = AliceSystem(config=HiveMemoryConfig())

    health = await system.health()

    assert health["runtime"]["koakuma_runtime"]["status"] == "ok"
    assert "agent_runtime" in health["runtime"]
