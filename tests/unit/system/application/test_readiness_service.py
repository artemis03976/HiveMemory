"""SystemReadinessService 委托测试。"""

from unittest.mock import AsyncMock

import pytest

from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class TestSystemReadinessService:
    @pytest.mark.asyncio
    async def test_warmup_models_uses_public_route(self):
        bus = GlobalSystemBus()
        handler = AsyncMock()
        bus.register(GlobalRoutes.PATCHOULI_WARMUP_MODELS, handler)
        service = SystemReadinessService(global_bus=bus)

        await service.warmup_models()

        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_readiness_maps_models_ready_state(self):
        bus = GlobalSystemBus()
        handler = AsyncMock(return_value=True)
        bus.register(GlobalRoutes.PATCHOULI_MODELS_READY, handler)
        service = SystemReadinessService(global_bus=bus)

        assert await service.is_models_ready() is True
        assert await service.readiness() == {"status": "ready", "models_ready": True}

        # 未就绪分支：生产代码应映射为 warming_up
        handler.return_value = False
        assert await service.is_models_ready() is False
        assert await service.readiness() == {
            "status": "warming_up",
            "models_ready": False,
        }



