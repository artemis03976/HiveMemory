"""
ModelReadinessService 单元测试

测试覆盖:
- warmup_models: 模型预热
- is_models_ready: 检查模型是否就绪
"""

import pytest
from unittest.mock import AsyncMock, Mock

from hivememory.patchouli.application.model_readiness_service import ModelReadinessService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes


class TestModelReadinessService:
    """ModelReadinessService 测试套件"""

    @pytest.mark.asyncio
    async def test_warmup_models_delegates_to_bus(self):
        bus = Mock()
        bus.request = AsyncMock()

        service = ModelReadinessService(bus=bus)
        await service.warmup_models()

        bus.request.assert_awaited_once_with(PatchouliLocalRoutes.RUNTIME_MODELS_WARMUP)

    @pytest.mark.asyncio
    async def test_is_models_ready_delegates_to_bus(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=True)

        service = ModelReadinessService(bus=bus)
        result = await service.is_models_ready()

        assert result is True
        bus.request.assert_awaited_once_with(PatchouliLocalRoutes.RUNTIME_MODELS_READY)

    @pytest.mark.asyncio
    async def test_is_models_ready_returns_false_when_not_ready(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=False)

        service = ModelReadinessService(bus=bus)
        result = await service.is_models_ready()

        assert result is False
        bus.request.assert_awaited_once_with(PatchouliLocalRoutes.RUNTIME_MODELS_READY)
