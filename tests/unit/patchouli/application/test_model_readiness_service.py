"""
ModelReadinessService 单元测试

测试覆盖:
- warmup_models: 模型预热（真实 handler 副作用）
"""

import pytest

from hivememory.patchouli.application.model_readiness_service import ModelReadinessService
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus


class TestModelReadinessService:
    """ModelReadinessService 测试套件"""

    @pytest.mark.asyncio
    async def test_warmup_models_invokes_registered_handler(self):
        bus = PatchouliBus()
        warmed = []

        async def warmup():
            warmed.append(True)

        bus.register(PatchouliLocalRoutes.RUNTIME_MODELS_WARMUP, warmup)
        service = ModelReadinessService(bus=bus)

        await service.warmup_models()

        assert warmed == [True]
