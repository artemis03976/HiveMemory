"""
TheEye 单元测试

测试覆盖:
- gaze: 正常 / fallback / identity 默认值 / active_topics_menu 传递

Note: 被动模式 observer session 管理测试已迁移至
    tests/unit/patchouli/test_passive_ingressor.py
"""

import pytest
from unittest.mock import AsyncMock, Mock

from hivememory.core.models import Identity
from hivememory.engines.gateway.models import GatewayIntent, GatewayResult
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.protocol.models import EyeGazeResult


def _make_identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1", session_id="s1")


def _make_gateway_result(**kwargs) -> GatewayResult:
    """构建 mock GatewayResult"""
    defaults = dict(
        intent=GatewayIntent.RAG,
        rewritten_query="重写查询",
        search_keywords=["kw1"],
        worth_saving=True,
        reason="有价值",
        target_topic="topic_001",
        new_topic_title=None,
        new_topic_summary=None,
        processing_time_ms=0.0,
    )
    defaults.update(kwargs)
    result = Mock(spec=GatewayResult)
    for k, v in defaults.items():
        setattr(result, k, v)
    return result


@pytest.mark.asyncio
class TestTheEyeGaze:
    """gaze() 方法测试"""

    def setup_method(self):
        self.mock_engine = Mock()
        self.mock_engine.process = AsyncMock()
        self.eye = TheEye(engine=self.mock_engine, bus=None)

    async def test_gaze_success(self):
        """正常调用 engine.process，返回 EyeGazeResult(is_fallback=False)"""
        self.mock_engine.process.return_value = _make_gateway_result()

        result = await self.eye.gaze("测试查询", identity=_make_identity())

        self.mock_engine.process.assert_called_once()
        assert isinstance(result, EyeGazeResult)
        assert result.is_fallback is False
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "重写查询"

    async def test_gaze_fallback_on_exception(self):
        """engine.process 抛异常时返回 fallback"""
        self.mock_engine.process.side_effect = RuntimeError("boom")

        result = await self.eye.gaze("测试查询", identity=_make_identity())

        assert result.is_fallback is True
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "测试查询"
        assert result.target_topic == "NEW_TOPIC"

    async def test_gaze_identity_default(self):
        """identity=None 时使用 Identity() 默认值"""
        self.mock_engine.process.return_value = _make_gateway_result()

        result = await self.eye.gaze("查询", identity=None)

        assert result.identity is not None

    async def test_gaze_forwards_active_topics_menu(self):
        """engine.process 接收 active_topics_menu 关键字参数"""
        self.mock_engine.process.return_value = _make_gateway_result()

        await self.eye.gaze(
            "查询",
            topic_snapshots=[],
            identity=_make_identity(),
        )

        call_args = self.mock_engine.process.call_args
        assert call_args[0][0] == "查询"
        assert call_args[1]["active_topics_menu"] is None
