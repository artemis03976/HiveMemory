import pytest
from unittest.mock import AsyncMock, Mock

from hivememory.core.models import Identity, TopicSnapshot
from hivememory.core.protocol.models import EyeGazeResult
from hivememory.engines.gateway.models import GatewayIntent, GatewayResult
from hivememory.system.gateway.eye import TheEye


def _make_identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1", session_id="s1")


def _make_gateway_result(**kwargs) -> GatewayResult:
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
    for key, value in defaults.items():
        setattr(result, key, value)
    return result


@pytest.mark.asyncio
class TestTheEyeGaze:
    def setup_method(self):
        self.mock_engine = Mock()
        self.mock_engine.process = AsyncMock()
        self.eye = TheEye(engine=self.mock_engine)

    async def test_gaze_success(self):
        self.mock_engine.process.return_value = _make_gateway_result()

        result = await self.eye.gaze("测试查询", identity=_make_identity())

        self.mock_engine.process.assert_called_once()
        assert isinstance(result, EyeGazeResult)
        assert result.is_fallback is False
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "重写查询"

    async def test_gaze_fallback_on_exception(self):
        self.mock_engine.process.side_effect = RuntimeError("boom")

        result = await self.eye.gaze("测试查询", identity=_make_identity())

        assert result.is_fallback is True
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "测试查询"
        assert result.target_topic == "NEW_TOPIC"

    async def test_gaze_identity_default(self):
        self.mock_engine.process.return_value = _make_gateway_result()

        result = await self.eye.gaze("查询", identity=None)

        assert result.identity is not None

    async def test_gaze_forwards_active_topics_menu(self):
        self.mock_engine.process.return_value = _make_gateway_result()
        snapshots = [
            TopicSnapshot(
                topic_id="topic-1",
                topic_title="测试话题",
                state_summary="正在调试",
                last_turn={"user": "上一句", "assistant": "上一答"},
            )
        ]

        await self.eye.gaze(
            "查询",
            topic_snapshots=snapshots,
            identity=_make_identity(),
        )

        call_args = self.mock_engine.process.call_args
        assert call_args.args[0] == "查询"
        assert "topic-1: 测试话题" in call_args.kwargs["active_topics_menu"]
        assert "状态: 正在调试" in call_args.kwargs["active_topics_menu"]
