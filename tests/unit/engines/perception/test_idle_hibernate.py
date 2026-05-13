"""
Idle Hibernate 单元测试

测试覆盖:
- 空闲 flush 后话题从活跃池 swap-out
- 空闲 flush 触发 generation_callback
- swap-out 后释放坑位，新话题可正常创建

参考: ShortTermMemory.md §5.1

Note:
    Phase 4.5 重构：使用 topic_id 替代 session_id
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock

from hivememory.core.models import Identity, TurnEvent
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.engines.perception.models import (
    FlushReason,
)
from hivememory.patchouli.config import SemanticFlowPerceptionConfig
from hivememory.patchouli.protocol import InteractionPayload


def _make_identity(user="u1", agent="a1"):
    return Identity(user_id=user, agent_id=agent)


def _make_payload(user_msg="hello", assistant_msg="world", identity=None):
    if identity is None:
        identity = _make_identity()
    return InteractionPayload(
        user_message=user_msg,
        assistant_final_text=assistant_msg,
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=assistant_msg,
            )
        ],
        identity=identity,
    )


class TestIdleHibernateSwapOut:
    """验证空闲超时后话题被 swap-out"""

    @pytest.mark.asyncio
    async def test_idle_flush_swaps_out_topic(self):
        """设置短超时，触发扫描，验证话题从活跃池移除"""
        config = SemanticFlowPerceptionConfig(
            idle_timeout_seconds=1,  # 1 秒超时
            fold_token_threshold=999999,
        )
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        layer = SemanticFlowPerceptionLayer(config=config, relay_controller=mock_relay)
        identity = _make_identity()

        # 路由到新话题并摄入
        await layer.route_and_ingest("NEW_TOPIC", _make_payload("question", "answer", identity))

        # 验证话题在活跃池中
        active = layer.list_active_buffers()
        assert len(active) == 1

        # 等待超时
        time.sleep(1.5)

        # 手动触发扫描（异步方法需要 await）
        flushed = await layer.scan_idle_buffers_once()

        # 验证话题已被 flush 并 swap-out
        assert len(flushed) == 1
        active_after = layer.list_active_buffers()
        assert len(active_after) == 0

    @pytest.mark.asyncio
    async def test_idle_flush_triggers_generation_callback(self):
        """验证空闲 flush 会触发 generation_callback"""
        config = SemanticFlowPerceptionConfig(
            idle_timeout_seconds=1,
            fold_token_threshold=999999,
        )
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        layer = SemanticFlowPerceptionLayer(config=config, relay_controller=mock_relay)
        mock_callback = AsyncMock(return_value=None)
        layer.set_generation_callback(mock_callback)

        identity = _make_identity()
        await layer.route_and_ingest("NEW_TOPIC", _make_payload("question", "answer", identity))

        time.sleep(1.5)
        await layer.scan_idle_buffers_once()
        await asyncio.sleep(0)

        mock_callback.assert_called()
        call_args = mock_callback.call_args
        payload = call_args[0][0]
        # 验证 payload 包含必要字段
        assert payload.topic_id is not None
        assert len(payload.blocks) > 0

    @pytest.mark.asyncio
    async def test_idle_flush_frees_slot(self):
        """swap-out 后坑位释放，新话题可正常创建"""
        config = SemanticFlowPerceptionConfig(
            idle_timeout_seconds=1,
            max_resident_topics=2,
            fold_token_threshold=999999,
        )
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        layer = SemanticFlowPerceptionLayer(config=config, relay_controller=mock_relay)

        # 填满 2 个话题坑位
        id1 = _make_identity("u1", "a1")
        id2 = _make_identity("u2", "a2")
        await layer.route_and_ingest("NEW_TOPIC", _make_payload("q1", "a1", id1))
        await layer.route_and_ingest("NEW_TOPIC", _make_payload("q2", "a2", id2))

        assert len(layer.list_active_buffers()) == 2

        # 等待超时并扫描
        time.sleep(1.5)
        flushed = await layer.scan_idle_buffers_once()
        assert len(flushed) == 2

        # 坑位已释放，新话题可正常创建（不触发 LRU 驱逐）
        id3 = _make_identity("u3", "a3")
        await layer.route_and_ingest("NEW_TOPIC", _make_payload("q3", "a3", id3))

        active = layer.list_active_buffers()
        assert len(active) == 1

    @pytest.mark.asyncio
    async def test_shutdown_flush_archives_and_swaps_out_all_topics(self):
        """验证 shutdown flush 会归档并驱逐所有活跃话题"""
        config = SemanticFlowPerceptionConfig(
            idle_timeout_seconds=999,
            fold_token_threshold=999999,
            max_resident_topics=4,
        )
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        layer = SemanticFlowPerceptionLayer(config=config, relay_controller=mock_relay)
        mock_callback = AsyncMock(return_value=None)
        layer.set_generation_callback(mock_callback)

        await layer.route_and_ingest("NEW_TOPIC", _make_payload("q1", "a1", _make_identity("u1", "a1")))
        await layer.route_and_ingest("NEW_TOPIC", _make_payload("q2", "a2", _make_identity("u2", "a2")))

        result = await layer.flush_all_for_shutdown()

        assert result["trigger_reason"] == FlushReason.SHUTDOWN.value
        assert len(result["flushed_topics"]) == 2
        assert result["archived_blocks"] == 2
        assert layer.list_active_buffers() == []
        assert mock_callback.await_count == 2

