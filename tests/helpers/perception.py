"""Perception 相关测试的共享装配辅助。

统一构造 ``ShortTermMemoryStore + TopicBufferService +
SemanticFlowPerceptionLayer`` 三件套，保持测试与生产装配一致：
Layer 不再直接持有 Store，状态操作一律经由 TopicBufferService。
"""

from unittest.mock import Mock

from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.topic_buffer import TopicBufferService
from hivememory.system.config import SemanticFlowPerceptionConfig


def build_perception_stack(
    *,
    store: ShortTermMemoryStore | None = None,
    relay: Mock | None = None,
    config: SemanticFlowPerceptionConfig | None = None,
    journal: InMemoryInteractionApplyJournal | None = None,
) -> tuple[SemanticFlowPerceptionLayer, ShortTermMemoryStore, TopicBufferService]:
    """构造 (layer, store, service) 测试三元组。

    Args:
        store: 可选的 ShortTermMemoryStore（默认新建内存实例）
        relay: 可选的 Relay mock（默认 ``Mock()``，``should_relay`` 返回 None）
        config: 可选的感知配置（默认关闭自动 folding 的宽松阈值）
        journal: 可选的 interaction apply journal（默认新建）
    """
    store = store or ShortTermMemoryStore()
    relay = relay or Mock()
    if isinstance(relay, Mock):
        relay.should_relay.return_value = None
    config = config or SemanticFlowPerceptionConfig(fold_token_threshold=999999)
    service = TopicBufferService(store=store, relay_controller=relay)
    layer = SemanticFlowPerceptionLayer(
        config=config,
        relay_controller=relay,
        topic_buffer=service,
        interaction_journal=journal or InMemoryInteractionApplyJournal(),
    )
    return layer, store, service
