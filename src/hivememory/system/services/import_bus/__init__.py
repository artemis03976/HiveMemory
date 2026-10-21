"""
Import Bus 业务能力

system 层的 Import Bus 子模块，收口外部 harness 交互导入的编排、输入模型、
失败分类与观测：
    - ImportBusIngestor
    - ImportedTurn / ImportedTurnEvent / ImportedRecentTurn
    - ImportConversationKey / ImportTurnResult / derive_import_interaction_id
    - ImportBusUnavailableError / ImportBusContractError
    - ImportBusEventEmitter
"""

from hivememory.system.services.import_bus.errors import (
    ImportBusContractError,
    ImportBusUnavailableError,
    ImportUnavailableReason,
    is_recoverable_gateway_error,
)
from hivememory.system.services.import_bus.events import ImportBusEventEmitter
from hivememory.system.services.import_bus.ingestor import ImportBusIngestor
from hivememory.system.services.import_bus.models import (
    IMPORT_INTERACTION_PREFIX,
    ImportConversationKey,
    ImportedEventKind,
    ImportedRecentTurn,
    ImportedTurn,
    ImportedTurnEvent,
    ImportTurnResult,
    ImportTurnStatus,
    derive_import_interaction_id,
)

__all__ = [
    "IMPORT_INTERACTION_PREFIX",
    "ImportBusContractError",
    "ImportBusEventEmitter",
    "ImportBusIngestor",
    "ImportBusUnavailableError",
    "ImportConversationKey",
    "ImportTurnResult",
    "ImportTurnStatus",
    "ImportUnavailableReason",
    "ImportedEventKind",
    "ImportedRecentTurn",
    "ImportedTurn",
    "ImportedTurnEvent",
    "derive_import_interaction_id",
    "is_recoverable_gateway_error",
]
