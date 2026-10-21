"""
System 业务能力提供者

`services/` 收口 system 层内部那些拥有自己一套业务管理逻辑的子模块：
它们不是对外 API 的薄适配层，而是持有独立状态、时序契约与失败语义的业务能力。

层级定位：
    - services/ system 自己的业务能力提供者，是 system 的子模块。
    - application/  面向外部入口的应用服务，把一次请求翻译成一次系统调用，
                本身不持有跨请求的业务状态。

依赖方向单向：`application/` → `services/`，反向不允许。

当前能力：
    - `import_bus`  Import Bus（外部 harness 一轮完整交互的串行导入、查重、
      Gateway 落位与经 Patchouli 公开路由的提交）
"""

from hivememory.system.services.import_bus import (
    ImportBusIngestor,
    ImportConversationKey,
    ImportedTurn,
    ImportTurnResult,
)

__all__ = [
    "ImportBusIngestor",
    "ImportConversationKey",
    "ImportTurnResult",
    "ImportedTurn",
]
