"""Patchouli 馆长核心兼容壳。"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LibrarianCore:
    """过渡期兼容对象；生成、感知、生命周期职责已迁出。"""

    def __init__(
        self,
        bus: Optional[Any] = None,
    ) -> None:
        self._bus = bus
        logger.info("LibrarianCore 兼容壳初始化完成")


__all__ = [
    "LibrarianCore",
]
